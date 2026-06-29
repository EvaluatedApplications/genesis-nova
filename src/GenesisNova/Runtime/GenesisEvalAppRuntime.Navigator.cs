using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Runtime;

/// <summary>The outcome of ONE light navigator-training cycle (a few sampled queries + a few gradient steps on the live
/// space): the final-epoch cross-entropy <see cref="Loss"/>, how many <see cref="Queries"/> were sampled this cycle, and
/// the on-policy probe split — <see cref="ResolvePct"/> (confident halt on the cued ancestor) vs <see cref="AbstainPct"/>
/// (no confident halt inside the step budget). The gym logs these so an overnight run shows the walker improving.</summary>
public readonly record struct NavTrainCycleResult(double Loss, int Queries, double ResolvePct, double AbstainPct)
{
    public static NavTrainCycleResult Empty => new(0.0, 0, 0.0, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  PER-CYCLE NAVIGATOR TRAINING ON THE LIVE SPACE  (PLATONIC_NAVIGATOR.md §7; the gym loop calls this once per cycle so
//  the shared query-conditioned policy net accumulates overnight, alongside the gym's model training).
//
//  Each call is a LIGHT step (it runs EVERY gym cycle, so it must be cheap): sample a handful of leaf concepts that have
//  relational ancestors, derive the cue from the LIVE relation graph's depth (immediate parent = GENUS, a 2+-hop
//  ancestor = DOMAIN, the top = ROOT — NO hardcoded taxonomy), distil the flow-field oracle into BC trajectories, run a
//  few Adam steps on State.Navigator, probe whether the freshly-trained queries resolve, and fold every traversed
//  concept back into the engine's persistent self (the vital loop). Serialized with REPL/model-train/save via the SAME
//  _modelOpsGate, bounded work per call, and CUDA-OOM-safe (falls back to CPU for the step rather than crashing the gym).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
public sealed partial class GenesisEvalAppRuntime
{
    // Light per-cycle budget knobs — kept small so the navigator step never dominates a gym cycle.
    private const int NavSampleMembers = 8;   // leaf concepts sampled per cycle (each yields up to 3 cued queries)
    private const int NavBcEpochs = 3;        // a few gradient passes — a light step, NOT a full train
    private const int NavEvalWalks = 6;       // on-policy probes for the resolve%/abstain% signal
    private const int NavMaxSteps = 8;        // the walk's cognitive light-cone for the probe
    private const int NavK = NavQueryDaggerTrainer.DefaultK;
    private const double NavLr = 1e-3;

    private int _navCycle;                              // rotates the sampling window so the whole space is covered over time
    private NavTrainCycleResult _lastNavTrain = NavTrainCycleResult.Empty;
    private double _navResolveEma;                      // running resolve% (EMA) for a later inspect tab
    private bool _navResolveSeeded;

    /// <summary>The most recent navigator-training cycle result (for a later inspect tab / diagnostics).</summary>
    public NavTrainCycleResult LastNavTrain => _lastNavTrain;

    /// <summary>A smoothed running resolve% over navigator-training cycles (EMA) — the overnight "is the walker getting
    /// better" trend a later inspect tab can read.</summary>
    public double NavResolveRunningPct => _navResolveEma;

    /// <summary>
    /// Train the shared platonic navigator for ONE light cycle on the LIVE space. Serialized with model-train / REPL /
    /// save via the model gate, bounded, and GPU-OOM-safe. Returns the cycle metrics (and stashes them for inspection).
    /// A flat or empty space (no concept with a more-general neighbour yet) is a no-op returning <see
    /// cref="NavTrainCycleResult.Empty"/> — expected early in a run, before the space has grown a taxonomy.
    /// </summary>
    public NavTrainCycleResult TrainNavigatorCycle(int maxMembers = NavSampleMembers, int epochs = NavBcEpochs)
        => WithModelGate(() => TrainNavigatorCycleLocked(maxMembers, epochs));

    private NavTrainCycleResult TrainNavigatorCycleLocked(int maxMembers, int epochs)
    {
        // The navigator walks the DialecticalSpace's relation graph; the legacy substrate exposes no ancestor depth, so
        // there is nothing to train on there.
        if (_state.Memory is not DialecticalSpace ds)
            return _lastNavTrain = NavTrainCycleResult.Empty;

        List<NavQueryDaggerTrainer.Query> queries;
        try { queries = SampleNavigatorQueries(ds, maxMembers); }
        catch { return _lastNavTrain = NavTrainCycleResult.Empty; }
        if (queries.Count == 0)
            return _lastNavTrain = NavTrainCycleResult.Empty;

        IReadOnlyList<NavQueryTrajectory> trajectories;
        try { trajectories = NavQueryDaggerTrainer.BuildQueryTrajectories(ds, queries, NavK); }
        catch { return _lastNavTrain = NavTrainCycleResult.Empty; }
        if (trajectories.Count == 0)
            return _lastNavTrain = NavTrainCycleResult.Empty;

        var net = _state.Navigator;
        var device = cuda.is_available() ? CUDA : CPU;

        NavTrainLosses losses;
        try
        {
            losses = NavQueryDaggerTrainer.TrainQuery(net, trajectories, Math.Max(1, epochs), NavLr, device, NavK);
        }
        catch (Exception ex) when (device.type == DeviceType.CUDA && IsGpuMemoryError(ex))
        {
            // The gym model is already resident on the GPU; if the navigator can't fit, DON'T crash the overnight run —
            // fall back to CPU for this one step and log it. (Bounded retry: a CPU OOM here would be a real failure.)
            Console.WriteLine($"[nav] CUDA alloc failed ({ex.GetType().Name}) — training this cycle on CPU: {ex.Message}");
            try { net.to(CPU); } catch { }
            device = CPU;
            losses = NavQueryDaggerTrainer.TrainQuery(net, trajectories, Math.Max(1, epochs), NavLr, CPU, NavK);
        }

        // Cheap on-policy probe + the vital loop: do the freshly-trained queries RESOLVE, and fold every traversed
        // concept into the engine's persistent self.
        var (resolvePct, abstainPct) = EvaluateAndFoldSelf(ds, net, queries, device);

        // Keep the net at REST on CPU between cycles — frees GPU memory for the gym model (overnight-safe) and makes the
        // persister's CPU weight export cheap. The move is ~tens of ms for this thin controller.
        try { if (device.type == DeviceType.CUDA) net.to(CPU); } catch { }

        var result = new NavTrainCycleResult(losses.CrossEntropy, queries.Count, resolvePct, abstainPct);
        _lastNavTrain = result;
        _navResolveEma = _navResolveSeeded ? 0.7 * _navResolveEma + 0.3 * resolvePct : resolvePct;
        _navResolveSeeded = true;
        return result;
    }

    /// <summary>Sample leaf concepts that HAVE relational ancestors and turn each into up to three cued queries, deriving
    /// the cue from the LIVE relation graph's depth (immediate parent = GENUS, 2-hop ancestor = DOMAIN, the top = ROOT).
    /// The qualifying set is sorted deterministically (leaf-first) and windowed by a rotating offset so a fixed query set
    /// is trained when the space is small (the whole set every cycle) while a big space gets variety over many cycles.</summary>
    private List<NavQueryDaggerTrainer.Query> SampleNavigatorQueries(DialecticalSpace ds, int maxMembers)
    {
        var queries = new List<NavQueryDaggerTrainer.Query>();

        // MEMBERS: living, non-numeric concepts with at least one relation (so they CAN have a more-general ancestor).
        // Leaf-first ordering (lowest relational degree) puts the specific concepts — the ones with a real ancestor
        // ladder — first; ThenBy symbol makes the window deterministic across calls (so the trainer actually converges).
        var qualifying = ds.ActiveConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c) && !double.TryParse(c, out _) && ds.GetRelationDegree(c) > 0)
            .OrderBy(ds.GetRelationDegree).ThenBy(c => c, StringComparer.Ordinal)
            .ToList();
        if (qualifying.Count == 0) return queries;

        var take = Math.Min(Math.Max(1, maxMembers), qualifying.Count);
        // Rotate so, over many cycles, every leaf is covered. When the set is no bigger than the window, every cycle sees
        // ALL of it (the small-space / test case) — a stable set, so loss trends down cleanly.
        var start = qualifying.Count <= take ? 0 : (int)((long)_navCycle * take % qualifying.Count);
        _navCycle++;

        var added = 0;
        for (var i = 0; i < qualifying.Count && added < take; i++)
        {
            var member = qualifying[(start + i) % qualifying.Count];
            var chain = ClimbAncestors(ds, member);
            if (chain.Count == 0) continue;                                                   // flat below this node
            queries.Add(new NavQueryDaggerTrainer.Query(member, (int)NavCue.Genus, chain[0])); // immediate parent = GENUS
            if (chain.Count >= 2) queries.Add(new(member, (int)NavCue.Domain, chain[1]));       // 2-hop ancestor = DOMAIN
            if (chain.Count >= 3) queries.Add(new(member, (int)NavCue.Root, chain[^1]));        // the top reached = ROOT
            added++;
        }
        return queries;
    }

    /// <summary>Climb the live relation graph UPWARD from <paramref name="start"/>, each hop to the strictly-more-general
    /// neighbour — categories are relational HUBS, so higher relational degree is the substrate's own "is-a-parent" signal
    /// (no hardcoded taxonomy, no test-specific labels). Returns the generality chain [parent, grandparent, …]; EMPTY when
    /// the node has no more-general neighbour — a flat space, where only the GENUS cue can be trained (expected early in a
    /// run, deepening as the space grows).</summary>
    private static List<string> ClimbAncestors(DialecticalSpace ds, string start, int maxDepth = 4)
    {
        var chain = new List<string>();
        var visited = new HashSet<string>(StringComparer.Ordinal) { start };
        var cur = start;
        var curDeg = ds.GetRelationDegree(start);
        for (var depth = 0; depth < maxDepth; depth++)
        {
            IReadOnlyList<PlatonicNeighbor> nbrs;
            try { nbrs = ds.GetNeighbors(cur, PlatonicNeighborhoodType.Relational, 16, 0.0); }
            catch { break; }
            string? best = null;
            var bestDeg = curDeg;
            foreach (var n in nbrs)
            {
                if (visited.Contains(n.Concept)) continue;
                if (double.TryParse(n.Concept, out _)) continue;   // numbers never form ancestor edges (substrate hard rule)
                var dg = ds.GetRelationDegree(n.Concept);
                if (dg > bestDeg) { bestDeg = dg; best = n.Concept; }
            }
            if (best is null) break;                                // no strictly-more-general neighbour → top reached
            chain.Add(best);
            visited.Add(best);
            cur = best;
            curDeg = bestDeg;
        }
        return chain;
    }

    /// <summary>Probe a few of the freshly-trained queries with the on-policy walk (NO answer supplied — relies on the
    /// net's learned HALT) and fold every traversed concept into the engine's persistent self (the vital loop). Returns
    /// (resolve%, abstain%): RESOLVE = confident halt on the cued ancestor, ABSTAIN = no confident halt in the budget.</summary>
    private (double ResolvePct, double AbstainPct) EvaluateAndFoldSelf(
        DialecticalSpace ds, NavQueryPolicyNet net, List<NavQueryDaggerTrainer.Query> queries, Device device)
    {
        var evalCount = Math.Min(NavEvalWalks, queries.Count);
        if (evalCount == 0) return (0.0, 0.0);

        var walk = new NavigatorWalk();
        int resolved = 0, abstained = 0;
        for (var i = 0; i < evalCount; i++)
        {
            var q = queries[i];
            if (!ds.TryGetConceptFace(q.Member, out var anchor)) continue;
            try
            {
                using var policy = new QueryNavPolicy(net, ds, anchor, q.Cue, device, NavK, 0.0, 0.5);
                var res = walk.Walk(ds, q.Member, anchor, null, policy, new NavWalkOptions(MaxSteps: NavMaxSteps));
                if (policy.LastHalt)
                {
                    if (string.Equals(res.FinalSymbol, q.Ancestor, StringComparison.Ordinal)) resolved++;
                }
                else abstained++;
                foreach (var s in res.Trajectory) _state.Inference.PerceiveSelf(s); // close the loop through the one self
            }
            catch { /* a single bad walk must never crash the gym */ }
        }
        return ((double)resolved / evalCount, (double)abstained / evalCount);
    }

    private static bool IsGpuMemoryError(Exception ex)
    {
        var m = ex.Message ?? string.Empty;
        return m.Contains("out of memory", StringComparison.OrdinalIgnoreCase)
            || m.Contains("CUDA", StringComparison.OrdinalIgnoreCase) && m.Contains("alloc", StringComparison.OrdinalIgnoreCase)
            || m.Contains("CUBLAS", StringComparison.OrdinalIgnoreCase)
            || m.Contains("cudaErrorMemoryAllocation", StringComparison.OrdinalIgnoreCase);
    }
}
