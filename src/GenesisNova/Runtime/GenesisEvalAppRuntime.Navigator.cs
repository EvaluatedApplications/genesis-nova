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

/// <summary>The outcome of ONE observational navigator walk (the REPL `/nav` command). <see cref="Answer"/> is the
/// emergent landing concept the query relaxed to; <see cref="Reached"/> = a confident halt (vs an abstain); <see
/// cref="Trajectory"/> is the path of concepts walked; <see cref="SelfConditioned"/> = the live engine self was
/// non-empty and tilted the walk; <see cref="Message"/> carries an unknown-concept / dead-net note when it can't walk.</summary>
public readonly record struct NavWalkObservation(
    string Anchor, NavCue Cue, string Answer, bool Reached, bool SelfConditioned,
    IReadOnlyList<string> Trajectory, string Message);

/// <summary>A snapshot of the navigator's observable state for the Inspect panel (see
/// <see cref="GenesisEvalAppRuntime.GetNavigatorDiagnostics"/>): last-cycle training vitals, the running resolve%, the
/// engine self's magnitude + whether it conditions cognition, the live concepts nearest the self ("what the mind is
/// dwelling on"), and the most recent `/nav` walk (null until one has run).</summary>
public readonly record struct NavigatorDiagnostics(
    double LastLoss, int LastQueries, double LastResolvePct, double LastAbstainPct,
    double RunningResolvePct, double SelfMagnitude, bool SelfConditions,
    IReadOnlyList<(string Concept, double Similarity)> SelfFocus, NavWalkObservation? LastWalk);

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
    private const int NavDaggerRounds = 1;    // on-policy DAgger rounds after the BC warm-start (0 = pure BC)
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

        // ── DAgger (Dataset Aggregation): the BC step above imitates the ORACLE's path; this trains the policy on the
        // states IT actually reaches. Roll out the net's OWN walk, label every visited node with the oracle's correct
        // next-step (free — one Dijkstra already covers the whole graph), aggregate with the BC set, retrain. Teaches the
        // walker to RECOVER from its own slips instead of only replaying a perfect path. Bounded: NavDaggerRounds rounds.
        var aggregate = new List<NavQueryTrajectory>(trajectories);
        for (var round = 0; round < NavDaggerRounds; round++)
        {
            try
            {
                var rollouts = NavQueryDaggerTrainer.RolloutQueryTrajectories(net, ds, queries, device, NavMaxSteps, NavK);
                if (rollouts.Count == 0) break;
                aggregate.AddRange(rollouts);
                losses = NavQueryDaggerTrainer.TrainQuery(net, aggregate, Math.Max(1, epochs), NavLr, device, NavK);
            }
            catch (Exception ex) when (device.type == DeviceType.CUDA && IsGpuMemoryError(ex))
            {
                Console.WriteLine($"[nav] DAgger CUDA alloc failed — CPU for this round: {ex.Message}");
                try { net.to(CPU); } catch { }
                device = CPU;
                var rollouts = NavQueryDaggerTrainer.RolloutQueryTrajectories(net, ds, queries, device, NavMaxSteps, NavK);
                if (rollouts.Count == 0) break;
                aggregate.AddRange(rollouts);
                losses = NavQueryDaggerTrainer.TrainQuery(net, aggregate, Math.Max(1, epochs), NavLr, CPU, NavK);
            }
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

    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    //  READ-ONLY OBSERVATION SURFACES (the REPL `/nav` walk + the Inspect "Navigator" panel). These NEVER train or
    //  persist — they run the ALREADY-trained shared net (State.Navigator) as a query walk and read the live engine
    //  self, all under the SAME model gate the predict path uses, so the UI thread never touches torch off-gate.
    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

    private NavWalkObservation? _lastNavWalk; // the most recent `/nav` walk (for the Inspect panel) — gate-guarded

    /// <summary>Walk the trained navigator LIVE from <paramref name="concept"/> with the given target-aspect cue (parsed
    /// from <paramref name="cue"/>: genus|domain|root, default GENUS), reading the engine's persistent self so the answer
    /// reflects what the mind has been dwelling on. PURELY OBSERVATIONAL — no training, no persistence, and (unlike the
    /// gym's evaluate-and-fold) it does NOT write the traversal back into the self. Gated like a predict. Returns where
    /// the query relaxed to, whether it confidently halted (reached) vs abstained, the trajectory, and whether the self
    /// conditioned the walk. Unknown concept / non-dialectical space / empty net all return a clear message, never throw.</summary>
    public NavWalkObservation WalkNavigator(string concept, string? cue = null)
        => WithModelGate(() =>
        {
            var w = WalkNavigatorLocked(concept, cue);
            _lastNavWalk = w;
            return w;
        });

    private static NavCue ParseCue(string? cue) => (cue?.Trim().ToLowerInvariant()) switch
    {
        "domain" => NavCue.Domain,
        "root" => NavCue.Root,
        _ => NavCue.Genus, // default + unknown → the immediate kind
    };

    private NavWalkObservation WalkNavigatorLocked(string concept, string? cue)
    {
        var nav = ParseCue(cue);
        var anchorSym = (concept ?? string.Empty).Trim();
        if (anchorSym.Length == 0)
            return new NavWalkObservation(anchorSym, nav, anchorSym, false, false, new[] { anchorSym }, "no concept given");

        if (_state.Memory is not DialecticalSpace ds)
            return new NavWalkObservation(anchorSym, nav, anchorSym, false, false, new[] { anchorSym },
                "navigator needs the dialectical core (UseDialecticalCore)");

        if (!ds.TryGetConceptFace(anchorSym, out var anchor))
            return new NavWalkObservation(anchorSym, nav, anchorSym, false, false, new[] { anchorSym },
                $"'{anchorSym}' is not a live concept in the space");

        // Read the LIVE engine self (null when self-conditioning is off, or before the mind has dwelt on anything).
        double[]? self = null;
        if (_state.Inference.SelfConditionsCognition)
        {
            var sf = _state.Inference.SelfField;
            if (sf.Count > 0) { self = new double[sf.Count]; for (var i = 0; i < sf.Count; i++) self[i] = sf[i]; }
        }

        // The thin controller rests on CPU between gym cycles; run the walk on whatever device its weights are on so the
        // tensors the policy builds match the net's params (no cross-device Step).
        var device = _state.Navigator.parameters().FirstOrDefault()?.device ?? CPU;

        try
        {
            using var policy = new QueryNavPolicy(_state.Navigator, ds, anchor, (int)nav, device, NavK, 0.0, 0.5, self);
            var res = new NavigatorWalk().Walk(ds, anchorSym, anchor, null, policy, new NavWalkOptions(MaxSteps: NavMaxSteps));
            var reached = policy.LastHalt; // a confident halt = relaxed to an answer; no halt in the budget = structural abstain
            return new NavWalkObservation(anchorSym, nav, res.FinalSymbol, reached, self is not null, res.Trajectory,
                reached ? "resolved" : "abstained (no confident halt in the step budget)");
        }
        catch (Exception ex)
        {
            return new NavWalkObservation(anchorSym, nav, anchorSym, false, self is not null, new[] { anchorSym },
                $"walk failed: {ex.GetType().Name}");
        }
    }

    /// <summary>Cheap, gate-guarded read of the navigator's observable state for the Inspect panel: the last training
    /// cycle's metrics, the running resolve%, the engine self's magnitude + whether it conditions cognition, the few
    /// live concepts NEAREST the self vector ("what the mind is dwelling on"), and the most recent <see
    /// cref="WalkNavigator"/> result. All plain CPU reads — no torch on the caller's thread.</summary>
    public NavigatorDiagnostics GetNavigatorDiagnostics(int topFocus = 6)
    {
        return WithModelGate(() =>
        {
            var t = _lastNavTrain;
            var sf = _state.Inference.SelfField;
            double mag = 0.0; for (var i = 0; i < sf.Count; i++) mag += sf[i] * sf[i]; mag = Math.Sqrt(mag);

            IReadOnlyList<(string Concept, double Similarity)> focus = Array.Empty<(string, double)>();
            if (sf.Count > 0 && _state.Memory is DialecticalSpace ds)
                focus = NearestToSelf(ds, sf, Math.Max(1, topFocus));

            return new NavigatorDiagnostics(
                LastLoss: t.Loss, LastQueries: t.Queries, LastResolvePct: t.ResolvePct, LastAbstainPct: t.AbstainPct,
                RunningResolvePct: _navResolveEma, SelfMagnitude: mag,
                SelfConditions: _state.Inference.SelfConditionsCognition,
                SelfFocus: focus, LastWalk: _lastNavWalk);
        });
    }

    // The top concepts whose meaning-cloud points most along the self vector — the LivingSelf made visible. Bounded scan
    // (the self is unit-normalised, so this is a cosine against each cloud); skips numbers/atoms via SemanticVectorOf.
    private static IReadOnlyList<(string Concept, double Similarity)> NearestToSelf(
        DialecticalSpace ds, IReadOnlyList<double> self, int top)
    {
        const int MaxScan = 6000; // keep the timer-driven read cheap even on a large overnight space
        var scored = new List<(string, double)>();
        var scanned = 0;
        foreach (var c in ds.ActiveConcepts)
        {
            if (scanned++ >= MaxScan) break;
            var v = ds.SemanticVectorOf(c);
            if (v is null) continue;
            double dot = 0, nv = 0;
            var n = Math.Min(v.Length, self.Count);
            for (var i = 0; i < n; i++) { dot += v[i] * self[i]; nv += v[i] * v[i]; }
            if (nv <= 1e-12) continue;
            scored.Add((c, dot / Math.Sqrt(nv))); // self is already unit-length
        }
        return scored.OrderByDescending(x => x.Item2).Take(top).ToArray();
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
