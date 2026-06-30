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

/// <summary>One HELD-OUT navigator evaluation point — the M4 acceptance signal. <see cref="Cycle"/> is the held-out
/// eval index (every eval appends one), <see cref="ResolvePct"/> = the fraction of held-out queries that confidently
/// HALTED (in [0,1]), <see cref="AccuracyPct"/> = the fraction that halted ON THE CUED ANCESTOR (correct landing — the
/// honest generalization number), <see cref="Count"/> = how many held-out queries were evaluated. The held-out set is
/// PLANTED into the graph but EXCLUDED from <see cref="GenesisEvalAppRuntime.SampleNavigatorQueries"/>, so a CLIMBING
/// series proves the navigator GENERALIZES the walk to queries it never trained on — not training-set memorization.</summary>
public readonly record struct NavHeldOutPoint(int Cycle, double ResolvePct, double AccuracyPct, int Count)
{
    public static NavHeldOutPoint Empty => new(0, 0.0, 0.0, 0);
}

/// <summary>A snapshot of the navigator's observable state for the Inspect panel (see
/// <see cref="GenesisEvalAppRuntime.GetNavigatorDiagnostics"/>): last-cycle training vitals, the running resolve%, the
/// engine self's magnitude + whether it conditions cognition, the live concepts nearest the self ("what the mind is
/// dwelling on"), the most recent `/nav` walk (null until one has run), and the latest HELD-OUT eval point (the M4
/// generalization curve's newest sample; <see cref="NavHeldOutPoint.Empty"/> until a held-out set is planted).</summary>
public readonly record struct NavigatorDiagnostics(
    double LastLoss, int LastQueries, double LastResolvePct, double LastAbstainPct,
    double RunningResolvePct, double SelfMagnitude, bool SelfConditions,
    IReadOnlyList<(string Concept, double Similarity)> SelfFocus, NavWalkObservation? LastWalk,
    NavHeldOutPoint LastHeldOut);

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
    private const int NavDaggerRounds = 2;    // on-policy DAgger rounds after the BC warm-start (teaches the walker to
                                              // RECOVER from its own slips, not just replay a perfect path). RE-ENABLED
                                              // (M3): gym nav training NO LONGER writes the SHARED model _selfField — the
                                              // vital-loop write was moved OUT of the training walk and INTO inference
                                              // resolution (only a genuinely AMBIGUOUS query folds its OWN concluded
                                              // answer into the self, in GenesisInferenceEngine.TryFieldRelax). So
                                              // DAgger's successful rollouts can no longer fold category HUBS into the
                                              // self / degrade the function-word foundation. Prebake stability with DAgger
                                              // on is proven by FunctionWordConservationTests + the SelfAssess separation.
    private const int NavEvalWalks = 6;       // on-policy probes for the resolve%/abstain% signal
    private const int NavMaxSteps = 8;        // the walk's cognitive light-cone for the probe
    private const int NavK = NavQueryDaggerTrainer.DefaultK;
    private const double NavLr = 1e-3;

    private int _navCycle;                              // rotates the sampling window so the whole space is covered over time
    private NavTrainCycleResult _lastNavTrain = NavTrainCycleResult.Empty;
    private double _navResolveEma;                      // running resolve% (EMA) for a later inspect tab
    private bool _navResolveSeeded;

    // ── HELD-OUT GENERALIZATION EVAL (M4) ────────────────────────────────────────────────────────────────────────
    // A PLANTED query set the training sampler NEVER sees — the honest "does the walker GENERALIZE" signal an overnight
    // run must show CLIMBING. Each held-out query is (member, cue, expected-ancestor): the member's ancestor chain is
    // grown in the live graph (so the walk is structurally possible) but the member is in `_navHeldOutMembers`, which
    // SampleNavigatorQueries/SampleCompositionQueries SKIP — so its trajectory is never distilled into the policy. The
    // curriculum (NavReasoningCurriculum) plants the structure + registers this set; EvaluateNavigatorHeldOut walks it
    // read-only every N cycles and appends a point to the warm-history the Inspect tab / a log line reads.
    private readonly record struct NavHeldOutQuery(string Member, NavCue Cue, string Ancestor);
    private readonly List<NavHeldOutQuery> _navHeldOut = new();
    private readonly HashSet<string> _navHeldOutMembers = new(StringComparer.Ordinal);
    private readonly List<NavHeldOutPoint> _navHeldOutHistory = new();
    private int _navHeldOutEvals;                        // monotonic eval index (the held-out curve's x-axis)
    private NavHeldOutPoint _lastHeldOut = NavHeldOutPoint.Empty;

    // ── LEARNED PER-LEVEL GOAL-REGIONS (M4 multi-hop fix) ─────────────────────────────────────────────────────────
    // The prototype/centroid FACE of the concepts at each abstraction LEVEL (Genus=0/Domain=1/Root=2), derived purely
    // from the live relation graph's DEPTH (no word list, no hardcoded taxonomy — climb each LEAF's is-a chain and pool
    // the genus/domain/root nodes it reaches; centroid their faces). This is the GOAL a LEVEL walk descends toward across
    // MULTIPLE hops — the missing signal that left domain/root LANDING to novel anchors at the ~0% structural ceiling.
    // Refreshed each training cycle + before each held-out eval (the graph grows); shared BY REFERENCE with the engine
    // (Inference.NavLevelGoalRegions) so the live PredictAsync ambiguous branch uses the same regions training learns on.
    private readonly double[]?[] _navLevelRegions = new double[NavQueryFeatures.CueCount][];
    private bool _navRegionsWired;
    private const int NavRegionMaxLeaves = 4000; // bound the per-cycle leaf climb on a large overnight space

    /// <summary>The learned per-level goal-REGION centroid for <paramref name="cue"/> (a full face), recomputed from the
    /// CURRENT live graph, or null when the space has no concept at that level yet. Public for the multi-hop ceiling
    /// acceptance test (it conditions a held-out walk on the same region training + inference use). Gated.</summary>
    public IReadOnlyList<double>? NavLevelRegion(NavCue cue)
        => WithModelGate(() =>
        {
            if (_state.Memory is not DialecticalSpace ds) return (IReadOnlyList<double>?)null;
            EnsureLevelRegions(ds);
            var i = (int)cue;
            return (IReadOnlyList<double>?)(i >= 0 && i < _navLevelRegions.Length ? _navLevelRegions[i] : null);
        });

    /// <summary>Recompute the per-level goal-regions from the live relation graph and publish them to the engine (M4).
    /// CHEAP (bounded leaf climb over a small taxonomy); refreshed each training cycle + before each held-out read so the
    /// regions track the growing graph. Derivation is hardcoding-free: a concept's LEVEL comes from where it sits in a
    /// LEAF's degree-climb chain (immediate parent ⇒ Genus, 2-hop ⇒ Domain, top ⇒ Root — the same depth signal M1's
    /// LearnNavLevelCue uses). Climbing ONLY from leaves keeps a genus node (e.g. "mammal") out of the Domain pool.</summary>
    private void EnsureLevelRegions(DialecticalSpace ds)
    {
        if (!_navRegionsWired) { _state.Inference.NavLevelGoalRegions = _navLevelRegions; _navRegionsWired = true; }

        var genus = new HashSet<string>(StringComparer.Ordinal);
        var domain = new HashSet<string>(StringComparer.Ordinal);
        var root = new HashSet<string>(StringComparer.Ordinal);
        var leaves = 0;
        foreach (var c in ds.ActiveConcepts)
        {
            if (leaves >= NavRegionMaxLeaves) break;
            if (string.IsNullOrWhiteSpace(c) || double.TryParse(c, out _)) continue;
            if (!IsTaxonomyLeaf(ds, c)) continue;
            leaves++;
            var chain = ClimbAncestors(ds, c);
            if (chain.Count >= 1) genus.Add(chain[0]);
            if (chain.Count >= 2) { domain.Add(chain[1]); root.Add(chain[^1]); }
        }
        _navLevelRegions[(int)NavCue.Genus] = Centroid(ds, genus);
        _navLevelRegions[(int)NavCue.Domain] = Centroid(ds, domain);
        _navLevelRegions[(int)NavCue.Root] = Centroid(ds, root);
    }

    // A concept is a TAXONOMY LEAF iff it has at least one strong relation but NO strictly-more-specific strong neighbour
    // (no neighbour of lower strong-degree) — i.e. nothing is-a it. Categories are relational HUBS (higher degree), so a
    // leaf sits at the bottom of the degree gradient; we climb regions from leaves only so each ancestor lands at its
    // true level (the level a real query about a leaf would ask for).
    private static bool IsTaxonomyLeaf(DialecticalSpace ds, string c)
    {
        var deg = ds.StrongRelationDegree(c, NavStrongRelation);
        if (deg == 0) return false;
        IReadOnlyList<PlatonicNeighbor> nbrs;
        try { nbrs = ds.GetNeighbors(c, PlatonicNeighborhoodType.Relational, 16, NavStrongRelation); }
        catch { return false; }
        foreach (var n in nbrs)
        {
            if (double.TryParse(n.Concept, out _)) continue;
            if (ds.StrongRelationDegree(n.Concept, NavStrongRelation) < deg) return false; // a more-specific neighbour → not a leaf
        }
        return true;
    }

    // Centroid of a set of concepts' faces (the level's prototype), or null when none has a face. Full-dim faces only.
    private static double[]? Centroid(DialecticalSpace ds, IEnumerable<string> nodes)
    {
        double[]? sum = null; var n = 0;
        foreach (var x in nodes)
        {
            if (!ds.TryGetConceptFace(x, out var f) || f.Length == 0) continue;
            sum ??= new double[f.Length];
            if (f.Length != sum.Length) continue;
            for (var i = 0; i < sum.Length; i++) sum[i] += f[i];
            n++;
        }
        if (sum is null || n == 0) return null;
        for (var i = 0; i < sum.Length; i++) sum[i] /= n;
        return sum;
    }

    // The level goal-region for a cue as a float[] (the Query.Kind / QueryNavPolicy.kindFace form), or null when absent.
    private float[]? LevelRegionFloat(NavCue cue)
    {
        var i = (int)cue;
        var r = i >= 0 && i < _navLevelRegions.Length ? _navLevelRegions[i] : null;
        if (r is not { Length: > 0 }) return null;
        var f = new float[r.Length];
        for (var j = 0; j < r.Length; j++) f[j] = (float)r[j];
        return f;
    }

    // The level goal-region for a cue as a double[] (for a direct QueryNavPolicy walk), or null when absent.
    private double[]? LevelRegionDouble(NavCue cue)
    {
        var i = (int)cue;
        var r = i >= 0 && i < _navLevelRegions.Length ? _navLevelRegions[i] : null;
        return r is { Length: > 0 } ? (double[])r.Clone() : null;
    }

    /// <summary>The HELD-OUT navigator generalization curve (one point per <see cref="EvaluateNavigatorHeldOut"/> call) —
    /// the M4 acceptance series. A climbing <see cref="NavHeldOutPoint.AccuracyPct"/> over an overnight run is the proof
    /// the loop improves GENERALIZING reasoning, not just training fit. Empty until a held-out set is registered.</summary>
    public IReadOnlyList<NavHeldOutPoint> NavHeldOutHistory { get { lock (_navHeldOutHistory) return _navHeldOutHistory.ToArray(); } }

    /// <summary>How many held-out queries are currently registered (0 = no held-out eval will run).</summary>
    public int NavHeldOutCount { get { lock (_navHeldOut) return _navHeldOut.Count; } }

    /// <summary>REGISTER a fixed HELD-OUT navigator query set (member, cue, expected-ancestor) plus any extra members to
    /// EXCLUDE from training (e.g. a curriculum's cue-marker-teaching subjects, whose shortcut edges would otherwise
    /// corrupt the degree-climb labels). Both the eval members AND <paramref name="trainExclusions"/> are added to the
    /// sampler's exclude set so their trajectories are NEVER trained — the eval then measures pure generalization.
    /// Re-registering REPLACES the set. The CALLER must have PLANTED each held-out member's ancestor chain in the live
    /// graph (adjacency-only, no shortcuts) so the held-out walk is structurally possible.</summary>
    public void RegisterNavigatorHeldOut(
        IEnumerable<(string Member, NavCue Cue, string Ancestor)> queries,
        IEnumerable<string>? trainExclusions = null)
        => WithModelGate(() =>
        {
            lock (_navHeldOut)
            {
                _navHeldOut.Clear();
                _navHeldOutMembers.Clear();
                foreach (var (m, cue, anc) in queries ?? Array.Empty<(string, NavCue, string)>())
                {
                    if (string.IsNullOrWhiteSpace(m) || string.IsNullOrWhiteSpace(anc)) continue;
                    _navHeldOut.Add(new NavHeldOutQuery(m, cue, anc));
                    _navHeldOutMembers.Add(m);
                }
                foreach (var x in trainExclusions ?? Array.Empty<string>())
                    if (!string.IsNullOrWhiteSpace(x)) _navHeldOutMembers.Add(x);
            }
            return 0;
        });

    /// <summary>PLANT a set of is-a ADJACENCY edges (child → strictly-more-general parent) into the live relation graph,
    /// each reinforced <paramref name="reinforce"/> times so the relation is firm. Adjacency-ONLY by contract (the caller
    /// passes parent = the IMMEDIATE generalisation), which keeps the degree gradient monotone so ClimbAncestors derives
    /// a clean genus→domain→root chain — the structure SampleNavigatorQueries distils into trajectories. Numbers are
    /// rejected (the substrate's hard rule). Gated + flushed. Used by NavReasoningCurriculum to grow its taxonomy.</summary>
    public void PlantNavigatorTaxonomy(IEnumerable<(string Child, string Parent)> edges, int reinforce = 4)
        => WithModelGate(() =>
        {
            if (_state.Memory is not DialecticalSpace ds) return 0;
            foreach (var (child, parent) in edges ?? Array.Empty<(string, string)>())
            {
                if (string.IsNullOrWhiteSpace(child) || string.IsNullOrWhiteSpace(parent)) continue;
                if (double.TryParse(child, out _) || double.TryParse(parent, out _)) continue; // numbers never form ancestor edges
                for (var i = 0; i < Math.Max(1, reinforce); i++) ds.ObserveContradiction(child, parent, 0.0);
            }
            ds.FlushCloudBatch();
            return 0;
        });

    /// <summary>TEACH the LEARNED navigator level cue from one (query, ancestor-answer) frame — the SAME
    /// <c>LearnNavLevelCue</c> the gym observe path runs, but invoked WITHOUT the structural input→output coupling +
    /// distractor-repulsion that <c>ObservePlatonicSpace</c> also performs (that repulsion writes weak edges + drifts the
    /// taxonomy clouds, destabilising the navigator's substrate). The level still comes from the answer's GRAPH DEPTH (no
    /// hardcoded cue-word list); only the marker→∘level relation is written. Gated. Used by NavReasoningCurriculum so its
    /// cue frames self-teach the cue as DATA without corrupting the planted is-a taxonomy. No-op for a non-ancestor frame.</summary>
    public void TeachNavLevelCue(string input, string output)
        => WithModelGate(() => { try { _state.Inference.LearnNavLevelCue(input, output); } catch { } return 0; });

    /// <summary>EVALUATE the registered held-out query set on the LIVE trained navigator (read-only — no training, no
    /// self-write), append a point to the warm-history, and return it. Bounded + GPU-OOM-safe (CPU walk of the resting
    /// thin controller) + gated like a predict. Returns <see cref="NavHeldOutPoint.Empty"/> when nothing is registered
    /// or the space is non-dialectical. THIS is the series the M4 overnight run must show climbing.</summary>
    public NavHeldOutPoint EvaluateNavigatorHeldOut()
        => WithModelGate(EvaluateNavigatorHeldOutLocked);

    private NavHeldOutPoint EvaluateNavigatorHeldOutLocked()
    {
        NavHeldOutQuery[] set;
        lock (_navHeldOut) set = _navHeldOut.ToArray();
        if (set.Length == 0 || _state.Memory is not DialecticalSpace ds)
            return _lastHeldOut;

        try { EnsureLevelRegions(ds); } catch { } // condition the held-out walk on the SAME learned regions training uses
        var net = _state.Navigator;
        var device = net.parameters().FirstOrDefault()?.device ?? CPU; // walk on whatever device the resting net is on
        int halted = 0, correct = 0, scored = 0;
        foreach (var q in set)
        {
            if (!ds.TryGetConceptFace(q.Member, out _)) continue;
            scored++;
            var (h, c) = HeldOutWalk(ds, net, device, q);
            halted += h; correct += c;
        }
        if (scored == 0) return _lastHeldOut;

        var point = new NavHeldOutPoint(++_navHeldOutEvals, (double)halted / scored, (double)correct / scored, scored);
        _lastHeldOut = point;
        lock (_navHeldOutHistory)
        {
            _navHeldOutHistory.Add(point);
            if (_navHeldOutHistory.Count > 2000) _navHeldOutHistory.RemoveAt(0); // bounded for an overnight run
        }
        return point;
    }

    // ONE held-out walk: condition on the cue's learned level goal-region (the unified goal channel; M4), NO self (pure
    // policy generalization). Returns (halted, correct) as 0/1. Exception-isolated — one bad walk never breaks the read.
    private (int Halted, int Correct) HeldOutWalk(DialecticalSpace ds, NavQueryPolicyNet net, Device device, NavHeldOutQuery q)
    {
        if (!ds.TryGetConceptFace(q.Member, out var anchor)) return (0, 0);
        try
        {
            using var policy = new QueryNavPolicy(net, ds, anchor, (int)q.Cue, device, NavK, 0.0, 0.5,
                selfVec: null, kindFace: LevelRegionDouble(q.Cue));
            var res = new NavigatorWalk().Walk(ds, q.Member, anchor, null, policy, new NavWalkOptions(MaxSteps: NavMaxSteps));
            if (!policy.LastHalt) return (0, 0);
            return (1, string.Equals(res.FinalSymbol, q.Ancestor, StringComparison.Ordinal) ? 1 : 0);
        }
        catch { return (0, 0); }
    }

    /// <summary>One held-out generalization point PER CUE (Genus/Domain/Root) on the LIVE trained navigator — the honest
    /// breakdown that shows WHETHER THE MULTI-HOP (Domain/Root) LANDING ceiling broke, not just the aggregate. Read-only,
    /// gated, GPU-OOM-safe; conditions each walk on the cue's learned level goal-region. Empty when nothing is registered.</summary>
    public IReadOnlyList<NavHeldOutPoint> EvaluateNavigatorHeldOutPerCue()
        => WithModelGate(() =>
        {
            NavHeldOutQuery[] set;
            lock (_navHeldOut) set = _navHeldOut.ToArray();
            if (set.Length == 0 || _state.Memory is not DialecticalSpace ds)
                return (IReadOnlyList<NavHeldOutPoint>)Array.Empty<NavHeldOutPoint>();

            try { EnsureLevelRegions(ds); } catch { }
            var net = _state.Navigator;
            var device = net.parameters().FirstOrDefault()?.device ?? CPU;
            var points = new List<NavHeldOutPoint>(NavQueryFeatures.CueCount);
            foreach (var cue in new[] { NavCue.Genus, NavCue.Domain, NavCue.Root })
            {
                int halted = 0, correct = 0, scored = 0;
                foreach (var q in set)
                {
                    if (q.Cue != cue || !ds.ContainsConcept(q.Member)) continue;
                    scored++;
                    var (h, c) = HeldOutWalk(ds, net, device, q);
                    halted += h; correct += c;
                }
                if (scored > 0) points.Add(new NavHeldOutPoint((int)cue, (double)halted / scored, (double)correct / scored, scored));
            }
            return (IReadOnlyList<NavHeldOutPoint>)points;
        });

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

        // Refresh the learned per-level goal-regions from the current graph BEFORE sampling, so the level queries train on
        // the same goal the live walk + held-out eval condition on (the unified goal channel; M4 multi-hop fix).
        try { EnsureLevelRegions(ds); } catch { }

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

        // Cheap on-policy probe — do the freshly-trained queries RESOLVE under the net's learned halt? This is now a
        // READ-ONLY signal: it does NOT write the shared engine self (M3). The vital-loop self-write was moved to
        // INFERENCE resolution (TryFieldRelax folds a genuinely-ambiguous query's OWN concluded answer), so gym training
        // — including the DAgger rollouts above — can never pollute the self with traversed category hubs.
        var (resolvePct, abstainPct) = EvaluateNavigatorResolve(ds, net, queries, device);

        // Keep the net at REST on CPU between cycles — frees GPU memory for the gym model (overnight-safe) and makes the
        // persister's CPU weight export cheap. The move is ~tens of ms for this thin controller.
        try { if (device.type == DeviceType.CUDA) net.to(CPU); } catch { }

        var result = new NavTrainCycleResult(losses.CrossEntropy, queries.Count, resolvePct, abstainPct);
        _lastNavTrain = result;
        _navResolveEma = _navResolveSeeded ? 0.7 * _navResolveEma + 0.3 * resolvePct : resolvePct;
        _navResolveSeeded = true;
        return result;
    }

    /// <summary>Sample concepts that HAVE relational ancestors and turn each into up to three cued queries, deriving the
    /// cue from the LIVE relation graph's depth (immediate parent = GENUS, 2-hop ancestor = DOMAIN, the TOP of the chain
    /// = ROOT). ROOT is emitted whenever the chain is ≥2 hops — so a GENUS HUB (chain [domain, root]) gets a genus→root
    /// pair too, not just leaf members (chain ≥3): the sampler is now deep enough that TrainNavigatorCycle ALONE trains
    /// the full genus→domain→root range (the M1 deepening). The qualifying set is sorted deterministically (leaf-first)
    /// and windowed by a rotating offset so a fixed query set trains when the space is small (the whole set every cycle)
    /// while a big space gets variety over many cycles.</summary>
    private List<NavQueryDaggerTrainer.Query> SampleNavigatorQueries(DialecticalSpace ds, int maxMembers)
    {
        var queries = new List<NavQueryDaggerTrainer.Query>();

        // MEMBERS: living, non-numeric concepts with at least one relation (so they CAN have a more-general ancestor).
        // Leaf-first ordering (lowest relational degree) puts the specific concepts — the ones with a real ancestor
        // ladder — first; ThenBy symbol makes the window deterministic across calls (so the trainer actually converges).
        // EXCLUDE the registered HELD-OUT members (M4): their ancestor chains are in the graph so they CAN be walked,
        // but the policy must never train on their trajectories — that is what makes EvaluateNavigatorHeldOut an honest
        // GENERALIZATION signal rather than training fit.
        bool heldOut(string c) { lock (_navHeldOutMembers) return _navHeldOutMembers.Contains(c); }
        var qualifying = ds.ActiveConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c) && !double.TryParse(c, out _) && ds.GetRelationDegree(c) > 0 && !heldOut(c))
            .OrderBy(ds.GetRelationDegree).ThenBy(c => c, StringComparer.Ordinal)
            .ToList();
        if (qualifying.Count == 0) return queries;

        var take = Math.Min(Math.Max(1, maxMembers), qualifying.Count);
        // Rotate so, over many cycles, every leaf is covered. When the set is no bigger than the window, every cycle sees
        // ALL of it (the small-space / test case) — a stable set, so loss trends down cleanly.
        var start = qualifying.Count <= take ? 0 : (int)((long)_navCycle * take % qualifying.Count);
        _navCycle++;

        var added = 0;
        var sampledMembers = new List<string>(take);
        for (var i = 0; i < qualifying.Count && added < take; i++)
        {
            var member = qualifying[(start + i) % qualifying.Count];
            sampledMembers.Add(member);
            var chain = ClimbAncestors(ds, member);
            if (chain.Count == 0) continue;                                                   // flat below this node
            // Each LEVEL query carries the learned per-level goal-REGION as its goal (M4): the per-candidate cand−goal
            // descent + the W_k halt-bias let the DOMAIN/ROOT walks aim at the right abstraction region across multiple
            // hops instead of stalling 1-hop — the fix for the held-out multi-hop landing ceiling. Null region (flat/cold
            // space) ⇒ Kind null ⇒ the M1 query-only walk (byte-identical).
            queries.Add(new NavQueryDaggerTrainer.Query(member, (int)NavCue.Genus, chain[0], Kind: LevelRegionFloat(NavCue.Genus)));
            if (chain.Count >= 2)
            {
                queries.Add(new(member, (int)NavCue.Domain, chain[1], Kind: LevelRegionFloat(NavCue.Domain))); // 2-hop = DOMAIN
                // ROOT = the TOP of the chain, emitted whenever it is ≥2 hops up — so a GENUS HUB (whose own chain is
                // only [domain, root], length 2) ALSO gets a genus→root pair, not just leaf members (chain ≥3).
                queries.Add(new(member, (int)NavCue.Root, chain[^1], Kind: LevelRegionFloat(NavCue.Root)));     // the top = ROOT
            }
            added++;
        }

        // M2 DEEPENING — CROSS-RELATION COMPOSITION + LOOKAHEAD-TRAP pairs. The genus/domain/root pairs above ride ONE
        // greedy degree-climb chain; this pass emits KIND-CONDITIONED composition targets: for each sampled member, the
        // specific concept that is-a CATEGORY HUB and lies ≥2 hops away (reached by composing across relation types). The
        // flow-field oracle defines the optimal path to it — routing AROUND a face-near low-degree distractor (the trap)
        // that a 1-hop heuristic falls for. Same member, different kind hub ⇒ a different composed answer. Robust to the
        // trap because targets are chosen by hub-degree (the substrate's own "kind" signal), never by proximity/strength.
        queries.AddRange(SampleCompositionQueries(ds, sampledMembers));
        return queries;
    }

    // M2: how many composition targets to emit per member (bounded — this runs every gym cycle alongside model training).
    private const int NavCompPerMemberCap = 3;
    private const int NavKindMinDegree = 3;   // a category hub must have ≥ this many members (mirrors the inference floor)

    /// <summary>Emit KIND-CONDITIONED cross-relation composition queries (M2). For each member, BFS the relation graph;
    /// any concept T reachable ≥2 hops away (composed across relations) that BELONGS to a CATEGORY HUB — i.e. T's
    /// highest-degree relational neighbour K is a real hub (degree ≥ floor, and above T's own degree) — becomes a
    /// composition target: emit (member, Genus, T, kindFace = face(K)). The kind is derived from the TARGET's own
    /// category (the substrate's own "is-a-a-hub" degree signal — no word list, no relation-name list); at inference the
    /// query word names that same hub. The oracle's flow field to the SPECIFIC T routes around face-near 1-hop
    /// distractors (the lookahead trap, which sits at distance 1 and is never itself a target), so the trajectory teaches
    /// LOOKAHEAD; the kind face teaches "halt on a concept of THIS kind" — so the same member under a different kind hub
    /// composes to a different answer along a different chain. Bounded (≤ cap targets/member) and exception-free.</summary>
    private List<NavQueryDaggerTrainer.Query> SampleCompositionQueries(DialecticalSpace ds, IReadOnlyList<string> members)
    {
        var result = new List<NavQueryDaggerTrainer.Query>();
        foreach (var m in members)
        {
            if (string.IsNullOrWhiteSpace(m) || !ds.TryGetConceptFace(m, out _)) continue;
            var dist = BfsDistances(ds, m, maxDepth: 3);
            // The cheap degree-climb already reaches its chain; composition adds value ONLY where the climb does NOT reach
            // the target — the genuine CROSS-RELATION / lookahead-trap cases (the climb is greedy and is fooled by a
            // high-degree distractor or stops at a non-hub intermediate). On a pure is-a taxonomy every target is on the
            // chain, so NO composition query is emitted there (it would only collide with the level GENUS cue). This is
            // what keeps the M1 level-cue gym training byte-stable while M2 trains the truly multi-hop cases.
            var climbChain = new HashSet<string>(ClimbAncestors(ds, m), StringComparer.Ordinal);
            var added = 0;
            // Prefer the nearest composition targets (a 2-hop answer before a 3-hop one) for a stable, cheap signal.
            foreach (var t in dist.Where(kv => kv.Value >= 2).OrderBy(kv => kv.Value).Select(kv => kv.Key))
            {
                if (added >= NavCompPerMemberCap) break;
                if (double.TryParse(t, out _) || t.Equals(m, StringComparison.Ordinal)) continue;
                lock (_navHeldOutMembers) { if (_navHeldOutMembers.Contains(t)) continue; } // never train a halt ON a held-out member (M4 leakage)
                if (climbChain.Contains(t)) continue;       // the degree-climb already reaches t → not a composition gap
                var tDeg = ds.GetRelationDegree(t);

                // T's CATEGORY = its highest-degree relational neighbour, when that neighbour is a real HUB (a category
                // with members) strictly more central than T. That hub IS the kind the query will name. No hub ⇒ T is not
                // a categorised answer (skip) — keeps people/intermediates from being mistaken for kind-members.
                string? kind = null; var kindDeg = NavKindMinDegree - 1;
                IReadOnlyList<PlatonicNeighbor> tn;
                try { tn = ds.GetNeighbors(t, PlatonicNeighborhoodType.Relational, 16, 0.0); }
                catch { continue; }
                foreach (var nb in tn)
                {
                    if (double.TryParse(nb.Concept, out _) || nb.Concept.Equals(m, StringComparison.Ordinal)) continue;
                    var d = ds.GetRelationDegree(nb.Concept);
                    if (d > kindDeg && d > tDeg) { kindDeg = d; kind = nb.Concept; }
                }
                if (kind is null || !ds.TryGetConceptFace(kind, out var kindFace)) continue;

                var kf = new float[kindFace.Length];
                for (var i = 0; i < kindFace.Length; i++) kf[i] = (float)kindFace[i];
                // GENUS cue + the kind face: the halt head learns "stop on the FIRST concept of this kind" (the specific
                // member — france/spain — NOT the category hub it would overshoot to under a deeper cue). The climb-chain
                // skip above guarantees this never collides with a level GENUS query (different target, same member).
                result.Add(new NavQueryDaggerTrainer.Query(m, (int)NavCue.Genus, t, Self: null, Kind: kf));
                added++;
            }
        }
        return result;
    }

    // Breadth-first hop distances from <paramref name="start"/> over the relation graph, capped at <paramref name="maxDepth"/>.
    // Distance 0 = start. Used to find composition targets ≥2 hops away (cross-relation) and to exclude 1-hop direct facts.
    private static Dictionary<string, int> BfsDistances(DialecticalSpace ds, string start, int maxDepth)
    {
        var dist = new Dictionary<string, int>(StringComparer.Ordinal) { [start] = 0 };
        var frontier = new Queue<string>();
        frontier.Enqueue(start);
        while (frontier.Count > 0)
        {
            var cur = frontier.Dequeue();
            var d = dist[cur];
            if (d >= maxDepth) continue;
            IReadOnlyList<PlatonicNeighbor> nbrs;
            try { nbrs = ds.GetNeighbors(cur, PlatonicNeighborhoodType.Relational, 32, NavStrongRelation); }
            catch { continue; }
            foreach (var n in nbrs)
            {
                if (double.TryParse(n.Concept, out _) || dist.ContainsKey(n.Concept)) continue;
                dist[n.Concept] = d + 1;
                frontier.Enqueue(n.Concept);
            }
        }
        return dist;
    }

    /// <summary>Climb the live relation graph UPWARD from <paramref name="start"/>, each hop to the strictly-more-general
    /// neighbour — categories are relational HUBS, so higher relational degree is the substrate's own "is-a-parent" signal
    /// (no hardcoded taxonomy, no test-specific labels). Returns the generality chain [parent, grandparent, …]; EMPTY when
    /// the node has no more-general neighbour — a flat space, where only the GENUS cue can be trained (expected early in a
    /// run, deepening as the space grows).</summary>
    // Strong-relation floor for the is-a climb: an is-a edge is a CONFIDENT relation (planted/reinforced ⇒ Strength≈1.0);
    // the trainer's distractor REPULSION writes Strength≈0.1 edges that would otherwise inflate a leaf's degree and break
    // the climb. Following + ranking by STRONG edges only keeps the taxonomy gradient intact under the live observe path.
    private const double NavStrongRelation = 0.5;

    private static List<string> ClimbAncestors(DialecticalSpace ds, string start, int maxDepth = 4)
    {
        var chain = new List<string>();
        var visited = new HashSet<string>(StringComparer.Ordinal) { start };
        var cur = start;
        var curDeg = ds.StrongRelationDegree(start, NavStrongRelation);
        for (var depth = 0; depth < maxDepth; depth++)
        {
            IReadOnlyList<PlatonicNeighbor> nbrs;
            try { nbrs = ds.GetNeighbors(cur, PlatonicNeighborhoodType.Relational, 16, NavStrongRelation); }
            catch { break; }
            string? best = null;
            var bestDeg = curDeg;
            foreach (var n in nbrs)
            {
                if (visited.Contains(n.Concept)) continue;
                if (double.TryParse(n.Concept, out _)) continue;   // numbers never form ancestor edges (substrate hard rule)
                var dg = ds.StrongRelationDegree(n.Concept, NavStrongRelation);
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
    /// net's learned HALT) for the resolve%/abstain% signal. READ-ONLY (M3): it does NOT write the engine's shared
    /// persistent self — folding navigator-TRAINING traversals (category hubs) into the self every cycle was the
    /// write-side pollution that derailed held-out walks and forced DAgger off. The vital-loop self-write now lives in
    /// INFERENCE (TryFieldRelax folds the mind's OWN ambiguous conclusion). Returns (resolve%, abstain%): RESOLVE =
    /// confident halt on the cued ancestor, ABSTAIN = no confident halt in the budget.</summary>
    private (double ResolvePct, double AbstainPct) EvaluateNavigatorResolve(
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
                double[]? kindD = null;
                if (q.Kind is { Length: > 0 } kf) { kindD = new double[kf.Length]; for (var d = 0; d < kf.Length; d++) kindD[d] = kf[d]; }
                using var policy = new QueryNavPolicy(net, ds, anchor, q.Cue, device, NavK, 0.0, 0.5, selfVec: null, kindFace: kindD);
                var res = walk.Walk(ds, q.Member, anchor, null, policy, new NavWalkOptions(MaxSteps: NavMaxSteps));
                if (policy.LastHalt)
                {
                    if (string.Equals(res.FinalSymbol, q.Ancestor, StringComparison.Ordinal)) resolved++;
                }
                else abstained++;
                // NO self-write here (M3): a navigator-TRAINING traversal must NOT fold its visited concepts (category
                // hubs) into the SHARED engine self — that was the pollution. The self accumulates only from the mind's
                // own AMBIGUOUS conclusions at inference time (GenesisInferenceEngine.TryFieldRelax).
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

        // Condition the level walk on the learned goal-region for the cue (M4 multi-hop fix) — the same goal training +
        // held-out eval use, so a diagnostic `/nav X domain` descends toward the right region across hops.
        try { EnsureLevelRegions(ds); } catch { }

        try
        {
            using var policy = new QueryNavPolicy(_state.Navigator, ds, anchor, (int)nav, device, NavK, 0.0, 0.5, self,
                kindFace: LevelRegionDouble(nav));
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
                SelfFocus: focus, LastWalk: _lastNavWalk, LastHeldOut: _lastHeldOut);
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
