using GenesisNova.Cognition;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE NAVIGATOR FLOW-FIELD ORACLE  (PLATONIC_NAVIGATOR.md §7, build-order §11.1)
//
//  This is the TEACHER for a future NN walker — NOT a neural net and NOT trained. It is a deterministic
//  backward-Dijkstra "flow field" over the platonic concept graph, lifted in shape from the proven game-AI
//  primitive NavPathfinder.Domain.NavMeshFlowField.Compute (backward Dijkstra from a goal over a neighbour
//  graph, filling cost[] and next[] via a PriorityQueue). We lift the ALGORITHM, not the navmesh geometry
//  (no Triangles / EdgeCost / funnel / LOS — the platonic substrate has none of those).
//
//  Property we are buying (PLATONIC_NAVIGATOR.md §7): computed ONCE per answer and cacheable, cost[] is the
//  dense reward field defined at EVERY reachable node, and next[] is the expert policy at EVERY reachable node
//  — so when a learner strays off the optimal path the field ALREADY has the correct next move there. That is
//  "DAgger for free": no teacher re-query, and a dense distance-to-goal reward at every state the learner can
//  occupy. (A* would label one trajectory only.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// The result of one backward-Dijkstra flow-field computation toward a single <see cref="Target"/> answer.
/// <para><see cref="Cost"/> is the optimal cost-to-answer from each reachable concept (the dense reward field);
/// <see cref="Next"/> is the concept to step to from each reachable concept to move TOWARD the answer (the
/// expert policy at every node). Following <see cref="Next"/> from any reachable node walks to the answer.</para>
/// <para>Keys are concept symbols in the substrate's NORMALIZED form (the same strings the space stores in its
/// relation adjacency and returns from <see cref="IPlatonicSpace.GetNeighbors"/>). Callers should query with
/// already-normalized symbols (lower-case, trimmed) — the form they planted facts with.</para>
/// </summary>
public sealed class PlatonicFlowField
{
    /// <summary>The answer/target symbol this field flows toward. <see cref="Cost"/>[Target] == 0.</summary>
    public string Target { get; }

    /// <summary>Optimal cost-to-answer from each reachable concept (dense reward field, defined everywhere reachable).</summary>
    public IReadOnlyDictionary<string, double> Cost { get; }

    /// <summary>The next concept to step to from each reachable concept, toward the answer (expert policy everywhere).
    /// The target itself has no entry here (it is the destination, not a step-from node).</summary>
    public IReadOnlyDictionary<string, string> Next { get; }

    /// <summary>True if the expansion hit the <c>maxNodes</c>/<c>maxCost</c> bound and the field is a SILENT TRUNCATION
    /// of the full reachable graph (some far nodes may be missing or carry a non-optimal cost). Flagged so callers can
    /// log it — per repo norms a silent cap must never pass unnoticed.</summary>
    public bool Truncated { get; }

    public PlatonicFlowField(
        string target,
        IReadOnlyDictionary<string, double> cost,
        IReadOnlyDictionary<string, string> next,
        bool truncated)
    {
        Target = target;
        Cost = cost;
        Next = next;
        Truncated = truncated;
    }

    /// <summary>Optimal cost-to-answer from <paramref name="from"/>; false if it has no path to the answer.</summary>
    public bool TryCost(string from, out double cost) => Cost.TryGetValue(from, out cost);

    /// <summary>The next concept to step to from <paramref name="from"/> toward the answer; false if none
    /// (either unreachable, or <paramref name="from"/> IS the answer).</summary>
    public bool TryNext(string from, out string next)
    {
        if (Next.TryGetValue(from, out var n)) { next = n; return true; }
        next = string.Empty;
        return false;
    }
}

/// <summary>
/// Deterministic flow-field oracle: a backward Dijkstra from a known answer over the platonic RELATION graph.
/// Mirrors NavPathfinder's <c>NavMeshFlowField.Compute</c> (cost[]/next[] over a neighbour graph), adapted to the
/// concept graph: nodes are concepts, edges are relations enumerated via <see cref="IPlatonicSpace.GetNeighbors"/>.
/// </summary>
public static class FlowFieldOracle
{
    /// <summary>Default expansion bound — how many nodes the field will settle before truncating (PLATONIC_NAVIGATOR §7,
    /// "a huge graph doesn't blow up"). ~4096 keeps a per-answer field cheap and cacheable.</summary>
    public const int DefaultMaxNodes = 4096;

    // Relations index BOTH endpoints, so adjacency is symmetric and a relation traverses cleanly in REVERSE
    // (backward Dijkstra from the answer is just forward Dijkstra over the undirected relation graph). We ask
    // the space for up to this many relational neighbours per node (GetNeighbors itself clamps to 64).
    private const int NeighborFanout = 64;

    /// <summary>
    /// Compute the flow field toward <paramref name="answer"/> by backward Dijkstra over the relation graph.
    /// <para>EDGE COST: 1.0 per hop by default — so <c>cost[node]</c> is the hop distance to the answer (the simplest,
    /// most legible first cut; PLATONIC_NAVIGATOR §7 sanctions starting here). Set <paramref name="useStrengthCost"/>
    /// to instead weight each hop by <c>1.0 - strength</c> (strength = relation Confidence in (0,1]; a strong, oft-seen
    /// edge costs less), floored at <see cref="MinEdgeCost"/> so every hop still has positive cost.</para>
    /// </summary>
    /// <param name="space">The platonic substrate (production: DialecticalSpace).</param>
    /// <param name="answer">The known answer/target symbol (already normalized — lower-case, trimmed).</param>
    /// <param name="maxNodes">Settle-count cap; on reaching it the field is truncated and <see cref="PlatonicFlowField.Truncated"/> set.</param>
    /// <param name="maxCost">Optional cost/hop horizon: neighbours beyond this are not expanded (cognitive light-cone, §8). Default unbounded.</param>
    /// <param name="useStrengthCost">false = 1.0/hop (default); true = (1 - edge strength) per hop, floored at <see cref="MinEdgeCost"/>.</param>
    /// <param name="minEdgeConfidence">Ignore relations weaker than this (noise floor). Default 0 = follow every edge.</param>
    /// <param name="onTruncate">Optional sink for the truncation notice (log it — a silent cap is a repo no-no).</param>
    public static PlatonicFlowField Compute(
        IPlatonicSpace space,
        string answer,
        int maxNodes = DefaultMaxNodes,
        double maxCost = double.PositiveInfinity,
        bool useStrengthCost = false,
        double minEdgeConfidence = 0.0,
        Action<string>? onTruncate = null)
    {
        ArgumentNullException.ThrowIfNull(space);
        if (string.IsNullOrWhiteSpace(answer)) throw new ArgumentException("answer must be a concept symbol", nameof(answer));

        // cost[answer] = 0; everything else implicitly +inf (absent from the dict). next[] starts empty.
        // (NavMeshFlowField uses dense arrays over a fixed triangle count; the concept graph is sparse and string-keyed,
        //  so we use dictionaries — same algorithm, sparse storage.)
        var cost = new Dictionary<string, double>(StringComparer.Ordinal) { [answer] = 0.0 };
        var next = new Dictionary<string, string>(StringComparer.Ordinal);
        var settled = new HashSet<string>(StringComparer.Ordinal);

        var queue = new PriorityQueue<string, double>();
        queue.Enqueue(answer, 0.0);

        var truncated = false;

        while (queue.TryDequeue(out var current, out var priority))
        {
            // Lazy-deletion: a node can sit in the queue multiple times; only process the first (lowest-cost) pop.
            if (!settled.Add(current)) continue;
            // The dict is the source of truth; skip a stale queue entry whose recorded cost was since improved.
            if (priority > cost[current]) continue;

            // Expansion bound — Dijkstra settles in non-decreasing cost order, so the FIRST maxNodes settled are the
            // closest-to-answer nodes; truncating here drops only far nodes (PLATONIC_NAVIGATOR §7). Flag it loudly.
            if (settled.Count > maxNodes)
            {
                truncated = true;
                break;
            }

            var currentCost = cost[current];

            // FOLLOW-edge, reversed: a concept's relational neighbours. Relations index both endpoints, so this is the
            // reverse graph for free. (TODO §5.1/§7: this FIRST CUT is the RELATION graph only — lattice-neighbour
            // STEP-near moves and reversible COMPUTE-jumps are NOT yet edges here; add them as the action seams land.)
            var neighbours = space.GetNeighbors(current, PlatonicNeighborhoodType.Relational, NeighborFanout, minEdgeConfidence);
            for (var i = 0; i < neighbours.Count; i++)
            {
                var n = neighbours[i];
                if (settled.Contains(n.Concept)) continue; // already optimal, skip (Dijkstra)

                var edge = useStrengthCost ? Math.Max(MinEdgeCost, 1.0 - n.Confidence) : 1.0;
                var newCost = currentCost + edge;
                if (newCost > maxCost) continue; // beyond the horizon

                if (cost.TryGetValue(n.Concept, out var old) && newCost >= old) continue; // no improvement (relax test)

                cost[n.Concept] = newCost;
                next[n.Concept] = current; // step from neighbour -> current heads TOWARD the answer
                queue.Enqueue(n.Concept, newCost);
            }
        }

        if (truncated)
            onTruncate?.Invoke($"FlowFieldOracle: field toward '{answer}' truncated at maxNodes={maxNodes} " +
                               $"(settled {settled.Count}); far nodes may be missing or non-optimal.");

        return new PlatonicFlowField(answer, cost, next, truncated);
    }

    /// <summary>Positive floor on a single hop's cost when <c>useStrengthCost</c> is on, so a near-certain edge
    /// (strength≈1 → 1-strength≈0) still advances cost monotonically and Dijkstra terminates.</summary>
    public const double MinEdgeCost = 1e-3;
}
