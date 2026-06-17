namespace GenesisNova.Cognition;

public enum PlatonicNeighborhoodType
{
    Any = 0,
    Semantic = 1,
    Relational = 2,
    Numeric = 3
}

public readonly record struct PlatonicNeighbor(
    string Concept,
    double Confidence,
    PlatonicNeighborhoodType Type,
    int ObservationCount);

/// <summary>
/// A relation as a FIRST-CLASS, POSITIONED ELEMENT of the platonic space (genesis-faithful Kind=Relation):
/// its two endpoints, a strength (1 − synthesis contradiction), and a POSITION at the centroid of the
/// endpoints' faces — so the relation is itself a positioned object that can be related/composed
/// (higher-order relations). This is the canonical, immutable VIEW of a <see cref="RelationElementNode"/>
/// (the mutable element behind the keyed <c>_relationIndex</c>); <see cref="PlatonicSpaceMemory.GetRelationElements"/>
/// projects the index into these positioned elements.
/// </summary>
public readonly record struct RelationElement(
    string Left,
    string Right,
    double Strength,
    double[] Embedding);

public sealed class PlatonicSpaceMemory
{
    // Capacity caps (hard eviction). The defaults here are the standalone/test fallback; the RUNTIME
    // passes these from GenesisNovaConfig (the single source of truth), so the memory's eviction cap and
    // the SpaceManager's maintenance pruning use the SAME limits instead of two independent constants.
    private const int DefaultMaxPlatonicNodes = 12_000;
    private const int DefaultMaxPlatonicRelations = 48_000;
    private readonly int _maxPlatonicNodes;
    private readonly int _maxPlatonicRelations;
    private const double GeometryLearningRate = 0.04;
    // ATTRACTION FLOOR: an OBSERVED relation is a structural fact and must be PULLED, even when the edit head's
    // magnitude m is weak (live attraction strength = m via contradiction=0.5−0.5m, so a timid head starves the
    // pull → related drifts as far as unrelated under the now-effective repulsion → flat separation). Floor the
    // attracting affinity so the pull BALANCES the push; repelling pairs (affinity<0) are untouched. Gated by
    // PlatonicGeometryDynamicsTests (weak-attraction scenario must still separate).
    private const double MinAttractAffinity = 0.5;

    // Contrastive repulsion — ported from the genesis source of truth (GraphAligner.NudgeGraphAlignment,
    // which "push[es] away from sampled non-neighbours ... maintain[s] discriminability between unrelated
    // elements"). Nova previously ONLY attracted observed pairs, so unrelated/fringe concepts drifted
    // together. This throttled batch pass pushes each concept away from a few sampled UNRELATED concepts
    // so proximity must be EARNED by confirmed relations. Runs every RepulsionInterval observations.
    // Strength is genesis-faithful: repulsion is a GENTLE discriminability pressure (~0.1x attraction),
    // not a strong separator. A stronger setting separates unrelated concepts more but overpowers weak
    // genuine attractions (e.g. text→frozen-number equivalence), so we keep the source-of-truth scale.
    // REBALANCED (2026): the old "~0.1× attraction" repulsion was ~65× weaker than attraction per observation,
    // so the semantic space COLLAPSED into a cone (measured: related FARTHER than unrelated). Repulsion now
    // runs far more often, with more samples and meaningfully stronger force, so unrelated concepts are pushed
    // toward orthogonality while edged (related) pairs — which are EXEMPT from repulsion — stay pulled. Gated
    // by PlatonicGeometryDynamicsTests (related must sit clearly closer than unrelated).
    private const int RepulsionInterval = 4;
    private const int RepulsionSamples = 10;
    private const double RepulsionRate = 0.02;
    private const double RepulsionRatio = 0.5; // repulsion strength relative to RepulsionRate floor
    private int _observationsSinceRepulsion;
    private int _repulsionPassCount;

    public sealed record SpaceMaintenanceRequest(
        int MaxRelationPrunes = 64,
        int MaxNodePrunes = 16,
        int MaxNodeMerges = 8,
        int MaxRebalancePrunes = 64,
        int MaxRelationsPerNode = 0,
        int TargetRelationCount = 0,
        int MinRelationsToKeep = 512,
        int MinNodesToKeep = 128,
        int MinRelationObservationToKeep = 2,
        int MinNodeObservationToKeep = 1,
        double MaxSynthesisContradictionToKeep = 0.75,
        double MergeDistanceThreshold = 0.018,
        IReadOnlyCollection<string>? ProtectedConcepts = null);

    public sealed record SpaceMaintenanceResult(
        int RelationsPruned,
        int NodesPruned,
        int NodesMerged);

    private readonly int _faceDimension;
    // Two element collections, each a keyed INDEX into first-class elements of the space (NOT side-graphs):
    //  • _nodes        — concept elements (objects), positioned by their faces.
    //  • _relationIndex — RELATION elements, each positioned at the centroid of its endpoints (see
    //    RelationElementNode / GetRelationElements). The dict is purely the O(1) access index over the
    //    relation-elements — exactly as _nodes indexes concept-elements — so relations are themselves
    //    positioned, composable objects in the substrate, not an opaque graph layered on top of it.
    private readonly Dictionary<string, ConceptNode> _nodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, RelationElementNode> _relationIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly PlatonicLattice _lattice;
    private long _utilityStep;

    public PlatonicSpaceMemory(
        int faceDimension,
        int seed = 42,
        int maxNodes = DefaultMaxPlatonicNodes,
        int maxRelations = DefaultMaxPlatonicRelations)
    {
        _faceDimension = Math.Max(4, faceDimension);
        _functions = new FunctionElementRegistry(_faceDimension);
        _words = new WordElementRegistry(_faceDimension);
        _maxPlatonicNodes = Math.Max(256, maxNodes);
        _maxPlatonicRelations = Math.Max(1024, maxRelations);
        _lattice = new PlatonicLattice(
            nodeNames: () => _nodes.Keys,
            nodeFaces: () => _nodes.Values.Select(n => (n.Name, n.PositiveFace)));
    }

    public int NodeCount => _nodes.Count;
    public int RelationCount => _relationIndex.Count;
    public int FaceDimension => _faceDimension;

    public IReadOnlyList<string> Concepts
        => _nodes.Keys.OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();

    public bool ContainsConcept(string concept)
        => _nodes.ContainsKey(Normalize(concept));

    // Op-token registry. An operation token — a language's verb (e.g. "find"), like arithmetic's role-token —
    // is a ROUTE TRIGGER, never a relation participant. It recurs in EVERY example of its operation, so if it
    // formed relation edges it would acquire its strongest edge to the most-frequent target and collapse all
    // queries to one result — the same failure class as number↔number edges. Registered tokens are excluded
    // from concept extraction (trainer coupling + route-label anchoring AND inference anchoring) while the GRU
    // route head still sees the raw input tokens and learns the route. Re-registered from the language
    // definition at startup; not persisted (it is a pure function of the definition). See LANGUAGE_CREATOR.md §2.
    private readonly HashSet<string> _operationTokens = new(StringComparer.OrdinalIgnoreCase);

    public void RegisterOperationToken(string token)
    {
        if (!string.IsNullOrWhiteSpace(token))
            _operationTokens.Add(Normalize(token));
    }

    public bool IsOperationToken(string concept)
        => _operationTokens.Count > 0 && _operationTokens.Contains(Normalize(concept));

    /// <summary>Reserved INTERNAL concepts — the op→face routing markers ("face:poly"/"face:log"). They are
    /// valid relation ENDPOINTS (the op↔face affinity that routes arithmetic) but are hyper-observed hubs, so
    /// they must NEVER be returned as a retrieval ANSWER, used as a retrieval anchor, or shown as an activated
    /// concept. They are not user-facing concepts.</summary>
    public static bool IsReservedConcept(string? concept)
        => !string.IsNullOrEmpty(concept) && concept.StartsWith("face:", System.StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Returns the positive face of a concept without side effects.
    /// For numeric concepts not in the space, returns their seeded face (homomorphic structure preserved).
    /// Returns false only for non-numeric unseen concepts.
    /// </summary>
    public bool TryGetConceptFace(string concept, out double[] positiveFace)
    {
        // Numbers always use the mathematical (homomorphic) face so that face arithmetic stays
        // exact regardless of whether a node was created by training side-effects.
        if (TryParseNumber(concept, out var numeric))
        {
            positiveFace = CreateNumericFace(numeric);
            return true;
        }
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
        {
            positiveFace = node.PositiveFace;
            return true;
        }
        positiveFace = Array.Empty<double>();
        return false;
    }

    /// <summary>
    /// Returns nearest known concepts to <paramref name="concept"/> in face space.
    /// Optionally restricts to a candidate set to keep per-turn work bounded.
    /// </summary>
    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConcepts(
        string concept,
        IReadOnlyCollection<string>? candidates = null,
        int maxNeighbors = 8,
        int maxCandidates = 96)
    {
        var limit = Math.Clamp(maxNeighbors, 1, 32);
        if (!TryGetConceptFace(concept, out var conceptFace) || conceptFace.Length == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        var conceptKey = Normalize(concept);

        // Bounded candidate set: score those directly to keep per-turn work bounded.
        if (candidates is { Count: > 0 })
        {
            var budget = Math.Clamp(maxCandidates, 8, 512);
            var scored = new List<(string Symbol, double Distance)>(Math.Min(budget, limit * 4));
            var seen = 0;
            foreach (var candidate in candidates)
            {
                if (seen >= budget)
                    break;
                var normalized = Normalize(candidate);
                if (normalized.Equals(conceptKey, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (!_nodes.TryGetValue(normalized, out var node))
                    continue;
                scored.Add((normalized, FaceAwareDistance(conceptFace, node.PositiveFace)));
                seen++;
            }
            return scored.Count == 0
                ? Array.Empty<(string Symbol, double Distance)>()
                : scored.OrderBy(x => x.Distance).Take(limit).ToArray();
        }

        // Global nearest: O(log N) VP-Tree over the whole space (replaces the brute-force O(N) scan).
        return _lattice.GetSemanticNeighbors(conceptFace, limit, conceptKey);
    }

    /// <summary>The magnitude of separation the contrastive dynamics (pull related / push unrelated) achieved.</summary>
    public sealed record GeometrySummary(
        int TotalConcepts, int MutableConcepts, int RelatedPairs, int UnrelatedPairs,
        double RelatedMean, double RelatedMin, double RelatedMax,
        double UnrelatedMean, double UnrelatedMin, double UnrelatedMax)
    {
        /// <summary>How much farther apart unrelated concepts sit than related ones — the push/pull gap.</summary>
        public double Separation => UnrelatedMean - RelatedMean;
    }

    /// <summary>
    /// PUSH/PULL geometry summary (READ-ONLY): over a sample of MUTABLE (non-frozen) concepts, the semantic-face
    /// distance to relation-edged neighbours (the PULL — should be SMALL) vs. to random UNRELATED concepts (the
    /// PUSH — should be LARGE). Faces are unit-normalised so distances live in [0, 2]; <c>Separation</c> =
    /// unrelatedMean − relatedMean is the magnitude the message-passing + contrastive repulsion actually moved.
    /// </summary>
    public GeometrySummary SummarizePushPullGeometry(int maxConcepts = 600, int unrelatedPerNode = 8, int seed = 1234)
    {
        var rng = new Random(seed);
        var mutable = _nodes.Values.Where(n => !IsFrozenConcept(n.Name) && n.PositiveFace is { Length: > 0 }).ToArray();
        var sample = mutable.Length <= maxConcepts ? mutable : mutable.OrderBy(_ => rng.Next()).Take(maxConcepts).ToArray();

        var related = new List<double>();
        var unrelated = new List<double>();
        foreach (var node in sample)
        {
            // PULL: distance to each relation-edged neighbour.
            foreach (var rn in _lattice.GetRelationalNeighbors(node.Name))
                if (_nodes.TryGetValue(Normalize(rn), out var other) && !IsFrozenConcept(other.Name) && other.PositiveFace.Length > 0)
                    related.Add(FaceAwareDistance(node.PositiveFace, other.PositiveFace));
            // PUSH: distance to random UNRELATED concepts (no relation edge).
            for (var s = 0; s < unrelatedPerNode && mutable.Length > 1; s++)
            {
                var other = mutable[rng.Next(mutable.Length)];
                if (ReferenceEquals(other, node))
                    continue;
                if (_relationIndex.ContainsKey(RelationKey(node.Name, other.Name)))
                    continue;
                unrelated.Add(FaceAwareDistance(node.PositiveFace, other.PositiveFace));
            }
        }

        static (double Mean, double Min, double Max) St(List<double> xs)
            => xs.Count == 0 ? (0.0, 0.0, 0.0) : (xs.Average(), xs.Min(), xs.Max());
        var r = St(related);
        var u = St(unrelated);
        return new GeometrySummary(_nodes.Count, mutable.Length, related.Count, unrelated.Count,
            r.Mean, r.Min, r.Max, u.Mean, u.Min, u.Max);
    }

    /// <summary>
    /// LIVE-FACE nearest concepts for WITHIN-STEP edit verification / perception. The global
    /// <see cref="GetNearestConcepts"/> reads the throttled VP-Tree, so a face that just moved this step is not
    /// reflected until the next rebuild — which staled the edit-head's pre/post retrievability DELTA and the
    /// route/edit perceptions (the controller couldn't "see" that a correct edit landed). This instead assembles
    /// a BOUNDED candidate set — the caller's <paramref name="seeds"/> (e.g. the example's target concepts), the
    /// anchor's relational neighbours (Layer-1, always current), and the anchor's current VP-Tree neighbourhood
    /// used ONLY as a candidate-NAME source — then re-scores every candidate against LIVE faces. So everything
    /// that moved is measured at its current position, in the same step, at O(candidates) cost (no rebuild). The
    /// periodic full rebuild remains the authority for discovering brand-new neighbourhoods across the space.
    /// </summary>
    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConceptsFresh(
        string concept, IReadOnlyCollection<string>? seeds = null, int maxNeighbors = 8)
    {
        if (!TryGetConceptFace(concept, out var conceptFace) || conceptFace.Length == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        var conceptKey = Normalize(concept);
        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (seeds is not null)
            foreach (var s in seeds)
            {
                var n = Normalize(s);
                if (!n.Equals(conceptKey, StringComparison.OrdinalIgnoreCase))
                    candidates.Add(n);
            }

        // Relational neighbours (Layer-1 adjacency is maintained incrementally → always current names).
        foreach (var n in _lattice.GetRelationalNeighbors(conceptKey))
            candidates.Add(n);

        // VP-Tree neighbourhood of the anchor's CURRENT face — used only to harvest candidate NAMES (their
        // stored positions may be slightly stale); they are RE-SCORED against live faces by GetNearestConcepts.
        var pool = Math.Clamp(maxNeighbors * 3, 8, 64);
        foreach (var (n, _) in _lattice.GetSemanticNeighbors(conceptFace, pool, conceptKey))
            candidates.Add(n);

        candidates.Remove(conceptKey);
        return candidates.Count == 0
            ? Array.Empty<(string Symbol, double Distance)>()
            : GetNearestConcepts(concept, candidates: candidates, maxNeighbors: maxNeighbors);
    }

    /// <summary>TARGET-AGNOSTIC route-perception vector (SPACE_AWARE_GRU.md §I): "can the space answer a query
    /// anchored here?" — usable at INFERENCE where the target is unknown. [has-neighbour, nearest-confidence,
    /// degree-norm, mean-top-confidence, transform-reliability, bias]; dims match
    /// <c>GenesisNeuralModel.EditPerceptionDim</c>. <paramref name="transformReliability"/> is the EARNED UCB
    /// reliability of the model's best applicable learned transform (0 when none / when the feature is off) —
    /// it bubbles proven transform capability up to the route head so it learns which functions to trust.</summary>
    public double[] ComputeRoutePerception(string anchor, double transformReliability = 0.0)
    {
        // LIVE faces (not the throttled tree) so the perception reflects edits made this step — see
        // GetNearestConceptsFresh. Target-agnostic here (no seeds): relational + current-neighbourhood candidates.
        var near = GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 4);
        var hasNeighbour = near.Count > 0 ? 1.0 : 0.0;
        var nearestConf = near.Count > 0 ? 1.0 / (1.0 + Math.Max(0.0, near[0].Distance)) : 0.0;
        var meanTopConf = near.Count > 0 ? near.Average(n => 1.0 / (1.0 + Math.Max(0.0, n.Distance))) : 0.0;
        var degreeNorm = Math.Clamp(GetRelationDegree(Normalize(anchor)) / 8.0, 0.0, 1.0);
        return new[] { hasNeighbour, nearestConf, degreeNorm, meanTopConf, Math.Clamp(transformReliability, 0.0, 1.0), 1.0 };
    }

    public int NumericDimensions => Math.Min(_faceDimension / 2, 21);
    public int LogFaceStart => NumericDimensions;

    public void ObserveContradiction(string left, string right, double observedContradiction)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            return;
        if (left.Equals(right, StringComparison.OrdinalIgnoreCase))
            return;

        var a = GetOrCreate(left, new[] { right });
        var b = GetOrCreate(right, new[] { left });
        a.ObservationCount++;
        b.ObservationCount++;

        var key = RelationKey(left, right);
        if (!_relationIndex.TryGetValue(key, out var relation))
        {
            EnsureRelationCapacity();
            relation = new RelationElementNode(
                left: Normalize(left),
                right: Normalize(right),
                thesisContradiction: observedContradiction,
                lastObservedContradiction: observedContradiction,
                synthesisContradiction: observedContradiction,
                observationCount: 0,
                useCount: 0,
                successCount: 0,
                failureCount: 0,
                lastUsedStep: 0);
            _relationIndex[key] = relation;
            IndexRelation(relation);
        }

        relation.LastObservedContradiction = Clamp01(observedContradiction);
        relation.SynthesisContradiction = relation.ObservationCount == 0
            ? relation.LastObservedContradiction
            : (0.85 * relation.SynthesisContradiction) + (0.15 * relation.LastObservedContradiction);
        relation.ObservationCount++;

        // Academic update: mutate ONLY the directly-observed pair. No triadic synthesis — pulling
        // unrelated third concepts toward this pair is cross-contamination, not signal.
        UpdateConceptGeometry(a, b, relation.SynthesisContradiction, 1.0);

        // Periodic contrastive repulsion keeps unrelated concepts discriminable (genesis parity).
        if (++_observationsSinceRepulsion >= RepulsionInterval)
        {
            _observationsSinceRepulsion = 0;
            ApplyContrastiveRepulsionPass();
        }
    }

    public double GetContradiction(string left, string right)
    {
        var key = RelationKey(left, right);
        return _relationIndex.TryGetValue(key, out var relation)
            ? relation.SynthesisContradiction
            : 0.5;
    }

    /// <summary>
    /// Projects the relation index into its canonical positioned relation-ELEMENTS (see
    /// <see cref="RelationElement"/>): each relation is an element located at the CENTROID of its endpoints'
    /// faces, strength = 1 − synthesis contradiction. This is the substrate-native form — relations are
    /// positioned objects that can themselves be related/composed (higher-order); the keyed index simply
    /// provides O(1) access to them (the same role <c>_nodes</c> plays for concept-elements).
    /// </summary>
    public IReadOnlyList<RelationElement> GetRelationElements()
    {
        var result = new List<RelationElement>(_relationIndex.Count);
        foreach (var rel in _relationIndex.Values)
        {
            var embedding = Array.Empty<double>();
            if (TryGetConceptFace(rel.Left, out var lf) && TryGetConceptFace(rel.Right, out var rf)
                && lf.Length == rf.Length && lf.Length > 0)
            {
                embedding = new double[lf.Length];
                for (var i = 0; i < lf.Length; i++)
                    embedding[i] = 0.5 * (lf[i] + rf[i]); // centroid: the relation sits between its endpoints
            }
            result.Add(new RelationElement(rel.Left, rel.Right, rel.Strength, embedding));
        }
        return result;
    }

    /// <summary>
    /// Strongest related neighbour of a concept by TRAVERSING relation-elements — the canonical
    /// element-native retrieval. Returns the other endpoint of the highest-strength relation-element
    /// referencing the concept.
    /// </summary>
    public bool TryRelationElementNeighbour(string concept, out string neighbour, out double strength)
    {
        neighbour = string.Empty;
        strength = 0.0;
        if (string.IsNullOrWhiteSpace(concept))
            return false;
        var key = Normalize(concept);
        var found = false;
        foreach (var re in GetRelationElements())
        {
            string? other = null;
            if (string.Equals(re.Left, key, StringComparison.OrdinalIgnoreCase)) other = re.Right;
            else if (string.Equals(re.Right, key, StringComparison.OrdinalIgnoreCase)) other = re.Left;
            if (other is null) continue;
            if (!found || re.Strength > strength)
            {
                neighbour = other;
                strength = re.Strength;
                found = true;
            }
        }
        return found;
    }

    // Shapes-as-Function-elements and the Seq chunk-element store are each their own class (SRP); the memory
    // delegates. Both are kept out of _nodes so they never contaminate concept retrieval. The executable
    // glider definitions live in PlatonicShapeRegistry; the Function elements here are their substrate existence.
    private readonly FunctionElementRegistry _functions;
    private readonly WordElementRegistry _words;
    private readonly ChunkElementStore _chunks = new();

    /// <summary>The canonical tag under which the Seq composer mines/looks-up its scaffold chunk.</summary>
    public const string SeqScaffoldTag = "⟨seq-scaffold⟩";

    /// <summary>The Function-elements registered in the space (shapes-as-elements).</summary>
    public IReadOnlyList<Core.PlatonicElement> FunctionElements => _functions.Elements;

    /// <summary>Register (idempotently) a shape as a positioned <see cref="Core.ElementKind.Function"/> element.</summary>
    public Core.PlatonicElement RegisterFunctionElement(string name, IReadOnlyList<string>? references = null)
        => _functions.Register(name, references);

    public bool TryGetFunctionElement(string name, out Core.PlatonicElement element)
        => _functions.TryGet(name, out element);

    /// <summary>The WORD ELEMENTS registered in the space (whole-string identities in the word face).</summary>
    public IReadOnlyList<Core.PlatonicElement> WordElements => _words.Elements;

    /// <summary>
    /// Register (idempotently) a concept as a first-class WORD ELEMENT — a distinct element whose identity
    /// lives in the word face (spelling-independent), NOT a region of the concept's char face. A MULTI-word
    /// concept auto-registers each constituent word element FIRST and RELATES to them (concat); reading those
    /// related parts is decompose (<see cref="DecomposeWordElement"/>).
    /// </summary>
    public Core.PlatonicElement RegisterWordElement(string concept)
    {
        var key = (concept ?? string.Empty).Trim();
        var tokens = key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 2)
            return _words.Register(key, null);                 // atomic word element
        foreach (var t in tokens)
            _words.Register(t, null);                          // ensure each part exists first
        return _words.Register(key, tokens);                   // the whole, relating to its parts (concat)
    }

    public bool TryGetWordElement(string concept, out Core.PlatonicElement element)
        => _words.TryGet(concept, out element);

    /// <summary>DECOMPOSE a (multi-word) word element into its constituent word elements — the inverse of the
    /// concat structure, read from the element's <c>RelatedTo</c>. Empty for an atomic / unregistered concept.</summary>
    public IReadOnlyList<Core.PlatonicElement> DecomposeWordElement(string concept)
        => _words.TryGet(concept, out var e) ? _words.Parts(e) : Array.Empty<Core.PlatonicElement>();

    /// <summary>Record one observation of <paramref name="chunk"/> as a scaffold for <paramref name="tag"/>.</summary>
    public void MineChunk(string tag, string chunk) => _chunks.Mine(tag, chunk);

    /// <summary>The most-reinforced chunk mined for <paramref name="tag"/> (false if none yet).</summary>
    public bool TryGetTopChunk(string tag, out string chunk) => _chunks.TryGetTop(tag, out chunk);

    public PlatonicMemorySnapshot ExportSnapshot()
    {
        return new PlatonicMemorySnapshot(
            FaceDimension: _faceDimension,
            Nodes: _nodes.Values
               .Select(n => new PlatonicNodeSnapshot(
                   n.Name,
                   n.PositiveFace.ToArray(),
                   n.NegativeFace.ToArray(),
                   n.ObservationCount,
                   n.UseCount,
                   n.SuccessCount,
                   n.FailureCount,
                   n.LastUsedStep))
               .ToArray(),
            Relations: _relationIndex.Values
               .Select(r => new PlatonicRelationSnapshot(
                   r.Left,
                   r.Right,
                   r.ThesisContradiction,
                   r.LastObservedContradiction,
                   r.SynthesisContradiction,
                   r.ObservationCount,
                   r.UseCount,
                   r.SuccessCount,
                   r.FailureCount,
                   r.LastUsedStep))
               .ToArray(),
            Chunks: _chunks.Export());
    }

    public PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops = 2,
        int beamWidth = 2)
        => QueryConceptChain(anchorConcepts, maxHops, beamWidth, out _);

    public PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops,
        int beamWidth,
        out IReadOnlyList<PlatonicEvidence> evidence)
    {
        evidence = Array.Empty<PlatonicEvidence>();
        var anchors = NormalizeConcepts(anchorConcepts)
            .Where(ContainsConcept)
            .ToArray();
        if (anchors.Length == 0)
            return new PlatonicQueryResult(string.Empty, 0.0, 0, 0);

        var hops = Math.Clamp(maxHops, 1, 6);
        var beam = Math.Clamp(beamWidth, 1, 4);
        var seen = new HashSet<string>(anchors, StringComparer.OrdinalIgnoreCase);
        var frontier = anchors;
        var decoded = new List<string>();
        var confidences = new List<double>();
        var evidenceTrail = new List<PlatonicEvidence>();

        for (var hop = 0; hop < hops; hop++)
        {
            var candidateScores = new Dictionary<string, List<double>>(StringComparer.OrdinalIgnoreCase);
            var candidateEvidence = new Dictionary<string, List<PlatonicEvidence>>(StringComparer.OrdinalIgnoreCase);
            foreach (var source in frontier)
            {
                // Relational-first retrieval: LEARNED relations are observed facts ("3" relates to
                // "three"); numeric value-proximity ("3" is near "4") and face KNN are geometry, not
                // relatedness — and for numbers they always win on full-vector distance (poly/log face
                // clusters values), returning the adjacent number instead of the related concept. So
                // prefer the relational tier; fall back to Any (semantic/numeric) only when the concept
                // has NO learned relations. General: 3→three, him→paul, apple→fruit.
                var neighbors = GetNeighbors(
                    source,
                    type: PlatonicNeighborhoodType.Relational,
                    maxNeighbors: Math.Max(8, beam * 8),
                    minConfidence: 0.35);
                if (neighbors.Count == 0)
                    neighbors = GetNeighbors(
                        source,
                        type: PlatonicNeighborhoodType.Any,
                        maxNeighbors: Math.Max(8, beam * 8),
                        minConfidence: 0.35);

                foreach (var neighbor in neighbors)
                {
                    var target = neighbor.Concept;
                    if (seen.Contains(target))
                        continue;
                    var confidence = neighbor.Confidence;

                    if (!candidateScores.TryGetValue(target, out var list))
                    {
                        list = new List<double>();
                        candidateScores[target] = list;
                    }
                    list.Add(confidence);
                    if (!candidateEvidence.TryGetValue(target, out var evList))
                    {
                       evList = new List<PlatonicEvidence>();
                       candidateEvidence[target] = evList;
                    }
                    evList.Add(new PlatonicEvidence(target, source, confidence / (hop + 1.0), hop + 1));
                }
            }

            if (candidateScores.Count == 0)
                break;

            var selected = candidateScores
                .Select(kvp => new
                {
                    Concept = kvp.Key,
                    Score = kvp.Value.Average()
                })
                .OrderByDescending(x => x.Score)
                .Take(beam)
                .ToArray();

            if (selected.Length == 0)
                break;

            foreach (var item in selected)
            {
                seen.Add(item.Concept);
                decoded.Add(item.Concept);
                confidences.Add(item.Score);
                if (candidateEvidence.TryGetValue(item.Concept, out var evList) && evList.Count > 0)
                {
                    evidenceTrail.AddRange(evList
                        .OrderByDescending(e => Math.Abs(e.Contribution))
                        .Take(2));
                }
            }

            frontier = selected.Select(x => x.Concept).ToArray();
        }

        if (decoded.Count == 0)
        {
            var first = anchors[0];
            evidence = new[] { new PlatonicEvidence(first, null, 0.42, 1) };
            return new PlatonicQueryResult(
                Text: first,
                Confidence: 0.42,
                Hops: 1,
                ConceptCount: 1);
        }

        evidence = evidenceTrail
            .GroupBy(e => (Concept: Normalize(e.Concept), Related: e.RelatedConcept is null ? null : Normalize(e.RelatedConcept), e.Hop))
            .Select(g => new PlatonicEvidence(g.Key.Concept, g.Key.Related, g.Sum(e => e.Contribution), g.Key.Hop))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(32)
            .ToArray();
        return new PlatonicQueryResult(
            Text: string.Join(' ', decoded),
            Confidence: Clamp01(confidences.DefaultIfEmpty(0.0).Average()),
            Hops: Math.Min(hops, Math.Max(1, decoded.Count)),
            ConceptCount: decoded.Count);
    }

    public IReadOnlyList<PlatonicNeighbor> GetNeighbors(
        string concept,
        PlatonicNeighborhoodType type = PlatonicNeighborhoodType.Any,
        int maxNeighbors = 16,
        double minConfidence = 0.0)
    {
        if (string.IsNullOrWhiteSpace(concept))
            return Array.Empty<PlatonicNeighbor>();

        var key = Normalize(concept);
        var capped = Math.Clamp(maxNeighbors, 1, 256);
        var threshold = Clamp01(minConfidence);
        var collected = new Dictionary<string, PlatonicNeighbor>(StringComparer.OrdinalIgnoreCase);

        void Consider(PlatonicNeighbor candidate)
        {
            if (candidate.Confidence < threshold)
                return;
            if (string.Equals(candidate.Concept, key, StringComparison.OrdinalIgnoreCase))
                return;
            if (IsReservedConcept(candidate.Concept)) // internal op→face markers (face:poly/log) are never neighbours/answers
                return;
            if (!collected.TryGetValue(candidate.Concept, out var existing) || candidate.Confidence > existing.Confidence)
                collected[candidate.Concept] = candidate;
        }

        var wantRelational = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Relational;
        var wantNumeric = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Numeric;
        var wantSemantic = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Semantic;

        // Relational tier: traverse the concept's RELATION-ELEMENTS — the adjacency index gives O(1) access
        // to them, and each element's Strength (1 − synthesis contradiction) is the confidence. This is the
        // element-native retrieval; the index is just the access path, not a separate graph.
        if (wantRelational)
        {
            foreach (var neighbor in _lattice.GetRelationalNeighbors(key))
            {
                if (!_relationIndex.TryGetValue(RelationKey(key, neighbor), out var relation))
                    continue;
                Consider(new PlatonicNeighbor(
                    Concept: neighbor,
                    Confidence: relation.Strength,
                    Type: ClassifyConceptType(neighbor),
                    ObservationCount: relation.ObservationCount));
            }
        }

        // Numeric tier: value-proximity neighbours ("position IS address"); confidence from distance.
        if (wantNumeric && TryParseNumber(key, out var numericValue))
        {
            foreach (var (name, distance) in _lattice.GetNumericNeighbors(numericValue, range: capped, k: capped))
                Consider(new PlatonicNeighbor(
                    Concept: name,
                    Confidence: DistanceConfidence(distance),
                    Type: PlatonicNeighborhoodType.Numeric,
                    ObservationCount: NodeObservation(name)));
        }

        // Semantic tier: embedding KNN via VP-Tree; confidence from face distance.
        if (wantSemantic && TryGetConceptFace(key, out var face) && face.Length > 0)
        {
            foreach (var (name, distance) in _lattice.GetSemanticNeighbors(face, capped, key))
                Consider(new PlatonicNeighbor(
                    Concept: name,
                    Confidence: DistanceConfidence(distance),
                    Type: PlatonicNeighborhoodType.Semantic,
                    ObservationCount: NodeObservation(name)));
        }

        if (collected.Count == 0)
            return Array.Empty<PlatonicNeighbor>();

        return collected.Values
            .OrderByDescending(n => n.Confidence)
            .ThenBy(n => n.Concept, StringComparer.OrdinalIgnoreCase)
            .Take(capped)
            .ToArray();
    }

    // Map a non-negative face/value distance to a confidence in (0, 1]; nearer ⇒ more confident.
    private static double DistanceConfidence(double distance) => 1.0 / (1.0 + Math.Max(0.0, distance));

    private int NodeObservation(string concept)
        => _nodes.TryGetValue(Normalize(concept), out var node) ? node.ObservationCount : 0;

    public IReadOnlyList<string> GetAdjacentConcepts(
        string concept,
        PlatonicNeighborhoodType type = PlatonicNeighborhoodType.Any,
        int maxNeighbors = 16,
        double minConfidence = 0.0)
    {
        return GetNeighbors(concept, type, maxNeighbors, minConfidence)
            .Select(n => n.Concept)
            .ToArray();
    }

    public void FineEditFromExample(
        IReadOnlyList<string> inputConcepts,
        IReadOnlyList<string> outputConcepts,
        bool isNegativeExample)
    {
        var inputs = NormalizeConcepts(inputConcepts);
        var outputs = NormalizeConcepts(outputConcepts);
        if (inputs.Count == 0 && outputs.Count == 0)
            return;

        var editScope = inputs.Concat(outputs).ToArray();
        var inputNodes = inputs.Select(c => GetOrCreate(c, editScope)).ToArray();
        var outputNodes = outputs.Select(c => GetOrCreate(c, editScope)).ToArray();
        if (inputNodes.Length == 0 || outputNodes.Length == 0)
            return;

        var inputCentroid = ComputeCentroid(inputNodes);
        var outputCentroid = ComputeCentroid(outputNodes);
        var rate = isNegativeExample ? 0.03 : 0.06;
        var outputSign = isNegativeExample ? -1.0 : 1.0;
        var inputSign = isNegativeExample ? -0.3 : 0.3;

        foreach (var node in outputNodes)
            ApplyCentroidNudge(node, inputCentroid, outputSign, rate);

        foreach (var node in inputNodes)
            ApplyCentroidNudge(node, outputCentroid, inputSign, rate * 0.5);
    }

    public IReadOnlyList<(string Left, string Right, long ObservationCount)> GetAllRelations()
    {
        var result = new List<(string Left, string Right, long ObservationCount)>();
        foreach (var rel in _relationIndex.Values)
        {
            result.Add((rel.Left, rel.Right, rel.ObservationCount));
        }
        return result;
    }

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        _nodes.Clear();
        _relationIndex.Clear();
        _lattice.Clear();

        foreach (var node in snapshot.Nodes)
        {
            var normalized = Normalize(node.Name);
            var positiveFace = Resize(node.PositiveFace, _faceDimension);
            // Numeric/operator faces are re-seeded so the homomorphic basis is exact on import;
            // all others are re-projected to unit norm (canonical) with frozen identity preserved.
            if (TryCreateSeededFace(normalized, out var seeded))
            {
                positiveFace = seeded;
            }
            else
            {
                // Re-normalize the free region; the char-face fingerprint stays as loaded.
                NormaliseFreeRegion(positiveFace, normalized);
            }
            // Hard G4 conservation: embed(¬x) = −embed(x).
            var negativeFace = new double[_faceDimension];
            for (var i = 0; i < _faceDimension; i++)
                negativeFace[i] = -positiveFace[i];
            _nodes[normalized] = new ConceptNode(
                name: normalized,
                positiveFace: positiveFace,
                negativeFace: negativeFace,
                observationCount: Math.Max(0, node.ObservationCount),
                useCount: Math.Max(0, node.UseCount),
                successCount: Math.Max(0, node.SuccessCount),
                failureCount: Math.Max(0, node.FailureCount),
                lastUsedStep: Math.Max(0, node.LastUsedStep));
            _lattice.RegisterNode(normalized);
        }

        foreach (var relation in snapshot.Relations)
        {
            var key = RelationKey(relation.Left, relation.Right);
            var conceptRelation = new RelationElementNode(
                left: Normalize(relation.Left),
                right: Normalize(relation.Right),
                thesisContradiction: Clamp01(relation.ThesisContradiction),
                lastObservedContradiction: Clamp01(relation.LastObservedContradiction),
                synthesisContradiction: Clamp01(relation.SynthesisContradiction),
                observationCount: Math.Max(0, relation.ObservationCount),
                useCount: Math.Max(0, relation.UseCount),
                successCount: Math.Max(0, relation.SuccessCount),
                failureCount: Math.Max(0, relation.FailureCount),
                lastUsedStep: Math.Max(0, relation.LastUsedStep));
            _relationIndex[key] = conceptRelation;
            IndexRelation(conceptRelation);
        }

        // Chunk store restored ADDITIVELY (counts merged) so a chunk-less maintenance snapshot never wipes
        // mined scaffolds; function-elements aren't snapshotted (re-registered deterministically by the registry).
        _chunks.ImportMerge(snapshot.Chunks);

        _utilityStep = Math.Max(
            _nodes.Values.Select(n => n.LastUsedStep).DefaultIfEmpty(0).Max(),
            _relationIndex.Values.Select(r => r.LastUsedStep).DefaultIfEmpty(0).Max());
    }

    public void ReinforceEvidence(IReadOnlyList<PlatonicEvidence> evidence, bool success)
    {
        if (evidence.Count == 0)
            return;

        foreach (var item in evidence
            .Where(e => !string.IsNullOrWhiteSpace(e.Concept))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(64))
        {
            var concept = Normalize(item.Concept);
            var related = string.IsNullOrWhiteSpace(item.RelatedConcept) ? null : Normalize(item.RelatedConcept);
            var step = ++_utilityStep;
            var magnitude = Math.Clamp(Math.Abs(item.Contribution), 0.05, 1.0);
            var hopScale = 1.0 / Math.Max(1, item.Hop);

            TouchNode(concept, success, step);
            if (related is not null)
                TouchNode(related, success, step);

            if (related is null || !_relationIndex.TryGetValue(RelationKey(concept, related), out var relation))
                continue;

            TouchRelation(relation, success, step);
            var degreeAttenuation = 1.0 / (1.0 + Math.Log(1.0 + GetRelationDegree(concept) + GetRelationDegree(related)));
            var delta = 0.025 * magnitude * hopScale * degreeAttenuation;
            relation.SynthesisContradiction = success
                ? Clamp01(relation.SynthesisContradiction - delta)
                : Clamp01(relation.SynthesisContradiction + delta);
            relation.LastObservedContradiction = relation.SynthesisContradiction;
            if (_nodes.TryGetValue(concept, out var left) && _nodes.TryGetValue(related, out var right))
                UpdateConceptGeometry(left, right, relation.SynthesisContradiction, magnitude * hopScale);
        }
    }

    public SpaceMaintenanceResult ApplyMaintenance(SpaceMaintenanceRequest request)
    {
        var protectedConcepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (request.ProtectedConcepts != null)
        {
            foreach (var concept in request.ProtectedConcepts)
            {
                if (!string.IsNullOrWhiteSpace(concept))
                    protectedConcepts.Add(Normalize(concept));
            }
        }

        var nodes = _nodes.Values
            .Select(n => new MutableNode(
                n.Name,
                n.PositiveFace.ToArray(),
                n.NegativeFace.ToArray(),
                n.ObservationCount,
                n.UseCount,
                n.SuccessCount,
                n.FailureCount,
                n.LastUsedStep))
            .ToDictionary(n => n.Name, StringComparer.OrdinalIgnoreCase);
        var relations = _relationIndex.Values
            .Select(r => new MutableRelation(
                r.Left,
                r.Right,
                r.ThesisContradiction,
                r.LastObservedContradiction,
                r.SynthesisContradiction,
                r.ObservationCount,
                r.UseCount,
                r.SuccessCount,
                r.FailureCount,
                r.LastUsedStep))
            .ToDictionary(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase);

        var minRelationsToKeep = Math.Max(0, request.MinRelationsToKeep);
        var minNodesToKeep = Math.Max(0, request.MinNodesToKeep);
        var relationPruneBudget = Math.Max(0, request.MaxRelationPrunes);
        var rebalanceBudget = Math.Max(0, request.MaxRebalancePrunes);
        var nodePruneBudget = Math.Max(0, request.MaxNodePrunes);
        var nodeMergeBudget = Math.Max(0, request.MaxNodeMerges);
        var minRelObs = Math.Max(0, request.MinRelationObservationToKeep);
        var minNodeObs = Math.Max(0, request.MinNodeObservationToKeep);
        var maxSynthesis = Clamp01(request.MaxSynthesisContradictionToKeep);

        var relationsPruned = 0;
        var nodesPruned = 0;
        var nodesMerged = 0;

        var staleRelations = relations.Values
            .Where(r =>
            {
                if (protectedConcepts.Contains(r.Left) || protectedConcepts.Contains(r.Right))
                    return false;
                if (RelationUtilityScore(r) > 0.45)
                    return false;
                if (r.ObservationCount > minRelObs && RelationUtilityScore(r) >= 0.15)
                    return false;
                if (r.SynthesisContradiction < maxSynthesis && RelationUtilityScore(r) >= 0.05)
                    return false;
                var leftObs = nodes.TryGetValue(r.Left, out var leftNode) ? leftNode.ObservationCount : 0;
                var rightObs = nodes.TryGetValue(r.Right, out var rightNode) ? rightNode.ObservationCount : 0;
                return leftObs <= (minNodeObs + 1) && rightObs <= (minNodeObs + 1);
            })
            .OrderBy(r => RelationUtilityScore(r))
            .ThenBy(r => r.ObservationCount)
            .ThenByDescending(r => r.SynthesisContradiction)
            .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var relation in staleRelations)
        {
            if (relationPruneBudget <= 0 || relations.Count <= minRelationsToKeep)
                break;
            if (relations.Remove(RelationKey(relation.Left, relation.Right)))
            {
                relationPruneBudget--;
                relationsPruned++;
            }
        }

        if (nodeMergeBudget > 0 && nodes.Count > minNodesToKeep)
        {
            var mergeCandidates = nodes.Values
                .Where(n => !protectedConcepts.Contains(n.Name) && NodeUtilityScore(n) < 0.55)
                .OrderBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            for (var i = 0; i < mergeCandidates.Length && nodeMergeBudget > 0; i++)
            {
                var sourceName = mergeCandidates[i].Name;
                if (!nodes.ContainsKey(sourceName))
                    continue;

                for (var j = i + 1; j < mergeCandidates.Length && nodeMergeBudget > 0; j++)
                {
                    if (!nodes.TryGetValue(sourceName, out var source))
                        break;
                    if (!nodes.TryGetValue(mergeCandidates[j].Name, out var target))
                        continue;
                    if (source.Name.Equals(target.Name, StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (protectedConcepts.Contains(source.Name) || protectedConcepts.Contains(target.Name))
                        continue;

                    var sourceDegree = CountRelationsForNode(relations.Values, source.Name);
                    var targetDegree = CountRelationsForNode(relations.Values, target.Name);
                    if (sourceDegree > 1 || targetDegree > 1)
                        continue;

                    var distance = FaceDistance(source.PositiveFace, target.PositiveFace);
                    if (distance > request.MergeDistanceThreshold)
                        continue;

                    var sourceValue = source.ObservationCount + NodeUtilityScore(source);
                    var targetValue = target.ObservationCount + NodeUtilityScore(target);
                    var canonical = sourceValue >= targetValue ? source : target;
                    var duplicate = ReferenceEquals(canonical, source) ? target : source;
                    MergeNodeInto(canonical, duplicate, nodes, relations);
                    nodesMerged++;
                    nodeMergeBudget--;
                }
            }
        }

        var maxRelationsPerNode = Math.Max(0, request.MaxRelationsPerNode);
        var targetRelationCount = Math.Max(0, request.TargetRelationCount);
        if (maxRelationsPerNode > 0 && rebalanceBudget > 0 && relations.Count > targetRelationCount)
        {
            var nodeOrder = nodes.Keys
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            foreach (var nodeName in nodeOrder)
            {
                if (rebalanceBudget <= 0 || relations.Count <= Math.Max(minRelationsToKeep, targetRelationCount))
                    break;
                if (protectedConcepts.Contains(nodeName))
                    continue;

                var incident = relations.Values
                    .Where(r => r.Left.Equals(nodeName, StringComparison.OrdinalIgnoreCase) ||
                                r.Right.Equals(nodeName, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(r => RelationUtilityScore(r))
                    .ThenBy(r => r.ObservationCount)
                    .ThenByDescending(r => r.SynthesisContradiction)
                    .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                var over = incident.Length - maxRelationsPerNode;
                if (over <= 0)
                    continue;

                foreach (var relation in incident)
                {
                    if (over <= 0 || rebalanceBudget <= 0 || relations.Count <= Math.Max(minRelationsToKeep, targetRelationCount))
                        break;
                    if (protectedConcepts.Contains(relation.Left) || protectedConcepts.Contains(relation.Right))
                        continue;
                    if (RelationUtilityScore(relation) > 0.45)
                        continue;
                    if (relation.ObservationCount > (minRelObs + 1) && relation.SynthesisContradiction < (maxSynthesis + 0.1) && RelationUtilityScore(relation) >= 0.15)
                        continue;

                    if (relations.Remove(RelationKey(relation.Left, relation.Right)))
                    {
                        over--;
                        rebalanceBudget--;
                        relationsPruned++;
                    }
                }
            }
        }

        if (nodePruneBudget > 0 && nodes.Count > minNodesToKeep)
        {
            var relationEndpoints = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var relation in relations.Values)
            {
                relationEndpoints.Add(relation.Left);
                relationEndpoints.Add(relation.Right);
            }

            var staleNodes = nodes.Values
                .Where(n =>
                    !protectedConcepts.Contains(n.Name) &&
                    n.ObservationCount <= minNodeObs &&
                    NodeUtilityScore(n) < 0.35 &&
                    !relationEndpoints.Contains(n.Name))
                .OrderBy(n => NodeUtilityScore(n))
                .ThenBy(n => n.ObservationCount)
                .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            foreach (var node in staleNodes)
            {
                if (nodePruneBudget <= 0 || nodes.Count <= minNodesToKeep)
                    break;
                if (nodes.Remove(node.Name))
                {
                    nodePruneBudget--;
                    nodesPruned++;
                }
            }
        }

        if (relationsPruned == 0 && nodesPruned == 0 && nodesMerged == 0)
            return new SpaceMaintenanceResult(0, 0, 0);

        ImportSnapshot(new PlatonicMemorySnapshot(
            FaceDimension: _faceDimension,
            Nodes: nodes.Values
                .OrderBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .Select(n => new PlatonicNodeSnapshot(
                    n.Name,
                    n.PositiveFace,
                    n.NegativeFace,
                    Math.Max(0, n.ObservationCount),
                    Math.Max(0, n.UseCount),
                    Math.Max(0, n.SuccessCount),
                    Math.Max(0, n.FailureCount),
                    Math.Max(0, n.LastUsedStep)))
                .ToArray(),
            Relations: relations.Values
                .OrderBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
                .Select(r => new PlatonicRelationSnapshot(
                    r.Left,
                    r.Right,
                    Clamp01(r.ThesisContradiction),
                    Clamp01(r.LastObservedContradiction),
                    Clamp01(r.SynthesisContradiction),
                    Math.Max(0, r.ObservationCount),
                    Math.Max(0, r.UseCount),
                    Math.Max(0, r.SuccessCount),
                    Math.Max(0, r.FailureCount),
                    Math.Max(0, r.LastUsedStep)))
                .ToArray()));

        return new SpaceMaintenanceResult(
            RelationsPruned: relationsPruned,
            NodesPruned: nodesPruned,
            NodesMerged: nodesMerged);
    }

    // Adjacency is owned by the lattice (Layer 1 topology). The relation payload
    // (contradiction, observation counts) stays in _relationIndex, keyed by RelationKey.
    private void IndexRelation(RelationElementNode relation)
        => _lattice.AddEdge(relation.Left, relation.Right);

    /// <summary>
    /// Canonical embedding message-passing update (conforms to GraphAligner / EmbeddingSpace.UpdateEmbedding
    /// in the source of truth). Replaces the non-canonical contradiction→spring-distance mutator.
    ///
    /// For each concept we (1) pull its positive face toward its neighbour (excluding the complement),
    /// (2) repel from the complement (the negative face) to maintain dual-space separation,
    /// (3) restore frozen identity dims (arithmetic poly/log seeds, operator faces) so the homomorphism
    /// never drifts, (4) re-normalize to UNIT norm (||e|| = 1), and (5) enforce the hard G4 conservation
    /// law embed(¬x) = −embed(x).
    ///
    /// The contradiction signal selects pull (low contradiction = agree → pull together) vs. push
    /// (high contradiction = disagree → push apart), preserving the external behavior callers rely on
    /// (ObserveContradiction / ReinforceEvidence), while routing all value updates through the canonical rule.
    /// </summary>
    private void UpdateConceptGeometry(ConceptNode a, ConceptNode b, double targetContradiction, double rateScale)
    {
        var baseRate = GeometryLearningRate * Math.Clamp(rateScale, 0.0, 1.0);
        // Canonical alpha = max(0.05, 2/(n+1)); n here is the relational degree (neighbour count proxy).
        var nA = GetRelationDegree(a.Name);
        var nB = GetRelationDegree(b.Name);
        var alphaA = baseRate * Math.Max(0.05, 2.0 / (nA + 1)) * NodePlasticity(a);
        var alphaB = baseRate * Math.Max(0.05, 2.0 / (nB + 1)) * NodePlasticity(b);

        // Low contradiction → concepts agree → pull together (+1).
        // High contradiction → concepts disagree → push apart (−1).
        var affinity = 1.0 - (2.0 * Clamp01(targetContradiction)); // [+1 .. -1]
        // Floor the PULL so an observed relation clusters even under a weak edit-head magnitude (balance the
        // push). Only when attracting — a repel (affinity<0) keeps its full strength.
        if (affinity > 0.0)
            affinity = Math.Max(affinity, MinAttractAffinity);

        MessagePassUpdate(a, b.PositiveFace, alphaA, affinity);
        MessagePassUpdate(b, a.PositiveFace, alphaB, affinity);
    }

    /// <summary>
    /// One canonical message-passing step on a single node's positive face:
    /// pull toward (or push from) a neighbour face, repel from the node's own complement (negative face),
    /// restore frozen identity dims, unit-normalize, then enforce the hard complement (NegativeFace = −PositiveFace).
    /// </summary>
    private void MessagePassUpdate(ConceptNode node, double[] neighbourFace, double alpha, double affinity)
    {
        // Numeric concepts are ground truth — never mutated (matches the source vocabulary skip);
        // their canonical poly/log faces are recomputed on demand.
        if (IsFrozenConcept(node.Name))
            return;

        var dim = _faceDimension;
        var original = (double[])node.PositiveFace.ToArray();
        var updated = node.PositiveFace;

        // (1) Neighbour pull / push — canonical: updated += lr * (neighbour - updated), signed by affinity.
        for (var i = 0; i < dim; i++)
            updated[i] += alpha * affinity * (neighbourFace[i] - updated[i]);

        // (2) Complement repulsion — push AWAY from the negative face to maintain dual-space separation
        // (canonical EmbeddingSpace.UpdateEmbedding repels from the complement when too close).
        var dist = EuclideanDistance(updated, node.NegativeFace);
        if (dist < 1.35)
        {
            var repulsion = alpha * 0.5;
            for (var i = 0; i < dim; i++)
                updated[i] += repulsion * (updated[i] - node.NegativeFace[i]);
        }

        // (3) Restore frozen identity dims (arithmetic/lexical identity never drifts), then
        // (4) unit-normalize ONLY the free region so learned wiggle magnitude survives.
        RestoreFrozenIdentity(updated, original, node.Name);
        NormaliseFreeRegion(updated, node.Name);

        // (5) Hard G4 conservation: embed(¬x) = −embed(x).
        for (var i = 0; i < dim; i++)
            node.NegativeFace[i] = -node.PositiveFace[i];

        _lattice.MarkEmbeddingsDirty();
    }

    /// <summary>
    /// Contrastive repulsion pass — the genesis <c>NudgeGraphAlignment</c> repulsion half, ported.
    /// For each mutable concept, push its face AWAY from a few sampled UNRELATED concepts (those with
    /// no relation edge), so unrelated/fringe things stay far apart and proximity is earned only by
    /// confirmed attraction. Frozen identity dims are restored and the free region re-normalised, so
    /// the homomorphism and lexical fingerprints never drift. Deterministic (seeded per node+pass).
    /// </summary>
    private void ApplyContrastiveRepulsionPass()
    {
        var n = _nodes.Count;
        if (n < 3)
            return;

        var snapshot = _nodes.Values.ToArray();
        // Repel ONLY from MUTABLE targets. Frozen numbers (often the MAJORITY of nodes at scale — e.g. ~72% in
        // the live gym) never move AND don't need separating from text concepts; sampling them wasted most of
        // the repulsion budget, so mutable-vs-mutable separation was starved and the space collapsed. Sampling
        // the mutable pool restores effective repulsion regardless of how many numbers exist.
        var targets = snapshot.Where(x => !IsFrozenConcept(x.Name) && x.PositiveFace is { Length: > 0 }).ToArray();
        if (targets.Length < 2)
            return;
        var dim = _faceDimension;
        _repulsionPassCount++;

        foreach (var node in snapshot)
        {
            if (IsFrozenConcept(node.Name)) // ground-truth numerics never move
                continue;

            var original = (double[])node.PositiveFace.Clone();
            var updated = node.PositiveFace;
            var rng = new Random(unchecked(StableHash(node.Name) + _repulsionPassCount));

            var samples = Math.Min(RepulsionSamples, targets.Length - 1);
            for (var s = 0; s < samples; s++)
            {
                var other = targets[rng.Next(targets.Length)];
                if (ReferenceEquals(other, node))
                    continue;
                // Repulsion is ONLY for unrelated pairs — a confirmed relation is exempt (it is what
                // earns proximity). This is the contrastive half: attract related, repel everything else.
                if (_relationIndex.ContainsKey(RelationKey(node.Name, other.Name)))
                    continue;

                var of = other.PositiveFace;
                var dist2 = 0.0;
                for (var d = 0; d < dim && d < of.Length; d++)
                {
                    var dx = updated[d] - of[d];
                    dist2 += dx * dx;
                }
                if (dist2 < 1e-12)
                    continue;

                // Repulsive force inversely proportional to distance (genesis: alpha/ max(dist,0.01)).
                var force = (RepulsionRate * RepulsionRatio) / Math.Max(Math.Sqrt(dist2), 0.01);
                for (var d = 0; d < dim && d < of.Length; d++)
                    updated[d] += force * (updated[d] - of[d]);
            }

            // Identity never drifts; free region re-normalised; hard G4 complement re-enforced.
            RestoreFrozenIdentity(updated, original, node.Name);
            NormaliseFreeRegion(updated, node.Name);
            for (var d = 0; d < dim; d++)
                node.NegativeFace[d] = -updated[d];
        }

        _lattice.MarkEmbeddingsDirty();
    }

    private static int StableHash(string value)
    {
        var hash = 2166136261u;
        foreach (var c in value)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return unchecked((int)hash);
    }

    private ConceptNode GetOrCreate(string concept, IReadOnlyCollection<string>? protectedConcepts = null)
    {
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
            return node;

        EnsureNodeCapacity(protectedConcepts is null
            ? new[] { key }
            : protectedConcepts.Append(key).ToArray());
        var positiveFace = TryCreateSeededFace(key, out var seeded)
            ? seeded
            : CreateFace(key);
        // SPAWN SPREAD: non-numeric elements are born on the UNIT SPHERE of their FREE region (a random
        // direction from the seed noise), so in ~800 free dims any two fresh elements are near-orthogonal and
        // kNN can separate them from birth — we do NOT want everything clumped at the word-face origin waiting
        // for attraction to drag it out. Attraction then pulls RELATED elements in (it just needs enough
        // observations to converge from the spread start); repulsion keeps unrelated apart. Identity dims (the
        // numeric homomorphism, the frozen char spelling) are left exact.
        if (!IsFrozenConcept(key))
            NormaliseFreeRegion(positiveFace, key);
        node = new ConceptNode(
            name: key,
            positiveFace: positiveFace,
            negativeFace: positiveFace.Select(x => -x).ToArray(),
            observationCount: 0,
            useCount: 0,
            successCount: 0,
            failureCount: 0,
            lastUsedStep: 0);
        _nodes[key] = node;
        _lattice.RegisterNode(key);
        return node;
    }

    private bool TryCreateSeededFace(string concept, out double[] face)
    {
        if (TryParseNumber(concept, out var numeric))
        {
            face = CreateNumericFace(numeric);
            return true;
        }

        // Operator words are no longer special-cased: the operation is determined by parsing the
        // operator symbol (and by the learned transforms), so "plus"/"times"/etc. are ordinary
        // learnable text concepts composed on the canonical face geometry like any other word.
        face = Array.Empty<double>();
        return false;
    }

    private double[] CreateNumericFace(double value)
    {
        var face = new double[_faceDimension];
        var numericDims = Math.Min(_faceDimension / 2, 21);
        var logStart = numericDims;
        var logDims = Math.Min(numericDims, _faceDimension - logStart);

        for (var i = 0; i < numericDims; i++)
            face[i] = value * Math.Pow(10, -(i + 1));

        if (Math.Abs(value) > 1e-12)
        {
            var logValue = Math.Log(Math.Abs(value));
            for (var i = 0; i < logDims; i++)
                face[logStart + i] = logValue * Math.Pow(10, -(i + 1));
        }

        return face;
    }

    // Text concepts are composed on the canonical face geometry (clean char slots / word face +
    // seeded free dims) so a stored concept face and its input embedding are directly comparable.
    // Numeric concepts never reach here — they are seeded via TryCreateSeededFace.
    private double[] CreateFace(string concept)
        => Core.PlatonicFaceComposer.Compose(concept, _faceDimension);

    private double[] ComputeCentroid(IReadOnlyList<ConceptNode> nodes)
    {
        var centroid = new double[_faceDimension];
        if (nodes.Count == 0)
            return centroid;

        foreach (var node in nodes)
        {
            for (var i = 0; i < _faceDimension; i++)
                centroid[i] += node.PositiveFace[i];
        }

        var scale = 1.0 / nodes.Count;
        for (var i = 0; i < _faceDimension; i++)
            centroid[i] *= scale;
        return centroid;
    }

    private void ApplyCentroidNudge(ConceptNode node, IReadOnlyList<double> centroid, double sign, double rate)
    {
        // Numeric concepts are ground truth — never nudged.
        if (IsFrozenConcept(node.Name))
            return;

        var effectiveRate = rate * NodePlasticity(node);
        var original = (double[])node.PositiveFace.ToArray();
        for (var i = 0; i < _faceDimension; i++)
        {
            var delta = centroid[i] - node.PositiveFace[i];
            node.PositiveFace[i] += sign * effectiveRate * delta;
        }

        // Restore identity dims, then unit-normalize ONLY the free region (preserve wiggle magnitude).
        RestoreFrozenIdentity(node.PositiveFace, original, node.Name);
        NormaliseFreeRegion(node.PositiveFace, node.Name);
        for (var i = 0; i < _faceDimension; i++)
            node.NegativeFace[i] = -node.PositiveFace[i];

        _lattice.MarkEmbeddingsDirty();
    }

    private double NodePlasticity(ConceptNode node)
    {
        var degree = GetRelationDegree(node.Name);
        var usage = Math.Log(1.0 + Math.Max(0, degree)) + Math.Log(1.0 + Math.Max(0, node.ObservationCount));
        return 1.0 / (1.0 + usage);
    }

    public int GetRelationDegree(string concept)
        => _lattice.Degree(Normalize(concept));

    private void TouchNode(string concept, bool success, long step)
    {
        if (!_nodes.TryGetValue(Normalize(concept), out var node))
            return;
        node.UseCount++;
        if (success)
            node.SuccessCount++;
        else
            node.FailureCount++;
        node.LastUsedStep = step;
    }

    private static void TouchRelation(RelationElementNode relation, bool success, long step)
    {
        relation.UseCount++;
        if (success)
            relation.SuccessCount++;
        else
            relation.FailureCount++;
        relation.LastUsedStep = step;
    }

    private double NodeUtilityScore(ConceptNode node)
        => UtilityScore(node.UseCount, node.SuccessCount, node.FailureCount, node.LastUsedStep);

    private double NodeUtilityScore(MutableNode node)
        => UtilityScore(node.UseCount, node.SuccessCount, node.FailureCount, node.LastUsedStep);

    private double RelationUtilityScore(RelationElementNode relation)
        => UtilityScore(relation.UseCount, relation.SuccessCount, relation.FailureCount, relation.LastUsedStep);

    private double RelationUtilityScore(MutableRelation relation)
        => UtilityScore(relation.UseCount, relation.SuccessCount, relation.FailureCount, relation.LastUsedStep);

    private double UtilityScore(int useCount, int successCount, int failureCount, long lastUsedStep)
    {
        if (useCount <= 0)
            return 0.0;
        var successRatio = (successCount + 0.5) / Math.Max(1.0, successCount + failureCount + 1.0);
        var useSignal = Math.Min(1.0, Math.Log(1.0 + useCount) / Math.Log(17.0));
        var recency = lastUsedStep <= 0 ? 0.0 : 1.0 / (1.0 + Math.Max(0, _utilityStep - lastUsedStep) / 512.0);
        var failurePenalty = failureCount > successCount ? Math.Min(0.4, (failureCount - successCount) / Math.Max(1.0, useCount) * 0.5) : 0.0;
        return Math.Clamp((0.55 * successRatio) + (0.25 * useSignal) + (0.20 * recency) - failurePenalty, 0.0, 1.0);
    }

    // Only numbers are frozen ground truth: their poly/log faces are the homomorphic basis and
    // must never drift. Everything else (including operator words) is learnable.
    private static bool IsFrozenConcept(string concept) => TryParseNumber(concept, out _);

    /// <summary>
    /// The frozen identity region of a concept's face — the dims that carry its exact, canonical
    /// signature and must never drift. Everything OUTSIDE this range is free to "wiggle" (learnable).
    /// <list type="bullet">
    ///   <item>Numbers: the arithmetic face [0 .. 2*NumericDimensions) (poly + log) —
    ///   the homomorphic basis (value*10^-i, ln|value|*10^-i).</item>
    ///   <item>Text concepts: the character-slot face [CharFaceStart .. WordFaceStart) — the
    ///   lexical fingerprint. The word face and all other dims stay learnable. (The spelling-independent
    ///   per-concept IDENTITY lives on a SEPARATE word ELEMENT, not in the concept's face — see
    ///   <c>RegisterWordElement</c> / WordElementRegistry — so it is stable by being in its own index.)</item>
    /// </list>
    /// </summary>
    private (int Start, int End) IdentityRange(string concept)
    {
        if (IsFrozenConcept(concept))
            return (0, Math.Min(2 * NumericDimensions, _faceDimension));

        var charStart = Core.FaceLayout.CharFaceStart(_faceDimension);
        var charEnd = Core.FaceLayout.WordFaceStart(_faceDimension);
        return charEnd > charStart && charEnd <= _faceDimension ? (charStart, charEnd) : (0, 0);
    }

    /// <summary>
    /// Restore the frozen identity dims after an embedding update so the concept's exact signature
    /// (arithmetic homomorphism for numbers, lexical fingerprint for text) never drifts. Conforms to
    /// EmbeddingSpace.RestoreFrozenIdentity, generalised per-region by concept type. Guarantees
    /// embed(a+b)=embed(a)+embed(b) and ln(a*b)=ln a + ln b survive message passing.
    /// </summary>
    private void RestoreFrozenIdentity(double[] updated, double[] original, string concept)
    {
        var (start, end) = IdentityRange(concept);
        for (var i = start; i < end && i < updated.Length; i++)
            updated[i] = original[i];
    }

    /// <summary>
    /// Unit-normalize ONLY the free (non-identity) region of a face, leaving the frozen identity dims
    /// exactly as-is. Fixes the prior whole-vector normalization, which renormalized every dim each
    /// tick and so erased the learned MAGNITUDE of the free "wiggle" dims (only direction survived).
    /// Per-region normalization keeps the free dims at a stable, non-vanishing scale while the identity
    /// dims stay exact.
    /// </summary>
    private void NormaliseFreeRegion(double[] face, string concept)
    {
        var (idStart, idEnd) = IdentityRange(concept);
        var sum = 0.0;
        for (var i = 0; i < face.Length; i++)
        {
            if (i >= idStart && i < idEnd) continue;
            sum += face[i] * face[i];
        }

        var norm = Math.Sqrt(sum);
        if (norm < 1e-15)
            return;

        var inv = 1.0 / norm;
        for (var i = 0; i < face.Length; i++)
        {
            if (i >= idStart && i < idEnd) continue;
            face[i] *= inv;
        }
    }

    /// <summary>
    /// Total charge of the space (G4 diagnostic): sum of every coordinate across all faces.
    /// With the hard complement embed(¬x) = −embed(x) enforced on every update, positive and
    /// negative faces cancel pairwise so this should stay ≈ 0.
    /// </summary>
    public double TotalCharge()
    {
        var sum = 0.0;
        foreach (var node in _nodes.Values)
        {
            for (var i = 0; i < node.PositiveFace.Length; i++)
                sum += node.PositiveFace[i] + node.NegativeFace[i];
        }
        return sum;
    }

    private void EnsureNodeCapacity(IReadOnlyCollection<string> protectedConcepts)
    {
        if (_nodes.Count < _maxPlatonicNodes)
            return;

        var protectedSet = protectedConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(Normalize)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var candidate = _nodes.Values
            .Where(n => !protectedSet.Contains(n.Name))
            .OrderBy(n => NodeUtilityScore(n))
            .ThenBy(n => n.ObservationCount)
            .ThenBy(n => GetRelationDegree(n.Name))
            .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault()
            ?? _nodes.Values
                .OrderBy(n => NodeUtilityScore(n))
                .ThenBy(n => n.ObservationCount)
                .ThenBy(n => GetRelationDegree(n.Name))
                .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();
        if (candidate is not null)
            RemoveNode(candidate.Name);
    }

    private void EnsureRelationCapacity()
    {
        if (_relationIndex.Count < _maxPlatonicRelations)
            return;

        var candidate = _relationIndex.Values
            .OrderBy(r => RelationUtilityScore(r))
            .ThenBy(r => r.ObservationCount)
            .ThenByDescending(r => r.SynthesisContradiction)
            .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
        if (candidate is not null)
            RemoveRelation(candidate);
    }

    private void RemoveNode(string concept)
    {
        var key = Normalize(concept);
        var incident = _relationIndex.Values
            .Where(r => r.Left.Equals(key, StringComparison.OrdinalIgnoreCase) ||
                        r.Right.Equals(key, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        foreach (var relation in incident)
            RemoveRelation(relation);

        _nodes.Remove(key);
        _lattice.UnregisterNode(key);
    }

    private void RemoveRelation(RelationElementNode relation)
    {
        _relationIndex.Remove(RelationKey(relation.Left, relation.Right));
        _lattice.RemoveEdge(relation.Left, relation.Right);
    }

    private static IReadOnlyList<string> NormalizeConcepts(IReadOnlyList<string> concepts)
        => concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(Normalize)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(32)
            .ToArray();

    private static string RelationKey(string left, string right)
    {
        var a = Normalize(left);
        var b = Normalize(right);
        return string.Compare(a, b, StringComparison.OrdinalIgnoreCase) <= 0
            ? $"{a}|{b}"
            : $"{b}|{a}";
    }

    private static string Normalize(string value)
        => value.Trim().ToLowerInvariant();

    private static PlatonicNeighborhoodType ClassifyConceptType(string token)
        => TryParseNumber(token, out _)
            ? PlatonicNeighborhoodType.Numeric
            : PlatonicNeighborhoodType.Semantic;

    // Strict numeric definition: leading sign + digits + optional decimal point only. NumberStyles.Any
    // additionally accepts a TRAILING sign (and thousands/exponent/currency/parens), which let malformed
    // concept-planner tokens like "0+" or "5-" parse as a value (0, 5) and become FROZEN "value-0/5"
    // garbage concepts that contaminate the numeric relation graph. A number is a number — nothing glued.
    private const System.Globalization.NumberStyles NumberParseStyle =
        System.Globalization.NumberStyles.AllowLeadingSign
        | System.Globalization.NumberStyles.AllowDecimalPoint
        | System.Globalization.NumberStyles.AllowLeadingWhite
        | System.Globalization.NumberStyles.AllowTrailingWhite;

    private static bool TryParseNumber(string token, out double value)
        => double.TryParse(token, NumberParseStyle, System.Globalization.CultureInfo.InvariantCulture, out value);

    private static double[] Resize(double[] source, int size)
    {
        if (source.Length == size)
            return source.ToArray();

        var target = new double[size];
        Array.Copy(source, 0, target, 0, Math.Min(source.Length, size));
        return target;
    }

    private static double Clamp01(double value)
        => Math.Max(0.0, Math.Min(1.0, value));

    private static double EuclideanDistance(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var length = Math.Min(a.Count, b.Count);
        var sum = 0.0;
        for (var i = 0; i < length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    // FACE-AWARE distance (genesis hybrid): relatedness is measured on the SEMANTIC face [WordFaceStart..dim)
    // so a concept's value (numeric face) and spelling (char face) cannot contaminate it. For a NUMERIC
    // query (arithmetic-face signal present) we ALSO measure the arithmetic face and take the MIN, so
    // numbers still surface value-near neighbours while text comparisons stay shielded. This is the real
    // fix the per-concept zero-freezing was bandaging.
    private double FaceAwareDistance(IReadOnlyList<double> query, IReadOnlyList<double> candidate)
    {
        var dim = _faceDimension;
        var arithEnd = Math.Min(2 * NumericDimensions, dim);
        var semStart = Core.FaceLayout.WordFaceStart(dim);
        var semantic = semStart > 0 && semStart < dim
            ? RangeDistance(query, candidate, semStart, dim)
            : RangeDistance(query, candidate, 0, dim);

        var arithNorm = 0.0;
        for (var i = 0; i < arithEnd && i < query.Count; i++)
            arithNorm += query[i] * query[i];
        if (arithNorm <= 0.01) // text query → semantic face only
            return semantic;

        return Math.Min(semantic, RangeDistance(query, candidate, 0, arithEnd)); // numeric → blend value face
    }

    private static double RangeDistance(IReadOnlyList<double> a, IReadOnlyList<double> b, int start, int end)
    {
        var n = Math.Min(a.Count, b.Count);
        var e = Math.Min(end, n);
        var s = Math.Min(Math.Max(0, start), e);
        var sum = 0.0;
        for (var i = s; i < e; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static int CountRelationsForNode(IEnumerable<MutableRelation> relations, string nodeName)
        => relations.Count(r =>
            r.Left.Equals(nodeName, StringComparison.OrdinalIgnoreCase) ||
            r.Right.Equals(nodeName, StringComparison.OrdinalIgnoreCase));

    private static double FaceDistance(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var length = Math.Min(left.Count, right.Count);
        var sum = 0.0;
        for (var i = 0; i < length; i++)
        {
            var d = left[i] - right[i];
            sum += d * d;
        }

        return Math.Sqrt(sum / Math.Max(1, length));
    }

    private static void MergeNodeInto(
        MutableNode canonical,
        MutableNode duplicate,
        IDictionary<string, MutableNode> nodes,
        IDictionary<string, MutableRelation> relations)
    {
        var totalObs = Math.Max(1, canonical.ObservationCount + duplicate.ObservationCount);
        var canonicalWeight = canonical.ObservationCount / (double)totalObs;
        var duplicateWeight = duplicate.ObservationCount / (double)totalObs;
        for (var i = 0; i < canonical.PositiveFace.Length; i++)
        {
            canonical.PositiveFace[i] = (canonical.PositiveFace[i] * canonicalWeight) + (duplicate.PositiveFace[i] * duplicateWeight);
            canonical.NegativeFace[i] = (canonical.NegativeFace[i] * canonicalWeight) + (duplicate.NegativeFace[i] * duplicateWeight);
        }
        canonical.ObservationCount += duplicate.ObservationCount;
        canonical.UseCount += duplicate.UseCount;
        canonical.SuccessCount += duplicate.SuccessCount;
        canonical.FailureCount += duplicate.FailureCount;
        canonical.LastUsedStep = Math.Max(canonical.LastUsedStep, duplicate.LastUsedStep);

        var incidentRelations = relations.Values
            .Where(r => r.Left.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase) ||
                        r.Right.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase))
            .OrderBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var relation in incidentRelations)
        {
            relations.Remove(RelationKey(relation.Left, relation.Right));
            var remappedLeft = relation.Left.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase)
                ? canonical.Name
                : relation.Left;
            var remappedRight = relation.Right.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase)
                ? canonical.Name
                : relation.Right;
            if (remappedLeft.Equals(remappedRight, StringComparison.OrdinalIgnoreCase))
                continue;

            var remappedKey = RelationKey(remappedLeft, remappedRight);
            if (relations.TryGetValue(remappedKey, out var existing))
            {
                var combinedObs = Math.Max(1, existing.ObservationCount + relation.ObservationCount);
                existing.ThesisContradiction = Clamp01(
                    ((existing.ThesisContradiction * existing.ObservationCount) + (relation.ThesisContradiction * relation.ObservationCount)) / combinedObs);
                existing.LastObservedContradiction = Clamp01((existing.LastObservedContradiction + relation.LastObservedContradiction) * 0.5);
                existing.SynthesisContradiction = Clamp01(
                    ((existing.SynthesisContradiction * existing.ObservationCount) + (relation.SynthesisContradiction * relation.ObservationCount)) / combinedObs);
                existing.ObservationCount = combinedObs;
                existing.UseCount += relation.UseCount;
                existing.SuccessCount += relation.SuccessCount;
                existing.FailureCount += relation.FailureCount;
                existing.LastUsedStep = Math.Max(existing.LastUsedStep, relation.LastUsedStep);
            }
            else
            {
                relation.Left = remappedLeft;
                relation.Right = remappedRight;
                relations[remappedKey] = relation;
            }
        }

        nodes.Remove(duplicate.Name);
    }

    private sealed class ConceptNode
    {
        public ConceptNode(
            string name,
            double[] positiveFace,
            double[] negativeFace,
            int observationCount,
            int useCount,
            int successCount,
            int failureCount,
            long lastUsedStep)
        {
            Name = name;
            PositiveFace = positiveFace;
            NegativeFace = negativeFace;
            ObservationCount = observationCount;
            UseCount = useCount;
            SuccessCount = successCount;
            FailureCount = failureCount;
            LastUsedStep = lastUsedStep;
        }

        public string Name { get; }
        public double[] PositiveFace { get; }
        public double[] NegativeFace { get; }
        public int ObservationCount { get; set; }
        public int UseCount { get; set; }
        public int SuccessCount { get; set; }
        public int FailureCount { get; set; }
        public long LastUsedStep { get; set; }
    }

    // A relation as a first-class POSITIONED element (genesis Kind=Relation): the mutable element behind the
    // keyed _relationIndex. Its POSITION is the centroid of its endpoints' faces (derived, always consistent —
    // projected by GetRelationElements); its STRENGTH is 1 − SynthesisContradiction. The remaining fields are
    // the relation's learned dynamics + lifecycle. Because it is a positioned element, it can itself become an
    // endpoint of another relation (higher-order) — the substrate is uniform.
    private sealed class RelationElementNode
    {
        public RelationElementNode(
            string left,
            string right,
            double thesisContradiction,
            double lastObservedContradiction,
            double synthesisContradiction,
            int observationCount,
            int useCount,
            int successCount,
            int failureCount,
            long lastUsedStep)
        {
            Left = left;
            Right = right;
            ThesisContradiction = thesisContradiction;
            LastObservedContradiction = lastObservedContradiction;
            SynthesisContradiction = synthesisContradiction;
            ObservationCount = observationCount;
            UseCount = useCount;
            SuccessCount = successCount;
            FailureCount = failureCount;
            LastUsedStep = lastUsedStep;
        }

        public string Left { get; }
        public string Right { get; }
        public double ThesisContradiction { get; }
        public double LastObservedContradiction { get; set; }
        public double SynthesisContradiction { get; set; }
        public int ObservationCount { get; set; }
        public int UseCount { get; set; }
        public int SuccessCount { get; set; }
        public int FailureCount { get; set; }
        public long LastUsedStep { get; set; }

        // Strength of the relation as a positioned element: confirmed attraction = 1 − synthesis contradiction.
        public double Strength => Math.Max(0.0, Math.Min(1.0, 1.0 - SynthesisContradiction));
    }

    private sealed class MutableNode
    {
        public MutableNode(
            string name,
            double[] positiveFace,
            double[] negativeFace,
            int observationCount,
            int useCount,
            int successCount,
            int failureCount,
            long lastUsedStep)
        {
            Name = name;
            PositiveFace = positiveFace;
            NegativeFace = negativeFace;
            ObservationCount = observationCount;
            UseCount = useCount;
            SuccessCount = successCount;
            FailureCount = failureCount;
            LastUsedStep = lastUsedStep;
        }

        public string Name { get; }
        public double[] PositiveFace { get; }
        public double[] NegativeFace { get; }
        public int ObservationCount { get; set; }
        public int UseCount { get; set; }
        public int SuccessCount { get; set; }
        public int FailureCount { get; set; }
        public long LastUsedStep { get; set; }
    }

    private sealed class MutableRelation
    {
        public MutableRelation(
            string left,
            string right,
            double thesisContradiction,
            double lastObservedContradiction,
            double synthesisContradiction,
            int observationCount,
            int useCount,
            int successCount,
            int failureCount,
            long lastUsedStep)
        {
            Left = left;
            Right = right;
            ThesisContradiction = thesisContradiction;
            LastObservedContradiction = lastObservedContradiction;
            SynthesisContradiction = synthesisContradiction;
            ObservationCount = observationCount;
            UseCount = useCount;
            SuccessCount = successCount;
            FailureCount = failureCount;
            LastUsedStep = lastUsedStep;
        }

        public string Left { get; set; }
        public string Right { get; set; }
        public double ThesisContradiction { get; set; }
        public double LastObservedContradiction { get; set; }
        public double SynthesisContradiction { get; set; }
        public int ObservationCount { get; set; }
        public int UseCount { get; set; }
        public int SuccessCount { get; set; }
        public int FailureCount { get; set; }
        public long LastUsedStep { get; set; }
    }

    public sealed record PlatonicQueryResult(
        string Text,
        double Confidence,
        int Hops,
        int ConceptCount);
}
