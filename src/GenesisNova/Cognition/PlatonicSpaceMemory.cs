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

public sealed partial class PlatonicSpaceMemory : IPlatonicSpace
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

    // PHASE 1 — DIMENSIONAL CONTRADICTION (the dialectic; PLATONIC_DIALECTIC.md §2, PLATONIC_THEORY.md Law D).
    // The legacy pull applied ONE affinity to EVERY free dim, contracting all of them toward the neighbour — so a
    // related pair collapsed toward IDENTICAL, erasing the contrasts that give them distinct meaning (violates Law
    // M: "cat≈dog on `animal` but cat≠dog on `sound`"). When DimensionalContradiction is on, MessagePassUpdate acts
    // PER DIMENSION: dims where the pair currently AGREES (face products ≥ 0) get the full pull/push (reinforce the
    // shared aspect / break a false agreement); dims where it currently CONTRADICTS (products < 0) move only at
    // DialecticPreserveFraction — the distinguishing aspects are PRESERVED, so the concept's place emerges as the
    // synthesis of what it shares and what it opposes. Frozen identity dims are restored regardless (homomorphism
    // untouched). Toggle so the scalar baseline can be A/B-measured (PlatonicDialecticTests).
    public bool DimensionalContradiction { get; set; } = true;
    private const double DialecticPreserveFraction = 0.15; // fraction of the pull applied on CONTRADICTING dims

    // C2: the free region is NO LONGER pinned to a unit sphere — the RADIAL axis is live so concepts can sit at
    // different DISTANCES (related pulled in close; unrelated pushed far out). NormaliseFreeRegion only CLAMPS the
    // norm to this ceiling as a numeric blow-up guard against repeated repulsion; below it, faces roam freely.
    private const double MaxFreeNorm = 5.0;

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
    private const double RepulsionRatio = 0.7; // repulsion strength relative to RepulsionRate floor (raised 0.5→0.7
                                               // for more discriminability — unrelated need to sit FURTHER apart for
                                               // KNN to separate them; related pairs are repulsion-EXEMPT so the pull holds)

    // INFONCE PUSH (PLATONIC_BACKPROP.md §3-4): the closed-form gradient of the contrastive loss's repulsion
    // term — push each negative weighted by softmax(-distance/τ), so the HARDEST (nearest) negative gets the
    // most gradient automatically (self-scaling; the single InfoNceStep replaces RepulsionRate·RepulsionRatio).
    // The manual fixed-force push is the gradient of an INDEPENDENT squared-distance loss; this is the gradient
    // of the *softmax* InfoNCE loss — the literal "backprop the geometry" step, computed analytically because
    // the faces are double[] (same gradient, no per-node torch tensors). Default OFF: opt-in, A/B vs the manual
    // push on the geometry gate before trusting it on the live model.
    public bool UseInfoNceRepulsion { get; set; }
    private const double InfoNceStep = 0.6;   // C4: de-spiked from 0.9. NOTE: InfoNCE is now OFF in the live runtime
                                              // (UseInfoNceRepulsion=false) because its push is PROPORTIONAL to the gap
                                              // and so grows unbounded OFF the unit sphere (C2 removed the sphere). The
                                              // stable MANUAL constant-step push is live. This only matters if InfoNCE
                                              // is re-enabled — which would require restoring some normalization.
    private const double InfoNceTau = 0.25;
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
    // AXIOM G6 (Irreversibility): "once a distinction has been made it cannot be unmade — the platonic space only
    // expands." Maintenance/capacity pressure therefore ARCHIVES rather than DESTROYS: an evicted node/relation is
    // moved DORMANT here (removed from the active lattice + geometry so the ACTIVE space stays bounded for speed)
    // with its learned structure RETAINED, and is REACTIVATED intact if it is ever re-observed. This is what stops
    // the space forgetting rare-but-real structure (the failure that killed the original engine's symbolic learner).
    private readonly Dictionary<string, ConceptNode> _archivedNodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, RelationElementNode> _archivedRelations = new(StringComparer.OrdinalIgnoreCase);
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
    // G6: dormant (archived) structure retained but excluded from the active space. These only ever grow.
    public int ArchivedNodeCount => _archivedNodes.Count;
    public int ArchivedRelationCount => _archivedRelations.Count;
    public int FaceDimension => _faceDimension;

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

    /// <summary>The magnitude of separation the contrastive dynamics (pull related / push unrelated) achieved.</summary>
    public sealed record GeometrySummary(
        int TotalConcepts, int MutableConcepts, int RelatedPairs, int UnrelatedPairs,
        double RelatedMean, double RelatedMin, double RelatedMax,
        double UnrelatedMean, double UnrelatedMin, double UnrelatedMax)
    {
        /// <summary>How much farther apart unrelated concepts sit than related ones — the push/pull gap.</summary>
        public double Separation => UnrelatedMean - RelatedMean;
    }

    public int NumericDimensions => Math.Min(_faceDimension / 2, 21);

    // Shapes-as-Function-elements and the Seq chunk-element store are each their own class (SRP); the memory
    // delegates. Both are kept out of _nodes so they never contaminate concept retrieval. The executable
    // glider definitions live in PlatonicShapeRegistry; the Function elements here are their substrate existence.
    private readonly FunctionElementRegistry _functions;
    private readonly WordElementRegistry _words;
    private readonly ChunkElementStore _chunks = new();

    /// <summary>The canonical tag under which the Seq composer mines/looks-up its scaffold chunk.</summary>
    public const string SeqScaffoldTag = "⟨seq-scaffold⟩";

    // How much a single wrong-answer disruption raises the producing edge's contradiction (weakens it).
    private const double DisruptionContradictionStep = 0.08;

    // Step size for the analytic function gradient (applied to the FREE region, then renormalised).
    private const double FunctionGradientRate = 0.05;

    private static string Normalize(string value)
        => value.Trim().ToLowerInvariant();

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
