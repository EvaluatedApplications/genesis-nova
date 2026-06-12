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

public sealed class PlatonicSpaceMemory
{
    private const int MaxPlatonicNodes = 12_000;
    private const int MaxPlatonicRelations = 48_000;
    private const double GeometryLearningRate = 0.04;

    // Contrastive repulsion — ported from the genesis source of truth (GraphAligner.NudgeGraphAlignment,
    // which "push[es] away from sampled non-neighbours ... maintain[s] discriminability between unrelated
    // elements"). Nova previously ONLY attracted observed pairs, so unrelated/fringe concepts drifted
    // together. This throttled batch pass pushes each concept away from a few sampled UNRELATED concepts
    // so proximity must be EARNED by confirmed relations. Runs every RepulsionInterval observations.
    // Strength is genesis-faithful: repulsion is a GENTLE discriminability pressure (~0.1x attraction),
    // not a strong separator. A stronger setting separates unrelated concepts more but overpowers weak
    // genuine attractions (e.g. text→frozen-number equivalence), so we keep the source-of-truth scale.
    private const int RepulsionInterval = 16;
    private const int RepulsionSamples = 5;
    private const double RepulsionRate = 0.02;
    private const double RepulsionRatio = 0.1; // repulsion strength relative to RepulsionRate floor
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
    private readonly Dictionary<string, ConceptNode> _nodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, ConceptRelation> _relations = new(StringComparer.OrdinalIgnoreCase);
    private readonly PlatonicLattice _lattice;
    private long _utilityStep;

    public PlatonicSpaceMemory(int faceDimension, int seed = 42)
    {
        _faceDimension = Math.Max(4, faceDimension);
        _lattice = new PlatonicLattice(
            nodeNames: () => _nodes.Keys,
            nodeFaces: () => _nodes.Values.Select(n => (n.Name, n.PositiveFace)));
    }

    public int NodeCount => _nodes.Count;
    public int RelationCount => _relations.Count;
    public int FaceDimension => _faceDimension;

    public IReadOnlyList<string> Concepts
        => _nodes.Keys.OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();

    public bool ContainsConcept(string concept)
        => _nodes.ContainsKey(Normalize(concept));

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
                scored.Add((normalized, EuclideanDistance(conceptFace, node.PositiveFace)));
                seen++;
            }
            return scored.Count == 0
                ? Array.Empty<(string Symbol, double Distance)>()
                : scored.OrderBy(x => x.Distance).Take(limit).ToArray();
        }

        // Global nearest: O(log N) VP-Tree over the whole space (replaces the brute-force O(N) scan).
        return _lattice.GetSemanticNeighbors(conceptFace, limit, conceptKey);
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
        if (!_relations.TryGetValue(key, out var relation))
        {
            EnsureRelationCapacity();
            relation = new ConceptRelation(
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
            _relations[key] = relation;
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
        return _relations.TryGetValue(key, out var relation)
            ? relation.SynthesisContradiction
            : 0.5;
    }

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
            Relations: _relations.Values
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
               .ToArray());
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
                var neighbors = GetNeighbors(
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
            if (!collected.TryGetValue(candidate.Concept, out var existing) || candidate.Confidence > existing.Confidence)
                collected[candidate.Concept] = candidate;
        }

        var wantRelational = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Relational;
        var wantNumeric = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Numeric;
        var wantSemantic = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Semantic;

        // Relational tier: explicit relation edges; confidence from the learned contradiction.
        if (wantRelational)
        {
            foreach (var neighbor in _lattice.GetRelationalNeighbors(key))
            {
                if (!_relations.TryGetValue(RelationKey(key, neighbor), out var relation))
                    continue;
                Consider(new PlatonicNeighbor(
                    Concept: neighbor,
                    Confidence: 1.0 - relation.SynthesisContradiction,
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
        foreach (var rel in _relations.Values)
        {
            result.Add((rel.Left, rel.Right, rel.ObservationCount));
        }
        return result;
    }

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        _nodes.Clear();
        _relations.Clear();
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
            var conceptRelation = new ConceptRelation(
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
            _relations[key] = conceptRelation;
            IndexRelation(conceptRelation);
        }
        _utilityStep = Math.Max(
            _nodes.Values.Select(n => n.LastUsedStep).DefaultIfEmpty(0).Max(),
            _relations.Values.Select(r => r.LastUsedStep).DefaultIfEmpty(0).Max());
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

            if (related is null || !_relations.TryGetValue(RelationKey(concept, related), out var relation))
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
        var relations = _relations.Values
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
    // (contradiction, observation counts) stays in _relations, keyed by RelationKey.
    private void IndexRelation(ConceptRelation relation)
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
        var dim = _faceDimension;
        _repulsionPassCount++;

        foreach (var node in snapshot)
        {
            if (IsFrozenConcept(node.Name)) // ground-truth numerics never move
                continue;

            var original = (double[])node.PositiveFace.Clone();
            var updated = node.PositiveFace;
            var rng = new Random(unchecked(StableHash(node.Name) + _repulsionPassCount));

            var samples = Math.Min(RepulsionSamples, n - 1);
            for (var s = 0; s < samples; s++)
            {
                var other = snapshot[rng.Next(snapshot.Length)];
                if (ReferenceEquals(other, node))
                    continue;
                // Repulsion is ONLY for unrelated pairs — a confirmed relation is exempt (it is what
                // earns proximity). This is the contrastive half: attract related, repel everything else.
                if (_relations.ContainsKey(RelationKey(node.Name, other.Name)))
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

    private int GetRelationDegree(string concept)
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

    private static void TouchRelation(ConceptRelation relation, bool success, long step)
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

    private double RelationUtilityScore(ConceptRelation relation)
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
    ///   lexical fingerprint. The word face and all other dims stay learnable.</item>
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
        if (_nodes.Count < MaxPlatonicNodes)
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
        if (_relations.Count < MaxPlatonicRelations)
            return;

        var candidate = _relations.Values
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
        var incident = _relations.Values
            .Where(r => r.Left.Equals(key, StringComparison.OrdinalIgnoreCase) ||
                        r.Right.Equals(key, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        foreach (var relation in incident)
            RemoveRelation(relation);

        _nodes.Remove(key);
        _lattice.UnregisterNode(key);
    }

    private void RemoveRelation(ConceptRelation relation)
    {
        _relations.Remove(RelationKey(relation.Left, relation.Right));
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

    private sealed class ConceptRelation
    {
        public ConceptRelation(
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
