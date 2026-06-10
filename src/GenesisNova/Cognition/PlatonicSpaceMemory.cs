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
    private readonly Dictionary<string, LatticeNeighborhood> _latticeByConcept = new(StringComparer.OrdinalIgnoreCase);

    public PlatonicSpaceMemory(int faceDimension, int seed = 42)
    {
        _faceDimension = Math.Max(4, faceDimension);
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
        var budget = Math.Clamp(maxCandidates, 8, 512);
        if (!TryGetConceptFace(concept, out var conceptFace) || conceptFace.Length == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        IEnumerable<string> source = candidates is { Count: > 0 }
            ? candidates
            : _nodes.Keys;

        var scored = new List<(string Symbol, double Distance)>(Math.Min(budget, limit * 4));
        var seen = 0;
        foreach (var candidate in source)
        {
            if (seen >= budget)
                break;

            var normalized = Normalize(candidate);
            if (normalized.Equals(Normalize(concept), StringComparison.OrdinalIgnoreCase))
                continue;

            if (!_nodes.TryGetValue(normalized, out var node))
                continue;

            var distance = EuclideanDistance(conceptFace, node.PositiveFace);
            scored.Add((normalized, distance));
            seen++;
        }

        if (scored.Count == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        return scored
            .OrderBy(x => x.Distance)
            .Take(limit)
            .ToArray();
    }

    public int NumericDimensions => Math.Min(_faceDimension / 2, 21);
    public int LogFaceStart => NumericDimensions;

    public void ObserveContradiction(string left, string right, double observedContradiction)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            return;
        if (left.Equals(right, StringComparison.OrdinalIgnoreCase))
            return;

        var a = GetOrCreate(left);
        var b = GetOrCreate(right);
        a.ObservationCount++;
        b.ObservationCount++;

        var key = RelationKey(left, right);
        if (!_relations.TryGetValue(key, out var relation))
        {
            relation = new ConceptRelation(
                left: Normalize(left),
                right: Normalize(right),
                thesisContradiction: observedContradiction,
                lastObservedContradiction: observedContradiction,
                synthesisContradiction: observedContradiction,
                observationCount: 0);
            _relations[key] = relation;
            IndexRelation(relation);
        }

        relation.LastObservedContradiction = Clamp01(observedContradiction);
        relation.SynthesisContradiction = relation.ObservationCount == 0
            ? relation.LastObservedContradiction
            : (0.85 * relation.SynthesisContradiction) + (0.15 * relation.LastObservedContradiction);
        relation.ObservationCount++;

        UpdateConceptGeometry(a, b, relation.SynthesisContradiction);
        ApplyTriadicSynthesis(a.Name, b.Name);
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
                .Select(n => new PlatonicNodeSnapshot(n.Name, n.PositiveFace.ToArray(), n.NegativeFace.ToArray(), n.ObservationCount))
                .ToArray(),
            Relations: _relations.Values
                .Select(r => new PlatonicRelationSnapshot(r.Left, r.Right, r.ThesisContradiction, r.LastObservedContradiction, r.SynthesisContradiction, r.ObservationCount))
                .ToArray());
    }

    public PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops = 2,
        int beamWidth = 2)
    {
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

        for (var hop = 0; hop < hops; hop++)
        {
            var candidateScores = new Dictionary<string, List<double>>(StringComparer.OrdinalIgnoreCase);
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
            }

            frontier = selected.Select(x => x.Concept).ToArray();
        }

        if (decoded.Count == 0)
        {
            var first = anchors[0];
            return new PlatonicQueryResult(
                Text: first,
                Confidence: 0.42,
                Hops: 1,
                ConceptCount: 1);
        }

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
        if (!_latticeByConcept.TryGetValue(key, out var neighborhood))
            return Array.Empty<PlatonicNeighbor>();

        var selectedNeighbors = neighborhood.GetNeighbors(type);
        if (selectedNeighbors.Count == 0)
            return Array.Empty<PlatonicNeighbor>();

        var capped = Math.Clamp(maxNeighbors, 1, 256);
        var threshold = Clamp01(minConfidence);
        var top = new List<PlatonicNeighbor>(capped);

        foreach (var neighbor in selectedNeighbors)
        {
            if (!neighborhood.RelationsByNeighbor.TryGetValue(neighbor, out var relation))
                continue;

            var confidence = 1.0 - relation.SynthesisContradiction;
            if (confidence < threshold)
                continue;

            var neighborType = ClassifyConceptType(neighbor);
            var candidate = new PlatonicNeighbor(
                Concept: neighbor,
                Confidence: confidence,
                Type: neighborType,
                ObservationCount: relation.ObservationCount);
            InsertTopNeighbor(top, candidate, capped);
        }

        return top;
    }

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

        var inputNodes = inputs.Select(GetOrCreate).ToArray();
        var outputNodes = outputs.Select(GetOrCreate).ToArray();
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
        _latticeByConcept.Clear();

        foreach (var node in snapshot.Nodes)
        {
            var normalized = Normalize(node.Name);
            _nodes[normalized] = new ConceptNode(
                name: normalized,
                positiveFace: Resize(node.PositiveFace, _faceDimension),
                negativeFace: Resize(node.NegativeFace, _faceDimension),
                observationCount: Math.Max(0, node.ObservationCount));
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
                observationCount: Math.Max(0, relation.ObservationCount));
            _relations[key] = conceptRelation;
            IndexRelation(conceptRelation);
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
                n.ObservationCount))
            .ToDictionary(n => n.Name, StringComparer.OrdinalIgnoreCase);
        var relations = _relations.Values
            .Select(r => new MutableRelation(
                r.Left,
                r.Right,
                r.ThesisContradiction,
                r.LastObservedContradiction,
                r.SynthesisContradiction,
                r.ObservationCount))
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
                if (r.ObservationCount > minRelObs)
                    return false;
                if (r.SynthesisContradiction < maxSynthesis)
                    return false;
                var leftObs = nodes.TryGetValue(r.Left, out var leftNode) ? leftNode.ObservationCount : 0;
                var rightObs = nodes.TryGetValue(r.Right, out var rightNode) ? rightNode.ObservationCount : 0;
                return leftObs <= (minNodeObs + 1) && rightObs <= (minNodeObs + 1);
            })
            .OrderBy(r => r.ObservationCount)
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
                .Where(n => !protectedConcepts.Contains(n.Name))
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

                    var canonical = source.ObservationCount >= target.ObservationCount ? source : target;
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
                    .OrderBy(r => r.ObservationCount)
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
                    if (relation.ObservationCount > (minRelObs + 1) && relation.SynthesisContradiction < (maxSynthesis + 0.1))
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
                    !relationEndpoints.Contains(n.Name))
                .OrderBy(n => n.ObservationCount)
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
                .Select(n => new PlatonicNodeSnapshot(n.Name, n.PositiveFace, n.NegativeFace, Math.Max(0, n.ObservationCount)))
                .ToArray(),
            Relations: relations.Values
                .OrderBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
                .Select(r => new PlatonicRelationSnapshot(
                    r.Left,
                    r.Right,
                    Clamp01(r.ThesisContradiction),
                    Clamp01(r.LastObservedContradiction),
                    Clamp01(r.SynthesisContradiction),
                    Math.Max(0, r.ObservationCount)))
                .ToArray()));

        return new SpaceMaintenanceResult(
            RelationsPruned: relationsPruned,
            NodesPruned: nodesPruned,
            NodesMerged: nodesMerged);
    }

    private void IndexRelation(ConceptRelation relation)
    {
        var left = GetOrCreateNeighborhood(relation.Left);
        left.UpsertNeighbor(relation.Right, relation);

        var right = GetOrCreateNeighborhood(relation.Right);
        right.UpsertNeighbor(relation.Left, relation);
    }

    private LatticeNeighborhood GetOrCreateNeighborhood(string concept)
    {
        if (_latticeByConcept.TryGetValue(concept, out var neighborhood))
            return neighborhood;

        neighborhood = new LatticeNeighborhood();
        _latticeByConcept[concept] = neighborhood;
        return neighborhood;
    }

    private static void InsertTopNeighbor(List<PlatonicNeighbor> top, PlatonicNeighbor candidate, int maxNeighbors)
    {
        if (maxNeighbors <= 0)
            return;

        var inserted = false;
        for (var i = 0; i < top.Count; i++)
        {
            if (candidate.Confidence <= top[i].Confidence)
                continue;
            top.Insert(i, candidate);
            inserted = true;
            break;
        }

        if (!inserted && top.Count < maxNeighbors)
            top.Add(candidate);

        if (top.Count > maxNeighbors)
            top.RemoveAt(top.Count - 1);
    }

    private void ApplyTriadicSynthesis(string left, string right)
    {
        var leftKey = Normalize(left);
        var rightKey = Normalize(right);
        foreach (var other in _nodes.Keys)
        {
            if (other.Equals(leftKey, StringComparison.OrdinalIgnoreCase) ||
                other.Equals(rightKey, StringComparison.OrdinalIgnoreCase))
                continue;

            var l = GetContradiction(leftKey, other);
            var r = GetContradiction(rightKey, other);
            var predicted = Clamp01(0.5 + 0.5 * Math.Abs(l - r));

            var key = RelationKey(leftKey, rightKey);
            if (_relations.TryGetValue(key, out var relation))
            {
                relation.SynthesisContradiction = (0.9 * relation.SynthesisContradiction) + (0.1 * predicted);
                UpdateConceptGeometry(GetOrCreate(leftKey), GetOrCreate(rightKey), relation.SynthesisContradiction);
            }
        }
    }

    private void UpdateConceptGeometry(ConceptNode a, ConceptNode b, double targetContradiction)
    {
        var learningRate = 0.04;
        var targetDistance = 0.25 + (1.75 * Clamp01(targetContradiction));

        // Freeze arithmetic dims for numeric seeded concepts so polynomial/log seeds never drift.
        // Operators ("add", "mul") are intentionally NOT frozen — their geometry is trained.
        var arithmeticBoundary = 2 * NumericDimensions;
        var aIsNumeric = TryParseNumber(a.Name, out _);
        var bIsNumeric = TryParseNumber(b.Name, out _);
        var freezeArithmetic = (aIsNumeric || bIsNumeric) && arithmeticBoundary > 0 && arithmeticBoundary < _faceDimension;

        var direction = new double[_faceDimension];
        var distSquared = 0.0;
        for (var i = 0; i < _faceDimension; i++)
        {
            direction[i] = a.PositiveFace[i] - b.PositiveFace[i];
            distSquared += direction[i] * direction[i];
        }

        var dist = Math.Sqrt(Math.Max(1e-9, distSquared));
        var error = dist - targetDistance;
        for (var i = 0; i < _faceDimension; i++)
        {
            if (freezeArithmetic && i < arithmeticBoundary)
                continue;
            var unit = direction[i] / dist;
            var delta = learningRate * error * unit;
            a.PositiveFace[i] -= delta;
            b.PositiveFace[i] += delta;
        }

        // Soft complement coupling (dual-face coherence).
        for (var i = 0; i < _faceDimension; i++)
        {
            if (freezeArithmetic && i < arithmeticBoundary)
                continue;
            a.NegativeFace[i] = (0.95 * a.NegativeFace[i]) + (0.05 * -a.PositiveFace[i]);
            b.NegativeFace[i] = (0.95 * b.NegativeFace[i]) + (0.05 * -b.PositiveFace[i]);
        }
    }

    private ConceptNode GetOrCreate(string concept)
    {
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
            return node;

        var positiveFace = TryCreateSeededFace(key, out var seeded)
            ? seeded
            : CreateFace(key);
        node = new ConceptNode(
            name: key,
            positiveFace: positiveFace,
            negativeFace: positiveFace.Select(x => -x).ToArray(),
            observationCount: 0);
        _nodes[key] = node;
        return node;
    }

    private bool TryCreateSeededFace(string concept, out double[] face)
    {
        if (TryParseNumber(concept, out var numeric))
        {
            face = CreateNumericFace(numeric);
            return true;
        }

        if (IsAddOperator(concept))
        {
            face = CreateOperatorFace(preferPoly: true);
            return true;
        }

        if (IsMultiplyOperator(concept))
        {
            face = CreateOperatorFace(preferPoly: false);
            return true;
        }

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

    private double[] CreateOperatorFace(bool preferPoly)
    {
        var face = new double[_faceDimension];
        var numericDims = Math.Min(_faceDimension / 2, 21);
        var logStart = numericDims;
        var logDims = Math.Min(numericDims, _faceDimension - logStart);

        if (preferPoly)
        {
            for (var i = 0; i < numericDims; i++)
                face[i] = 0.08 * Math.Pow(10, -(i + 1));
            for (var i = 0; i < logDims; i++)
                face[logStart + i] = 0.0;
        }
        else
        {
            for (var i = 0; i < numericDims; i++)
                face[i] = 0.0;
            for (var i = 0; i < logDims; i++)
                face[logStart + i] = 0.08 * Math.Pow(10, -(i + 1));
        }

        return face;
    }

    private double[] CreateFace(string concept)
    {
        var face = new double[_faceDimension];
        var hash = StableHash(concept);
        for (var i = 0; i < face.Length; i++)
        {
            hash = NextHash(hash, i);
            var unit = (hash & 0xFFFF) / 65535.0;
            face[i] = ((unit * 2.0) - 1.0) * 0.08;
        }
        return face;
    }

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
        for (var i = 0; i < _faceDimension; i++)
        {
            var delta = centroid[i] - node.PositiveFace[i];
            node.PositiveFace[i] += sign * rate * delta;
            node.NegativeFace[i] = (0.95 * node.NegativeFace[i]) + (0.05 * -node.PositiveFace[i]);
        }
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
    {
        if (TryParseNumber(token, out _))
            return PlatonicNeighborhoodType.Numeric;
        if (IsAddOperator(token) || IsMultiplyOperator(token))
            return PlatonicNeighborhoodType.Relational;
        return PlatonicNeighborhoodType.Semantic;
    }

    private static bool TryParseNumber(string token, out double value)
        => double.TryParse(token, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out value);

    private static bool IsAddOperator(string token)
        => token is "+" or "plus" or "add" or "sum";

    private static bool IsMultiplyOperator(string token)
        => token is "*" or "x" or "times" or "multiply" or "product";

    private static uint StableHash(string value)
    {
        uint h = 2166136261;
        foreach (var c in value)
        {
            h ^= c;
            h *= 16777619;
        }

        return h;
    }

    private static uint NextHash(uint hash, int salt)
    {
        unchecked
        {
            hash ^= (uint)(salt * 16777619);
            hash *= 2246822519u;
            hash ^= hash >> 13;
            hash *= 3266489917u;
            hash ^= hash >> 16;
            return hash;
        }
    }

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

    private sealed class LatticeNeighborhood
    {
        public Dictionary<string, ConceptRelation> RelationsByNeighbor { get; } = new(StringComparer.OrdinalIgnoreCase);
        private HashSet<string> NumericNeighbors { get; } = new(StringComparer.OrdinalIgnoreCase);
        private HashSet<string> RelationalNeighbors { get; } = new(StringComparer.OrdinalIgnoreCase);
        private HashSet<string> SemanticNeighbors { get; } = new(StringComparer.OrdinalIgnoreCase);

        public void UpsertNeighbor(string neighbor, ConceptRelation relation)
        {
            RelationsByNeighbor[neighbor] = relation;
            NumericNeighbors.Remove(neighbor);
            RelationalNeighbors.Remove(neighbor);
            SemanticNeighbors.Remove(neighbor);

            switch (ClassifyConceptType(neighbor))
            {
                case PlatonicNeighborhoodType.Numeric:
                    NumericNeighbors.Add(neighbor);
                    break;
                case PlatonicNeighborhoodType.Relational:
                    RelationalNeighbors.Add(neighbor);
                    break;
                default:
                    SemanticNeighbors.Add(neighbor);
                    break;
            }
        }

        public IReadOnlyCollection<string> GetNeighbors(PlatonicNeighborhoodType type)
        {
            return type switch
            {
                PlatonicNeighborhoodType.Numeric => NumericNeighbors,
                PlatonicNeighborhoodType.Relational => RelationalNeighbors,
                PlatonicNeighborhoodType.Semantic => SemanticNeighbors,
                _ => RelationsByNeighbor.Keys
            };
        }
    }

    private sealed class ConceptNode
    {
        public ConceptNode(string name, double[] positiveFace, double[] negativeFace, int observationCount)
        {
            Name = name;
            PositiveFace = positiveFace;
            NegativeFace = negativeFace;
            ObservationCount = observationCount;
        }

        public string Name { get; }
        public double[] PositiveFace { get; }
        public double[] NegativeFace { get; }
        public int ObservationCount { get; set; }
    }

    private sealed class ConceptRelation
    {
        public ConceptRelation(
            string left,
            string right,
            double thesisContradiction,
            double lastObservedContradiction,
            double synthesisContradiction,
            int observationCount)
        {
            Left = left;
            Right = right;
            ThesisContradiction = thesisContradiction;
            LastObservedContradiction = lastObservedContradiction;
            SynthesisContradiction = synthesisContradiction;
            ObservationCount = observationCount;
        }

        public string Left { get; }
        public string Right { get; }
        public double ThesisContradiction { get; }
        public double LastObservedContradiction { get; set; }
        public double SynthesisContradiction { get; set; }
        public int ObservationCount { get; set; }
    }

    private sealed class MutableNode
    {
        public MutableNode(string name, double[] positiveFace, double[] negativeFace, int observationCount)
        {
            Name = name;
            PositiveFace = positiveFace;
            NegativeFace = negativeFace;
            ObservationCount = observationCount;
        }

        public string Name { get; }
        public double[] PositiveFace { get; }
        public double[] NegativeFace { get; }
        public int ObservationCount { get; set; }
    }

    private sealed class MutableRelation
    {
        public MutableRelation(
            string left,
            string right,
            double thesisContradiction,
            double lastObservedContradiction,
            double synthesisContradiction,
            int observationCount)
        {
            Left = left;
            Right = right;
            ThesisContradiction = thesisContradiction;
            LastObservedContradiction = lastObservedContradiction;
            SynthesisContradiction = synthesisContradiction;
            ObservationCount = observationCount;
        }

        public string Left { get; set; }
        public string Right { get; set; }
        public double ThesisContradiction { get; set; }
        public double LastObservedContradiction { get; set; }
        public double SynthesisContradiction { get; set; }
        public int ObservationCount { get; set; }
    }

    public sealed record PlatonicQueryResult(
        string Text,
        double Confidence,
        int Hops,
        int ConceptCount);
}
