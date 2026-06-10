namespace GenesisNova.Cognition;

public sealed class PlatonicSpaceMemory
{
    private readonly int _faceDimension;
    private readonly Dictionary<string, ConceptNode> _nodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, ConceptRelation> _relations = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<ConceptRelation>> _relationsBySource = new(StringComparer.OrdinalIgnoreCase);

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
            
            // Add to source index (both directions since relations are bidirectional in queries)
            if (!_relationsBySource.TryGetValue(relation.Left, out var sourceList))
            {
                sourceList = new List<ConceptRelation>();
                _relationsBySource[relation.Left] = sourceList;
            }
            sourceList.Add(relation);
            
            if (!_relationsBySource.TryGetValue(relation.Right, out var targetList))
            {
                targetList = new List<ConceptRelation>();
                _relationsBySource[relation.Right] = targetList;
            }
            targetList.Add(relation);
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
                if (!_relationsBySource.TryGetValue(source, out var relations))
                    continue;
                
                foreach (var relation in relations)
                {
                    string? target = null;
                    if (relation.Left.Equals(source, StringComparison.OrdinalIgnoreCase))
                        target = relation.Right;
                    else if (relation.Right.Equals(source, StringComparison.OrdinalIgnoreCase))
                        target = relation.Left;

                    if (target is null || seen.Contains(target))
                        continue;

                    var confidence = 1.0 - relation.SynthesisContradiction;
                    if (confidence < 0.35)
                        continue;

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
        _relationsBySource.Clear();

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
            
            // Add to source index (both directions)
            if (!_relationsBySource.TryGetValue(conceptRelation.Left, out var sourceList))
            {
                sourceList = new List<ConceptRelation>();
                _relationsBySource[conceptRelation.Left] = sourceList;
            }
            sourceList.Add(conceptRelation);
            
            if (!_relationsBySource.TryGetValue(conceptRelation.Right, out var targetList))
            {
                targetList = new List<ConceptRelation>();
                _relationsBySource[conceptRelation.Right] = targetList;
            }
            targetList.Add(conceptRelation);
        }
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

    public sealed record PlatonicQueryResult(
        string Text,
        double Confidence,
        int Hops,
        int ConceptCount);
}
