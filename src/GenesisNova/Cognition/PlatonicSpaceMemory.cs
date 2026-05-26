namespace GenesisNova.Cognition;

public sealed class PlatonicSpaceMemory
{
    private readonly int _faceDimension;
    private readonly Random _rng;
    private readonly Dictionary<string, ConceptNode> _nodes = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, ConceptRelation> _relations = new(StringComparer.OrdinalIgnoreCase);

    public PlatonicSpaceMemory(int faceDimension, int seed = 42)
    {
        _faceDimension = Math.Max(4, faceDimension);
        _rng = new Random(seed);
    }

    public int NodeCount => _nodes.Count;
    public int RelationCount => _relations.Count;
    public int FaceDimension => _faceDimension;

    public IReadOnlyList<string> Concepts
        => _nodes.Keys.OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();

    public bool ContainsConcept(string concept)
        => _nodes.ContainsKey(Normalize(concept));

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

    public string DescribeConcept(string concept)
    {
        var key = Normalize(concept);
        if (!_nodes.TryGetValue(key, out var node))
            return $"concept '{concept}' not found";

        var neighbors = _relations.Values
            .Where(r => r.Left.Equals(key, StringComparison.OrdinalIgnoreCase) || r.Right.Equals(key, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(r => r.SynthesisContradiction)
            .Take(6)
            .Select(r =>
            {
                var other = r.Left.Equals(key, StringComparison.OrdinalIgnoreCase) ? r.Right : r.Left;
                return $"{other}:{r.SynthesisContradiction:F2}";
            })
            .ToArray();

        return neighbors.Length == 0
            ? $"concept={key} observations={node.ObservationCount} neighbors=none"
            : $"concept={key} observations={node.ObservationCount} neighbors=[{string.Join(", ", neighbors)}]";
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

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        _nodes.Clear();
        _relations.Clear();

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
            _relations[key] = new ConceptRelation(
                left: Normalize(relation.Left),
                right: Normalize(relation.Right),
                thesisContradiction: Clamp01(relation.ThesisContradiction),
                lastObservedContradiction: Clamp01(relation.LastObservedContradiction),
                synthesisContradiction: Clamp01(relation.SynthesisContradiction),
                observationCount: Math.Max(0, relation.ObservationCount));
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
            var unit = direction[i] / dist;
            var delta = learningRate * error * unit;
            a.PositiveFace[i] -= delta;
            b.PositiveFace[i] += delta;
        }

        // Soft complement coupling (dual-face coherence).
        for (var i = 0; i < _faceDimension; i++)
        {
            a.NegativeFace[i] = (0.95 * a.NegativeFace[i]) + (0.05 * -a.PositiveFace[i]);
            b.NegativeFace[i] = (0.95 * b.NegativeFace[i]) + (0.05 * -b.PositiveFace[i]);
        }
    }

    private ConceptNode GetOrCreate(string concept)
    {
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
            return node;

        node = new ConceptNode(
            name: key,
            positiveFace: CreateFace(),
            negativeFace: CreateFace().Select(x => -x).ToArray(),
            observationCount: 0);
        _nodes[key] = node;
        return node;
    }

    private double[] CreateFace()
    {
        var face = new double[_faceDimension];
        for (var i = 0; i < face.Length; i++)
            face[i] = (_rng.NextDouble() * 2.0 - 1.0) * 0.15;
        return face;
    }

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
}
