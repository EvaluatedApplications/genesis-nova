namespace GenesisNova.Cognition;

/// <summary>
/// Vantage-Point tree for O(log N) nearest-neighbour queries in face space.
/// Ported from the genesis-engine source of truth (Genesis.Engine.GenesisLearner.VPTree)
/// and adapted to nova's string-keyed concept model: concepts are identified by name
/// rather than integer id.
/// <para>
/// Critical for scaling beyond a few thousand concepts where the previous brute-force
/// O(N) nearest-neighbour scan in <c>GetNearestConcepts</c> becomes the bottleneck and,
/// more importantly, where semantic neighbours must be discoverable at all (the prior
/// design could only surface neighbours that had an explicitly stored relation).
/// </para>
/// </summary>
public sealed class VPTree
{
    private readonly VPNode? _root;
    private readonly double[][] _points;   // embedding vectors indexed by position
    private readonly string[] _ids;        // concept names indexed by position
    // FACE-AWARE distance: compare ONLY the dims [_rangeStart.._rangeEnd) so a face-scoped metric (e.g.
    // the semantic face) shields the numeric/char faces from contaminating the comparison. -1 = to end.
    private readonly int _rangeStart;
    private readonly int _rangeEnd;

    /// <summary>
    /// Build a tree from parallel arrays of concept names and their face embeddings.
    /// The arrays are referenced, not copied; callers must not mutate them after construction.
    /// </summary>
    /// <param name="names">Concept name per position.</param>
    /// <param name="embeddings">Face embedding per position (same length as <paramref name="names"/>).</param>
    /// <param name="seed">Deterministic vantage-point selection seed (matches source default of 42).</param>
    public VPTree(string[] names, double[][] embeddings, int seed = 42, int rangeStart = 0, int rangeEnd = -1)
    {
        if (names.Length != embeddings.Length)
            throw new ArgumentException("names and embeddings must be the same length");

        _rangeStart = Math.Max(0, rangeStart);
        _rangeEnd = rangeEnd;

        if (names.Length == 0)
        {
            _root = null;
            _points = Array.Empty<double[]>();
            _ids = Array.Empty<string>();
            return;
        }

        _points = embeddings;
        _ids = names;
        var indices = new int[names.Length];
        for (var i = 0; i < names.Length; i++)
            indices[i] = i;

        _root = Build(indices, 0, indices.Length, new Random(seed));
    }

    public int Count => _ids.Length;

    /// <summary>
    /// Find the k nearest neighbours to a query embedding. O(log N) average.
    /// </summary>
    /// <param name="query">Query face embedding.</param>
    /// <param name="k">Number of neighbours to return.</param>
    /// <param name="excludeName">Optional concept name to exclude (e.g. the query concept itself).</param>
    public IReadOnlyList<(string Name, double Distance)> KNearest(double[] query, int k, string? excludeName = null)
    {
        if (_root is null || k <= 0)
            return Array.Empty<(string, double)>();

        // SortedList ordered by distance; DuplicateKeyComparer keeps ties and orders ascending,
        // so the last entry is always the current farthest of the k-best set.
        var heap = new SortedList<double, int>(new DuplicateKeyComparer());
        var tau = double.MaxValue;

        Search(_root, query, k, excludeName, heap, ref tau);

        var result = new List<(string, double)>(heap.Count);
        foreach (var kvp in heap)
            result.Add((_ids[kvp.Value], kvp.Key));

        result.Sort((a, b) => a.Item2.CompareTo(b.Item2));
        return result;
    }

    // ── Build ──

    private VPNode? Build(int[] indices, int start, int end, Random rng)
    {
        if (start >= end)
            return null;
        if (start + 1 == end)
            return new VPNode(indices[start], 0, null, null);

        // Pick a random vantage point, swap it to the start.
        var vpIdx = start + rng.Next(end - start);
        (indices[start], indices[vpIdx]) = (indices[vpIdx], indices[start]);
        var vp = indices[start];

        var count = end - start - 1;
        var dists = new (double Dist, int Idx)[count];
        for (var i = 0; i < count; i++)
        {
            var idx = indices[start + 1 + i];
            dists[i] = (EuclideanDistance(_points[vp], _points[idx]), idx);
        }

        Array.Sort(dists, (a, b) => a.Dist.CompareTo(b.Dist));
        var median = count / 2;
        var mu = dists[median].Dist;

        for (var i = 0; i < count; i++)
            indices[start + 1 + i] = dists[i].Idx;

        var mid = start + 1 + median;
        var left = Build(indices, start + 1, mid, rng);
        var right = Build(indices, mid, end, rng);

        return new VPNode(vp, mu, left, right);
    }

    // ── Search ──

    private void Search(VPNode node, double[] query, int k, string? excludeName,
        SortedList<double, int> heap, ref double tau)
    {
        var dist = EuclideanDistance(_points[node.Index], query);

        if (excludeName is null || !string.Equals(_ids[node.Index], excludeName, StringComparison.OrdinalIgnoreCase))
        {
            if (heap.Count < k)
            {
                heap.Add(dist, node.Index);
                if (heap.Count == k)
                    tau = heap.Keys[heap.Count - 1];
            }
            else if (dist < tau)
            {
                heap.RemoveAt(heap.Count - 1);
                heap.Add(dist, node.Index);
                tau = heap.Keys[heap.Count - 1];
            }
        }

        if (node.Left is null && node.Right is null)
            return;

        if (dist < node.Mu)
        {
            if (node.Left is not null && dist - tau < node.Mu)
                Search(node.Left, query, k, excludeName, heap, ref tau);
            if (node.Right is not null && dist + tau >= node.Mu)
                Search(node.Right, query, k, excludeName, heap, ref tau);
        }
        else
        {
            if (node.Right is not null && dist + tau >= node.Mu)
                Search(node.Right, query, k, excludeName, heap, ref tau);
            if (node.Left is not null && dist - tau < node.Mu)
                Search(node.Left, query, k, excludeName, heap, ref tau);
        }
    }

    // Face-scoped Euclidean distance: only dims [_rangeStart .. min(_rangeEnd, len)) contribute, so the
    // tree is built AND queried in the same face subspace (e.g. the semantic face), excluding the
    // numeric/char faces that would otherwise dominate by magnitude / cluster by value or spelling.
    private double EuclideanDistance(double[] a, double[] b)
    {
        var n = Math.Min(a.Length, b.Length);
        var end = _rangeEnd < 0 ? n : Math.Min(_rangeEnd, n);
        var start = Math.Min(_rangeStart, end);
        var sum = 0.0;
        for (var i = start; i < end; i++)
        {
            var d = a[i] - b[i];
            sum += d * d;
        }
        return Math.Sqrt(sum);
    }

    // ── Internal types ──

    private sealed record VPNode(int Index, double Mu, VPNode? Left, VPNode? Right);

    /// <summary>
    /// Comparer that allows duplicate keys in SortedList (needed for distance ties).
    /// Never returns 0 — equal keys are treated as greater so both are retained.
    /// </summary>
    private sealed class DuplicateKeyComparer : IComparer<double>
    {
        public int Compare(double x, double y)
        {
            var result = x.CompareTo(y);
            return result == 0 ? 1 : result;
        }
    }
}
