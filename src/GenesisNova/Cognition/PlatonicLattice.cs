using System.Globalization;

namespace GenesisNova.Cognition;

/// <summary>
/// Dual-layer spatial index over the platonic concept space — ported from the
/// genesis-engine source of truth (Genesis.Engine.GenesisLearner.PlatonicLattice) and
/// adapted to nova's string-keyed <c>ConceptNode</c> model.
/// <para>
/// Key FTD insight: the lattice IS the index. Position = identity = O(1)/O(log N) neighbours.
/// Discrete topology (graph edges) is separated from continuous dynamics (embedding KNN):
/// </para>
/// <list type="bullet">
///   <item><b>Layer 1 — adjacency:</b> bidirectional relation edges. Maintained incrementally
///   (hot path: degree + relational neighbours).</item>
///   <item><b>Layer 2 — typed lattices:</b> a sorted numeric lattice for O(log N) value-proximity
///   neighbours ("position IS address"), a char lattice for O(1) char lookup, and a VP-Tree for
///   O(log N) semantic KNN by face distance. The numeric/char indices rebuild lazily when the node
///   set changes; the VP-Tree rebuilds lazily on accumulated embedding drift.</item>
/// </list>
/// <para>
/// This replaces the prior <c>LatticeNeighborhood</c> relation-bag, which could only surface
/// neighbours that had an explicitly stored relation — value-proximity and embedding KNN were
/// structurally impossible there.
/// </para>
/// </summary>
internal sealed class PlatonicLattice
{
    private const int SemanticRebuildThreshold = 48;

    private readonly Func<IEnumerable<string>> _nodeNames;
    private readonly Func<IEnumerable<(string Name, double[] Face)>> _nodeFaces;

    // ── Layer 1: adjacency (incremental, always current) ──
    private readonly Dictionary<string, HashSet<string>> _adjacency = new(StringComparer.OrdinalIgnoreCase);

    // ── Layer 2: numeric + char (lazy, rebuilt on node-set change) ──
    private readonly SortedList<double, string> _numericLattice = new();
    private readonly Dictionary<char, string> _charLattice = new();
    private bool _topologyDirty = true;

    // ── Layer 2: semantic VP-Tree (lazy, drift-throttled) ──
    private VPTree? _semanticTree;
    private bool _semanticDirty = true;
    private bool _topologyChangedSinceBuild = true;
    private int _mutationsSinceBuild;

    public PlatonicLattice(
        Func<IEnumerable<string>> nodeNames,
        Func<IEnumerable<(string Name, double[] Face)>> nodeFaces)
    {
        _nodeNames = nodeNames;
        _nodeFaces = nodeFaces;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Mutations
    // ═══════════════════════════════════════════════════════════════

    /// <summary>A node was created. Invalidates the numeric/char indices and the semantic tree.</summary>
    public void RegisterNode(string name)
    {
        _topologyDirty = true;
        _topologyChangedSinceBuild = true;
        _semanticDirty = true;
    }

    /// <summary>A node was removed. Drops its adjacency and invalidates the lazy indices.</summary>
    public void UnregisterNode(string name)
    {
        if (_adjacency.TryGetValue(name, out var set))
        {
            foreach (var neighbor in set)
                if (_adjacency.TryGetValue(neighbor, out var reverse))
                    reverse.Remove(name);
            _adjacency.Remove(name);
        }
        _topologyDirty = true;
        _topologyChangedSinceBuild = true;
        _semanticDirty = true;
    }

    /// <summary>Add a bidirectional relation edge.</summary>
    public void AddEdge(string a, string b)
    {
        Link(a, b);
        Link(b, a);
    }

    /// <summary>Remove a bidirectional relation edge.</summary>
    public void RemoveEdge(string a, string b)
    {
        if (_adjacency.TryGetValue(a, out var sa)) sa.Remove(b);
        if (_adjacency.TryGetValue(b, out var sb)) sb.Remove(a);
    }

    /// <summary>Embeddings drifted (a face was mutated). Throttled accumulation toward a VP-Tree rebuild.</summary>
    public void MarkEmbeddingsDirty()
    {
        _semanticDirty = true;
        _mutationsSinceBuild++;
    }

    public void Clear()
    {
        _adjacency.Clear();
        _numericLattice.Clear();
        _charLattice.Clear();
        _semanticTree = null;
        _topologyDirty = true;
        _topologyChangedSinceBuild = true;
        _semanticDirty = true;
        _mutationsSinceBuild = 0;
    }

    private void Link(string from, string to)
    {
        if (!_adjacency.TryGetValue(from, out var set))
            _adjacency[from] = set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        set.Add(to);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Layer 1 queries — O(1)
    // ═══════════════════════════════════════════════════════════════

    /// <summary>Number of distinct related neighbours (relation degree).</summary>
    public int Degree(string name)
        => _adjacency.TryGetValue(name, out var set) ? set.Count : 0;

    /// <summary>Bidirectional relation neighbours of a concept.</summary>
    public IReadOnlyCollection<string> GetRelationalNeighbors(string name)
        => _adjacency.TryGetValue(name, out var set)
            ? (IReadOnlyCollection<string>)set
            : Array.Empty<string>();

    // ═══════════════════════════════════════════════════════════════
    //  Layer 2 queries — numeric (O(log N)) and char (O(1))
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Numeric neighbours within ±range of a value, nearest first. O(log N) via binary search.
    /// Position IS address: finds 6 and 8 near 7 without any stored relation.
    /// </summary>
    public IReadOnlyList<(string Name, double Distance)> GetNumericNeighbors(double value, double range, int k)
    {
        EnsureTopology();
        if (_numericLattice.Count == 0 || k <= 0)
            return Array.Empty<(string, double)>();

        var lo = value - range;
        var hi = value + range;
        var keys = _numericLattice.Keys;
        var values = _numericLattice.Values;
        var start = LowerBound(keys, lo);

        var results = new List<(string, double)>();
        for (var i = start; i < keys.Count; i++)
        {
            var key = keys[i];
            if (key > hi) break;
            var dist = Math.Abs(key - value);
            if (dist > 1e-12)
                results.Add((values[i], dist));
        }

        results.Sort((a, b) => a.Distance.CompareTo(b.Distance));
        return results.Count > k ? results.GetRange(0, k) : results;
    }

    /// <summary>Find the concept representing a single character. O(1).</summary>
    public string? FindChar(char c)
    {
        EnsureTopology();
        return _charLattice.GetValueOrDefault(c);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Layer 2 queries — semantic (O(log N))
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// k nearest concepts to a query face by embedding distance, via VP-Tree. O(log N).
    /// Replaces the prior brute-force O(N) Euclidean scan.
    /// </summary>
    public IReadOnlyList<(string Name, double Distance)> GetSemanticNeighbors(double[] queryFace, int k, string? excludeName)
    {
        EnsureSemantic();
        return _semanticTree is null
            ? Array.Empty<(string, double)>()
            : _semanticTree.KNearest(queryFace, k, excludeName);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Lazy rebuilds
    // ═══════════════════════════════════════════════════════════════

    private void EnsureTopology()
    {
        if (!_topologyDirty)
            return;

        _numericLattice.Clear();
        _charLattice.Clear();

        foreach (var name in _nodeNames())
        {
            if (double.TryParse(name, NumberStyles.Any, CultureInfo.InvariantCulture, out var numeric))
            {
                // Duplicate numeric values get a smallest-possible offset so the sorted lattice stays total.
                while (_numericLattice.ContainsKey(numeric))
                    numeric = BitIncrement(numeric);
                _numericLattice[numeric] = name;
            }
            else if (name.Length == 1)
            {
                _charLattice.TryAdd(name[0], name);
            }
        }

        _topologyDirty = false;
    }

    private void EnsureSemantic()
    {
        var needRebuild = _semanticTree is null
            || _topologyChangedSinceBuild
            || (_semanticDirty && _mutationsSinceBuild >= SemanticRebuildThreshold);
        if (!needRebuild)
            return;

        var names = new List<string>();
        var faces = new List<double[]>();
        foreach (var (name, face) in _nodeFaces())
        {
            names.Add(name);
            // Clone: the VP-Tree must be a consistent static snapshot while faces keep mutating in place.
            faces.Add((double[])face.Clone());
        }

        _semanticTree = new VPTree(names.ToArray(), faces.ToArray());
        _semanticDirty = false;
        _topologyChangedSinceBuild = false;
        _mutationsSinceBuild = 0;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Helpers
    // ═══════════════════════════════════════════════════════════════

    private static int LowerBound(IList<double> keys, double target)
    {
        int lo = 0, hi = keys.Count;
        while (lo < hi)
        {
            var mid = lo + ((hi - lo) / 2);
            if (keys[mid] < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    private static double BitIncrement(double value)
    {
        if (double.IsNaN(value) || double.IsPositiveInfinity(value)) return value;
        if (value == 0) return double.Epsilon;
        var bits = BitConverter.DoubleToInt64Bits(value);
        bits += value > 0 ? 1 : -1;
        return BitConverter.Int64BitsToDouble(bits);
    }
}
