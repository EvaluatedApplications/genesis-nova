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
    // ── Rebuild policy (ADAPTIVE, fraction-of-space) ──────────────────────────────────────────────────────
    // A full index rebuild is O(N); doing one on EVERY mutation makes per-example queries O(N^2) (a training
    // hang). But a FIXED threshold either over-rebuilds when the space is tiny or STARVES ("dirty forever,
    // never recalculated") when it is huge. So we rebuild once a FRACTION of the space has changed, with a
    // small floor: the cadence scales with size and can never starve. A smaller fraction = a fresher global
    // index = more accurate bulk retrieval (and discovery of brand-new neighbourhoods a moved concept drifts
    // into), at amortized O(N log N) cost — negligible per-step even at a million nodes. This is the
    // global-correctness FLOOR. WITHIN-STEP edit verification does NOT depend on it: GetNearestConceptsFresh
    // scores live faces over a bounded candidate set, so a correct edit is recognised in the SAME step
    // regardless of when this last fired. Lower the fraction to spend compute on accuracy.
    private const int RebuildMinMutations = 16;
    // PERF: raised 0.05 → 0.20 — a global VP-tree rebuild re-assembles ALL N faces (no assembled-face cache), so at ~10k
    // nodes it was firing every ~50 examples and dominating substrate time. Within-step correctness is unaffected
    // (GetNearestConceptsFresh seeds live adjacency + always live-rescores); a staler global tree only widens the
    // candidate pool slightly. Quarters the rebuild churn.
    private const double RebuildSpaceFraction = 0.20;

    // Strict numeric classification: a genuine concept token is a plain signed decimal.
    // NumberStyles.Any would accept trailing-sign garbage like "0+"/"5-" (parses as 0/5),
    // plus thousands/exponent/currency/parens — none of which a real concept token has.
    // This must stay aligned with PlatonicSpaceMemory.TryParseNumber so malformed tokens are
    // never indexed as numeric.
    private const NumberStyles NumericStyle =
        NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint
        | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite;

    private readonly Func<IEnumerable<string>> _nodeNames;
    private readonly Func<IEnumerable<(string Name, double[] Face)>> _nodeFaces;

    // ── Layer 1: adjacency (incremental, always current) ──
    private readonly Dictionary<string, HashSet<string>> _adjacency = new(StringComparer.OrdinalIgnoreCase);

    // ── Layer 2: numeric + char (lazy; rebuilt on the same ADAPTIVE fraction policy as the semantic tree) ──
    private readonly SortedList<double, string> _numericLattice = new();
    private readonly Dictionary<char, string> _charLattice = new();
    private int _topologyChanges;
    private int _lastTopologyCount;
    private bool _numericBuilt;

    // ── Layer 2: semantic VP-Tree (lazy; rebuilt on the adaptive fraction policy) ──
    private VPTree? _semanticTree;
    private bool _semanticDirty = true;
    private int _mutationsSinceBuild;
    private int _lastSemanticCount;

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
        _topologyChanges++;
        _mutationsSinceBuild++;
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
        _topologyChanges++;
        _mutationsSinceBuild++;
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
        _topologyChanges = 0;
        _lastTopologyCount = 0;
        _numericBuilt = false;
        _semanticDirty = true;
        _mutationsSinceBuild = 0;
        _lastSemanticCount = 0;
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

        results.Sort((a, b) => a.Item2.CompareTo(b.Item2));
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
        // ADAPTIVE rebuild (see rebuild-policy note): refresh on first use or once a FRACTION of the space has
        // changed (floor RebuildMinMutations) — never on every node (O(N^2)) and never starving at scale.
        var trigger = Math.Max(RebuildMinMutations, (int)(RebuildSpaceFraction * _lastTopologyCount));
        if (_numericBuilt && _topologyChanges < trigger)
            return;

        _numericLattice.Clear();
        _charLattice.Clear();

        var total = 0;
        foreach (var name in _nodeNames())
        {
            total++;
            if (double.TryParse(name, NumericStyle, CultureInfo.InvariantCulture, out var numeric))
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

        _lastTopologyCount = total;
        _numericBuilt = true;
        _topologyChanges = 0;
    }

    private void EnsureSemantic()
    {
        // ADAPTIVE rebuild (see rebuild-policy note): rebuild on first use or once a FRACTION of the space has
        // drifted (floor RebuildMinMutations). This is the global-correctness floor — within-step accuracy
        // comes from live-face candidate scoring (GetNearestConceptsFresh), not from how recently this fired.
        var trigger = Math.Max(RebuildMinMutations, (int)(RebuildSpaceFraction * _lastSemanticCount));
        var needRebuild = _semanticTree is null
            || (_semanticDirty && _mutationsSinceBuild >= trigger);
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

        // FACE-AWARE semantic KNN: compare only the SEMANTIC face [WordFaceStart..dim) so value
        // proximity (numeric face) and spelling (char face) cannot contaminate relatedness — the genesis
        // rule that stops "3 is nearest 4" from dominating semantic retrieval. Falls back to the whole
        // vector when there is no word/semantic face at this dimension.
        var faceArray = faces.ToArray();
        var dim = faceArray.Length > 0 ? faceArray[0].Length : 0;
        var semanticStart = GenesisNova.Core.FaceLayout.WordFaceStart(dim);
        var rangeStart = (semanticStart > 0 && semanticStart < dim) ? semanticStart : 0;
        _semanticTree = new VPTree(names.ToArray(), faceArray, rangeStart: rangeStart, rangeEnd: dim);
        _lastSemanticCount = names.Count;
        _semanticDirty = false;
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
