using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Core;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// The ground-up dialectical platonic core (PLATONIC_THEORY.md §9-§11). A clean implementation of
/// <see cref="IPlatonicSpace"/> built from the ontology Π = (E, κind, π, ¬, κ, ▷): elements are born with a
/// NEUTRAL (spawn-spread) semantic orbital and their position SETTLES from per-aspect contradiction (Law D) — no
/// <c>AddWordIdentity</c> stamp, none of the accreted force constants. Identity is delegated VERBATIM to the kept
/// codec via <see cref="FaceCodec"/>, so the number homomorphism stays bit-exact. Composition (▷-hubs) and the
/// recognition hierarchy arrive in later milestones; M0 makes the core swappable and arithmetic-correct behind the
/// contract, with born-neutral per-aspect positioning already live so retrieval works.
///
/// Selected against the legacy <see cref="PlatonicSpaceMemory"/> by a NovaConfig switch; same contract, so the GRU /
/// inference / trainer are unchanged.
/// </summary>
public sealed class DialecticalSpace : IPlatonicSpace
{
    private readonly int _dim;
    private readonly int _semStart;
    private readonly int _semLen;
    private readonly ElementStore _concepts = new();
    private readonly Dictionary<string, Relation> _relations = new(StringComparer.Ordinal);
    private readonly Dictionary<string, HashSet<string>> _adjacency = new(StringComparer.Ordinal);
    private readonly HashSet<string> _operationTokens = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, int>> _chunks = new(StringComparer.Ordinal);
    private readonly FunctionElementRegistry _functions;

    // SPATIAL INDEX (ported discipline from PlatonicSpaceMemory). The new core was built on full O(N) linear scans
    // with a FullFace recompute per concept — fine while tiny, but every retrieval/perception/Reason call is O(N),
    // so a growing gym drifts toward O(N^2). The VP-tree (semantic face KNN) gives O(log N) candidate gathering.
    // HYBRID: below LatticeMinNodes the exact scan is cheap AND always fresh, so it stays the path (every existing
    // test + the keep-core proof are byte-identical); at/above it the lattice harvests a bounded candidate SET and
    // we ALWAYS re-score those candidates' LIVE faces (never trust the throttled tree's distances). So the lattice
    // only ever bounds *which* concepts we score, never *how* — correctness is identical to the scan, just faster.
    private readonly PlatonicLattice _lattice;
    private const int LatticeMinNodes = 384;

    // Distributional meaning needs NO force constants — the old per-aspect dialectic + hinge repulsion are gone. A
    // concept's cloud is simply the superposition of its relational context (AccumulateContext), normalized on read;
    // related cluster and unrelated go orthogonal for free. The one guard below bounds the task-gradient/imprint paths.
    private const double MaxOrbitalNorm = 3.0;

    public DialecticalSpace(int faceDimension, int seed = 42)
    {
        _dim = Math.Max(4, faceDimension);
        _semStart = FaceCodec.SemanticStart(_dim);
        _semLen = FaceCodec.SemanticLength(_dim);
        _functions = new FunctionElementRegistry(_dim);
        // Index the ACTIVE non-atom concepts by their FULL face; the lattice slices the semantic region
        // [WordFaceStart..dim) internally — the SAME offset FaceCodec.SemanticStart (and CloudOf) use, so the
        // VP-tree's neighbourhood and the relaxation's ranking live in one subspace.
        _lattice = new PlatonicLattice(
            nodeNames: () => _concepts.All.Where(e => !e.Archived && e.Kind != ElementKind.Atom).Select(e => e.Symbol),
            nodeFaces: () => _concepts.All.Where(e => !e.Archived && e.Kind != ElementKind.Atom)
                                          .Select(e => (e.Symbol, FullFace(e.Symbol, e))));
    }

    private sealed class Relation
    {
        public required string Left;
        public required string Right;
        public double Thesis;
        public double LastObserved;
        public double Synthesis;
        public int ObservationCount;
        public int UseCount;
        public int SuccessCount;
        public int FailureCount;
        public long LastUsedStep;
        public double Strength => Math.Clamp(1.0 - Synthesis, 0.0, 1.0);
    }

    private static string Normalize(string v) => v.Trim().ToLowerInvariant();
    private static string Key(string a, string b) => string.CompareOrdinal(a, b) <= 0 ? a + "" + b : b + "" + a;
    private static double Clamp01(double x) => x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;

    // ─────────────────────────────────────────────────────────────────────────────── Core (counts / config / glue)
    public bool UseInfoNceRepulsion { get; set; }
    public bool DimensionalContradiction { get; set; } = true; // the new core is per-aspect by construction
    // Concept counts EXCLUDE atoms — atoms are reusable sub-lexical components, not user concepts (keeps NodeCount's
    // meaning identical to the legacy store while the atom layer lives underneath).
    public int NodeCount => _concepts.All.Count(e => !e.Archived && e.Kind != ElementKind.Atom);
    public int RelationCount => _relations.Count;
    public int ArchivedNodeCount => _concepts.TotalCount - _concepts.ActiveCount;
    public int ArchivedRelationCount => 0;
    public int FaceDimension => _dim;
    public int NumericDimensions => Math.Min(_dim / 2, 21);
    public bool ContainsConcept(string concept)
        => _concepts.TryGet(Normalize(concept), out var e) && !e.Archived && e.Kind != ElementKind.Atom;
    public void RegisterOperationToken(string token) { if (!string.IsNullOrWhiteSpace(token)) _operationTokens.Add(Normalize(token)); }
    public bool IsOperationToken(string concept) => _operationTokens.Count > 0 && _operationTokens.Contains(Normalize(concept));

    // ─────────────────────────────────────────────────────────────────────────────── Faces (synthesize / π)
    private Element GetOrCreateConcept(string symbol)
    {
        var key = Normalize(symbol);
        // Idempotent + G6 REACTIVATION: re-observing an archived (ablated) concept brings it back to life — the very
        // same element, learned orbital intact. This is what lets a self regenerate its body from conserved memory.
        if (_concepts.TryGet(key, out var existing))
        {
            if (existing.Archived) { existing.Archived = false; _lattice.RegisterNode(key); } // G6 reactivation → back in the index
            return existing;
        }

        // ▷ COMPOSITION (Laws C/S): a concept is a HUB over REUSED components. Numbers ride the homomorphism (digit
        // places implicit). Multi-word text composes WORD elements; a single word composes CHAR atoms — the same φ
        // the kept codec already uses for the spelling face, now made explicit and reusable so growth is bounded
        // (a novel word over existing chars adds ~0 atoms; a novel sentence over existing words adds ~0 words).
        int[] components;
        ElementKind kind;
        if (FaceCodec.IsNumeric(key))
        {
            components = Array.Empty<int>();
            kind = ElementKind.Object;
        }
        else if (HasWhitespace(key))
        {
            components = key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
                .Select(w => GetOrCreateConcept(w).Id).ToArray();
            kind = ElementKind.Composition;
        }
        else
        {
            components = key.Select(c => GetOrCreateAtom(c).Id).ToArray();
            kind = ElementKind.Composition;
        }
        // Born as its OWN token: a concept with no relations means just itself; meaning EMERGES as it accumulates
        // context (so even a single direct relation big↔large makes them cluster — they share each other's token).
        var created = _concepts.GetOrCreate(key, kind, () => FaceCodec.Token(key, _dim), components);
        _lattice.RegisterNode(key); // a new non-atom concept entered the active set
        return created;
    }

    /// <summary>A reusable CHAR ATOM (Law S): one element per character, referenced (▷) by every word that uses it,
    /// so the atom set stays bounded (~the alphabet) no matter how large the vocabulary grows.</summary>
    private Element GetOrCreateAtom(char c)
        => _concepts.GetOrCreate("atom:" + c, ElementKind.Atom, () => new double[_semLen]);

    private static bool HasWhitespace(string s) { foreach (var c in s) if (char.IsWhiteSpace(c)) return true; return false; }

    /// <summary>Active CHAR/component atoms — bounded by the alphabet (Law S scaling witness).</summary>
    public int AtomCount => _concepts.All.Count(e => !e.Archived && e.Kind == ElementKind.Atom);
    /// <summary>Active composite hubs (words/text) — these grow with vocabulary; the atoms under them are reused.</summary>
    public int CompositeCount => _concepts.All.Count(e => !e.Archived && e.Kind == ElementKind.Composition);
    /// <summary>The ▷ component symbols of a concept (chars for a word, words for a text) — the reuse it shares.</summary>
    public IReadOnlyList<string> ComponentSymbolsOf(string symbol)
        => _concepts.TryGet(Normalize(symbol), out var e)
            ? e.Components.Select(id => _concepts.ById(id).Symbol).ToArray()
            : Array.Empty<string>();

    /// <summary>Assemble the full positive face π(e): numbers → exact homomorphism; text → identity ⊕ cloud. The
    /// large-face cloud is NORMALIZED to unit here, so the running distributional accumulation lives on a sphere and
    /// distances read as ≈cosine (the meaning is in the DIRECTION, not the magnitude).</summary>
    private double[] FullFace(string symbol, Element? element)
    {
        // A STORED concept carries its accumulated cloud. A number with no element (a synthesized arithmetic operand
        // like 84) gets a ZERO cloud — arithmetic-exact and semantically neutral; text with no element gets its own
        // token (its bare identity, before any context).
        var orbital = element?.SemanticFace
            ?? (FaceCodec.IsNumeric(symbol) ? new double[_semLen] : FaceCodec.Token(symbol, _dim));
        var face = FaceCodec.AssemblePositiveFace(symbol, orbital, _dim);
        NormalizeSemantic(face);
        return face;
    }

    /// <summary>Normalize the large-face region [_semStart, dim) of an assembled face to unit (leaves the frozen
    /// arithmetic/identity faces untouched).</summary>
    private void NormalizeSemantic(double[] face)
    {
        var sum = 0.0;
        for (var i = _semStart; i < _dim && i < face.Length; i++) sum += face[i] * face[i];
        var norm = Math.Sqrt(sum);
        if (norm <= 1e-9) return;
        var inv = 1.0 / norm;
        for (var i = _semStart; i < _dim && i < face.Length; i++) face[i] *= inv;
    }

    public bool TryGetConceptFace(string concept, out double[] positiveFace)
    {
        var key = Normalize(concept);
        if (FaceCodec.IsNumeric(key))
        {
            positiveFace = FullFace(key, null); // numbers always available (homomorphic), exact
            return true;
        }
        if (_concepts.TryGet(key, out var e) && !e.Archived)
        {
            positiveFace = FullFace(key, e);
            return true;
        }
        positiveFace = Array.Empty<double>();
        return false;
    }

    // FACE-AWARE distance: relatedness on the SEMANTIC face [_semStart, dim); for a NUMERIC query also blend the
    // arithmetic face and take the min (so numbers surface value-near neighbours while text stays shielded).
    private double FaceAwareDistance(IReadOnlyList<double> q, IReadOnlyList<double> c)
    {
        var semantic = RangeDistance(q, c, _semStart, _dim);
        var arithEnd = Math.Min(2 * NumericDimensions, _dim);
        var arithNorm = 0.0;
        for (var i = 0; i < arithEnd && i < q.Count; i++) arithNorm += q[i] * q[i];
        if (arithNorm <= 0.01) return semantic;
        return Math.Min(semantic, RangeDistance(q, c, 0, arithEnd));
    }

    private static double RangeDistance(IReadOnlyList<double> a, IReadOnlyList<double> b, int start, int end)
    {
        var n = Math.Min(Math.Min(a.Count, b.Count), end);
        var s = Math.Max(0, start);
        var sum = 0.0;
        for (var i = s; i < n; i++) { var d = a[i] - b[i]; sum += d * d; }
        return Math.Sqrt(sum);
    }

    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConcepts(
        string concept, IReadOnlyCollection<string>? candidates = null, int maxNeighbors = 8, int maxCandidates = 96)
    {
        var limit = Math.Clamp(maxNeighbors, 1, 32);
        if (!TryGetConceptFace(concept, out var q) || q.Length == 0)
            return Array.Empty<(string, double)>();
        var self = Normalize(concept);

        // Candidate POOL: explicit candidates → those; else at scale the VP-tree harvests a bounded near set in
        // O(log N); else (small space) the exact scan. Every branch then LIVE-rescores below, so the lattice only
        // bounds which concepts we measure — distances are always exact (never the tree's stale ones).
        IEnumerable<string> pool;
        if (candidates is { Count: > 0 })
            pool = candidates.Select(Normalize).Where(c => _concepts.Contains(c)).Take(Math.Clamp(maxCandidates, 8, 512));
        else if (_concepts.ActiveCount >= LatticeMinNodes)
            pool = _lattice.GetSemanticNeighbors(q, Math.Clamp(limit * 4, 16, 256), self).Select(n => n.Name);
        else
            pool = _concepts.All.Where(e => !e.Archived && e.Kind != ElementKind.Atom).Select(e => e.Symbol);

        var scored = new List<(string, double)>();
        foreach (var cand in pool)
        {
            if (cand.Equals(self, StringComparison.Ordinal)) continue;
            if (!_concepts.TryGet(cand, out var ce) || ce.Archived || ce.Kind == ElementKind.Atom) continue;
            scored.Add((cand, FaceAwareDistance(q, FullFace(cand, ce))));
        }
        return scored.OrderBy(x => x.Item2).Take(limit).ToArray();
    }

    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConceptsFresh(
        string concept, IReadOnlyCollection<string>? seeds = null, int maxNeighbors = 8)
    {
        // Small space → the exact scan is already fresh. At scale → the FRESH discipline (ported from the legacy
        // store): harvest candidate NAMES from always-current sources (caller seeds + this concept's live adjacency +
        // an inflated VP-tree pool, distances discarded), then re-score them against LIVE faces. So a concept that
        // moved THIS step is measured at its current position even before the throttled tree rebuilds.
        if (_concepts.ActiveCount < LatticeMinNodes)
            return GetNearestConcepts(concept, seeds, maxNeighbors);
        if (!TryGetConceptFace(concept, out var q) || q.Length == 0)
            return Array.Empty<(string, double)>();
        var self = Normalize(concept);
        var names = new HashSet<string>(StringComparer.Ordinal);
        if (seeds is not null)
            foreach (var s in seeds) { var n = Normalize(s); if (!n.Equals(self, StringComparison.Ordinal)) names.Add(n); }
        if (_adjacency.TryGetValue(self, out var adj))
            foreach (var n in adj) names.Add(n);
        foreach (var (n, _) in _lattice.GetSemanticNeighbors(q, Math.Clamp(maxNeighbors * 3, 8, 64), self))
            names.Add(n);
        names.Remove(self);
        return names.Count == 0 ? Array.Empty<(string, double)>() : GetNearestConcepts(concept, names, maxNeighbors);
    }

    public double[] ComputeRoutePerception(string anchor, double transformReliability = 0.0)
    {
        var near = GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 4);
        var hasNeighbour = near.Count > 0 ? 1.0 : 0.0;
        var nearestConf = near.Count > 0 ? 1.0 / (1.0 + Math.Max(0.0, near[0].Distance)) : 0.0;
        var meanTopConf = near.Count > 0 ? near.Average(n => 1.0 / (1.0 + Math.Max(0.0, n.Distance))) : 0.0;
        var degreeNorm = Math.Clamp(GetRelationDegree(Normalize(anchor)) / 8.0, 0.0, 1.0);
        return new[] { hasNeighbour, nearestConf, degreeNorm, meanTopConf, Math.Clamp(transformReliability, 0.0, 1.0), 1.0 };
    }

    public double TotalCharge() => 0.0; // ¬e = −e exactly (G4) ⇒ pairwise cancellation

    // ─────────────────────────────────────────────────────────────────────────── Observe / position (κ → π, Law D)
    public void ObserveContradiction(string left, string right, double observedContradiction)
    {
        var a = Normalize(left);
        var b = Normalize(right);
        if (a == b) return;
        // Hard rule: numbers NEVER form relation edges (they pollute and erase prior lessons).
        if (FaceCodec.IsNumeric(a) && FaceCodec.IsNumeric(b)) return;

        var ea = GetOrCreateConcept(a);
        var eb = GetOrCreateConcept(b);
        ea.ObservationCount++; eb.ObservationCount++;

        var key = Key(a, b);
        if (!_relations.TryGetValue(key, out var rel))
        {
            rel = new Relation { Left = a, Right = b, Thesis = observedContradiction,
                LastObserved = observedContradiction, Synthesis = observedContradiction };
            _relations[key] = rel;
            AddAdjacency(a, b);
        }
        rel.LastObserved = Clamp01(observedContradiction);
        rel.Synthesis = rel.ObservationCount == 0 ? rel.LastObserved : 0.85 * rel.Synthesis + 0.15 * rel.LastObserved;
        rel.ObservationCount++;

        // DISTRIBUTIONAL MEANING (PLATONIC_NUCLEUS.md; validated in LargeFaceMeaningTests). A concept's large-face
        // cloud = token(self) + Σ over its relational context of affinity·token(neighbour) (agree adds, contradict
        // subtracts) — computed PRESENCE-based (each neighbour once, NOT per-observation, so repeated training never
        // drowns the self-token). Related concepts share context → clouds overlap; unrelated go orthogonal, no
        // repulsion tuning. Both endpoints' neighbour sets just changed, so recompute both.
        RecomputeCloud(ea);
        RecomputeCloud(eb);
    }

    /// <summary>
    /// Recompute a concept's large-face cloud as the PRESENCE-based superposition of its relational context:
    /// token(self) + Σ_{neighbour n} (1−2κ)·token(n). Each neighbour contributes ONCE (not per-observation), so the
    /// self-token keeps weight 1 and repeated training cannot drown it — the fix for directly-related-but-context-poor
    /// pairs (big↔large) going orthogonal. Agree (low κ) adds the neighbour's token (clouds overlap); contradict (high
    /// κ) subtracts it. Drift-free: always reflects the current relation set. The arithmetic face is untouched.
    /// </summary>
    private void RecomputeCloud(Element e)
    {
        var cloud = FaceCodec.Token(e.Symbol, _dim); // self-token, weight 1
        if (_adjacency.TryGetValue(e.Symbol, out var nbrs))
        {
            foreach (var n in nbrs)
            {
                var aff = 1.0 - 2.0 * Clamp01(GetContradiction(e.Symbol, n));
                if (Math.Abs(aff) < 1e-9) continue;
                var t = FaceCodec.Token(n, _dim);
                for (var i = 0; i < _semLen; i++) cloud[i] += aff * t[i];
            }
        }
        var dst = e.SemanticFace;
        for (var i = 0; i < _semLen && i < cloud.Length && i < dst.Length; i++) dst[i] = cloud[i];
        _lattice.MarkEmbeddingsDirty(); // the semantic face moved → drift toward a VP-tree rebuild
    }

    private static void ClampNorm(double[] v)
    {
        var sum = 0.0;
        for (var i = 0; i < v.Length; i++) sum += v[i] * v[i];
        var norm = Math.Sqrt(sum);
        if (norm <= MaxOrbitalNorm || norm <= 1e-12) return;
        var inv = MaxOrbitalNorm / norm;
        for (var i = 0; i < v.Length; i++) v[i] *= inv;
    }

    private void AddAdjacency(string a, string b)
    {
        if (!_adjacency.TryGetValue(a, out var sa)) { sa = new HashSet<string>(StringComparer.Ordinal); _adjacency[a] = sa; }
        if (!_adjacency.TryGetValue(b, out var sb)) { sb = new HashSet<string>(StringComparer.Ordinal); _adjacency[b] = sb; }
        sa.Add(b); sb.Add(a);
    }

    public double GetContradiction(string left, string right)
        => _relations.TryGetValue(Key(Normalize(left), Normalize(right)), out var r) ? r.Synthesis : 0.5;

    public int GetRelationDegree(string concept)
        => _adjacency.TryGetValue(Normalize(concept), out var s) ? s.Count : 0;

    /// <summary>The body's living concepts (non-atom, non-archived) — what a self can SENSE of its own body.</summary>
    public IReadOnlyList<string> ActiveConcepts
        => _concepts.All.Where(e => !e.Archived && e.Kind != ElementKind.Atom).Select(e => e.Symbol).ToArray();

    public IReadOnlyList<(string Left, string Right, long ObservationCount)> GetAllRelations()
        => _relations.Values.Select(r => (r.Left, r.Right, (long)r.ObservationCount)).ToArray();

    public void FineEditFromExample(IReadOnlyList<string> inputConcepts, IReadOnlyList<string> outputConcepts, bool isNegativeExample)
    {
        var inputs = (inputConcepts ?? Array.Empty<string>()).Select(Normalize).Where(c => c.Length > 0).Distinct().ToArray();
        var outputs = (outputConcepts ?? Array.Empty<string>()).Select(Normalize).Where(c => c.Length > 0).Distinct().ToArray();
        // An example asserts input↔output relatedness (or, if negative, opposition). Reuse the observe path.
        var k = isNegativeExample ? 0.9 : 0.1;
        foreach (var i in inputs)
            foreach (var o in outputs)
                if (i != o) ObserveContradiction(i, o, k);
    }

    public void DisruptAssociation(string anchor, string answer)
    {
        var a = Normalize(anchor); var b = Normalize(answer);
        if (a == b || FaceCodec.IsNumeric(a) || FaceCodec.IsNumeric(b)) return;
        if (!_concepts.TryGet(a, out var ea) || !_concepts.TryGet(b, out var eb)) return;
        if (_relations.TryGetValue(Key(a, b), out var rel))
        {
            rel.FailureCount++; rel.Synthesis = Clamp01(rel.Synthesis + 0.08); rel.LastObserved = rel.Synthesis;
        }
        RecomputeCloud(ea); RecomputeCloud(eb); // the raised contradiction now subtracts the partner's token from the cloud
    }

    public void FunctionGradientStep(string anchor, string target, IReadOnlyList<string> distractors, double rate = 0.05)
    {
        var qn = Normalize(anchor); var tn = Normalize(target);
        if (qn == tn || FaceCodec.IsNumeric(tn)) return;
        if (!TryGetConceptFace(qn, out var q) || !_concepts.TryGet(tn, out var tNode)) return;

        var cands = new List<Element> { tNode };
        var seen = new HashSet<string>(StringComparer.Ordinal) { qn, tn };
        foreach (var d in distractors ?? Array.Empty<string>())
        {
            var dn = Normalize(d);
            if (!seen.Add(dn) || FaceCodec.IsNumeric(dn)) continue;
            if (_concepts.TryGet(dn, out var de)) cands.Add(de);
        }
        if (cands.Count < 2) return;

        // softmax-CE over dot products on the full faces; gradient dL/dc_i = (p_i − 1{target})·q, applied to orbitals.
        var faces = cands.Select(c => FullFace(c.Symbol, c)).ToArray();
        var p = new double[cands.Count];
        var maxS = double.NegativeInfinity;
        for (var i = 0; i < cands.Count; i++)
        {
            var s = 0.0;
            for (var d = 0; d < _dim && d < q.Length; d++) s += q[d] * faces[i][d];
            p[i] = s; if (s > maxS) maxS = s;
        }
        var sum = 0.0;
        for (var i = 0; i < cands.Count; i++) { p[i] = Math.Exp(p[i] - maxS); sum += p[i]; }
        for (var i = 0; i < cands.Count; i++) p[i] /= sum;

        for (var i = 0; i < cands.Count; i++)
        {
            var coeff = rate * (p[i] - (i == 0 ? 1.0 : 0.0));
            var orbital = cands[i].SemanticFace;
            for (var j = 0; j < _semLen; j++)
                orbital[j] -= coeff * q[_semStart + j];
            ClampNorm(orbital);
        }
        _lattice.MarkEmbeddingsDirty();
    }

    public void ReinforceEvidence(IReadOnlyList<PlatonicEvidence> evidence, bool success)
    {
        if (evidence == null) return;
        foreach (var ev in evidence)
        {
            if (ev.RelatedConcept == null) continue;
            var key = Key(Normalize(ev.Concept), Normalize(ev.RelatedConcept));
            if (!_relations.TryGetValue(key, out var rel)) continue;
            rel.UseCount++;
            if (success) { rel.SuccessCount++; rel.Synthesis = Clamp01(rel.Synthesis - 0.05); }
            else { rel.FailureCount++; rel.Synthesis = Clamp01(rel.Synthesis + 0.05); }
            rel.LastObserved = rel.Synthesis;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────── Reasoning by relaxation (PLATONIC_MIND)
    /// <summary>The outcome of a relaxation: the basin the field settled into, its confidence, whether it SETTLED at
    /// all (a near basin existed — else the field abstains rather than invents), and how many steps it took.</summary>
    public readonly record struct Thought(string Symbol, double Confidence, bool Settled, int Steps);

    /// <summary>
    /// REASON BY RELAXATION (PLATONIC_MIND.md; validated in FieldRelaxationTests). Build a query cloud from the
    /// anchors' tokens, then relax it over the stored concept clouds via a modern-Hopfield / attention update
    /// <c>x ← normalize(Σ softmax(β⟨x,μ⟩)·μ)</c> — the field flowing down its own free-energy gradient. The settled
    /// basin is the answer; if no basin is near the raw query (initial max similarity below <paramref name="settleThreshold"/>)
    /// the field is "surprised" and ABSTAINS instead of inventing. The anchors themselves are excluded from the
    /// basins (you retrieve a DIFFERENT concept that fits the context — recall / disambiguation / categorisation).
    /// </summary>
    public Thought Reason(IReadOnlyList<string> anchors, int maxCandidates = 64, double beta = 12.0, int steps = 8, double settleThreshold = 0.3,
        double[]? selfContext = null, double selfWeight = 0.0)
    {
        var anchorSet = new HashSet<string>(StringComparer.Ordinal);
        var q = new double[_semLen];
        foreach (var a in anchors ?? Array.Empty<string>())
        {
            var key = Normalize(a);
            anchorSet.Add(key);
            if (IsOperationToken(key) || FaceCodec.IsNumeric(key)) continue;
            // Relax from the anchor's learned MEANING (its cloud — token(self) + its relational context), not its
            // bare symbol: a member's cloud already points at its hub, so a large category hub no longer dilutes the
            // content signal below the settle threshold. Unknown anchors (no element) fall back to the raw token.
            if (_concepts.TryGet(key, out var ae) && !ae.Archived && ae.Kind != ElementKind.Atom)
            {
                var cloud = CloudOf(ae);
                for (var i = 0; i < _semLen; i++) q[i] += cloud[i];
            }
            else
            {
                var t = FaceCodec.Token(key, _dim);
                for (var i = 0; i < _semLen; i++) q[i] += t[i];
            }
        }
        // The reasoning is conditioned by the MIND'S SELF — a persistent meaning-space vector of what it has been
        // living (the engine accumulates it). It tilts the query toward the self's standing context so a genuinely
        // ambiguous anchor settles into the basin consistent with WHO THE MIND IS, while the anchor itself (weight 1)
        // keeps it on-topic. Zero weight (or no self) = the pure, self-free reasoning — the ablation.
        if (selfContext != null && selfWeight > 0.0 && selfContext.Length == _semLen)
            for (var i = 0; i < _semLen; i++) q[i] += selfWeight * selfContext[i];
        if (!NormalizeVec(q)) return new Thought(string.Empty, 0.0, false, 0);

        var cands = new List<(string Sym, double[] Cloud)>();
        void Consider(Element e)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || FaceCodec.IsNumeric(e.Symbol)) return;
            if (IsReservedConcept(e.Symbol) || IsOperationToken(e.Symbol) || anchorSet.Contains(e.Symbol)) return;
            cands.Add((e.Symbol, CloudOf(e)));
        }
        if (_concepts.ActiveCount >= LatticeMinNodes)
        {
            // At scale: harvest the query cloud's neighbourhood from the VP-tree (O(log N)), then build LIVE clouds
            // for those and relax. q is the normalized semantic query; embed it into a full face so the lattice
            // compares it on [WordFaceStart..dim) — exactly the region CloudOf/Dot rank on (so the harvested set is
            // the same one the full scan's top-Dot would pick). Distances from the tree are discarded; we re-rank below.
            var qFull = new double[_dim];
            for (var i = 0; i < _semLen && _semStart + i < _dim; i++) qFull[_semStart + i] = q[i];
            foreach (var (sym, _) in _lattice.GetSemanticNeighbors(qFull, Math.Clamp(maxCandidates, 16, 256), null))
                if (_concepts.TryGet(sym, out var e)) Consider(e);
        }
        else
        {
            foreach (var e in _concepts.All) Consider(e);
        }
        if (cands.Count == 0) return new Thought(string.Empty, 0.0, false, 0);

        var ranked = cands.OrderByDescending(c => Dot(q, c.Cloud)).Take(Math.Clamp(maxCandidates, 4, 256)).ToList();
        var initMax = Dot(q, ranked[0].Cloud);
        var settled = initMax >= settleThreshold;

        var x = (double[])q.Clone();
        for (var s = 0; s < steps; s++)
        {
            var sims = new double[ranked.Count];
            var mx = double.NegativeInfinity;
            for (var i = 0; i < ranked.Count; i++) { sims[i] = Dot(x, ranked[i].Cloud); if (sims[i] > mx) mx = sims[i]; }
            var z = 0.0;
            for (var i = 0; i < ranked.Count; i++) { sims[i] = Math.Exp(beta * (sims[i] - mx)); z += sims[i]; }
            var nx = new double[_semLen];
            for (var i = 0; i < ranked.Count; i++) { var wi = sims[i] / z; var cl = ranked[i].Cloud; for (var d = 0; d < _semLen; d++) nx[d] += wi * cl[d]; }
            if (!NormalizeVec(nx)) break;
            x = nx;
        }
        // Select the basin nearest the SETTLED query — blend the relaxed state with the original query so the answer
        // stays anchored to what was asked (pure relaxation can drift into a denser, unrelated basin; pure query
        // ignores the disambiguating context). The query keeps it on-topic; the relaxation lets context tip a tie.
        var best = 0; var bestSim = double.NegativeInfinity;
        for (var i = 0; i < ranked.Count; i++)
        {
            var s = 0.7 * Dot(q, ranked[i].Cloud) + 0.3 * Dot(x, ranked[i].Cloud);
            if (s > bestSim) { bestSim = s; best = i; }
        }
        return new Thought(ranked[best].Sym, Dot(q, ranked[best].Cloud), settled, steps);
    }

    // Reserved internal symbols (the codec's face anchors and any "∴"-prefixed reflexive elements) — observed by the
    // mind, never retrieved as an answer, so they never pollute ordinary retrieval.
    private static bool IsReservedConcept(string s)
        => s.StartsWith("face:", StringComparison.Ordinal) || s.StartsWith("∴", StringComparison.Ordinal);
    private double[] CloudOf(Element e)
    {
        var face = FullFace(e.Symbol, e); // semantic region already normalized by FullFace
        var c = new double[_semLen];
        for (var i = 0; i < _semLen && _semStart + i < face.Length; i++) c[i] = face[_semStart + i];
        return c;
    }

    /// <summary>The concept's MEANING as a semantic-face vector (its cloud — the same representation <see cref="Reason"/>
    /// relaxes over). Null for an unknown / numeric / operator token: there is nothing the mind can hold a persistent
    /// sense OF. Public so the MIND (the inference engine) can accumulate a persistent self from what it attends to.</summary>
    public double[]? SemanticVectorOf(string concept)
    {
        var key = Normalize(concept);
        if (IsOperationToken(key) || FaceCodec.IsNumeric(key)) return null;
        return _concepts.TryGet(key, out var e) && !e.Archived && e.Kind != ElementKind.Atom ? CloudOf(e) : null;
    }

    /// <summary>The width of a semantic-face vector (<see cref="SemanticVectorOf"/> / the self-field) — so the mind can
    /// allocate its persistent self in the same space the field reasons in.</summary>
    public int SemanticLength => _semLen;
    private double Dot(double[] a, double[] b) { var d = 0.0; for (var i = 0; i < _semLen; i++) d += a[i] * b[i]; return d; }
    private bool NormalizeVec(double[] v) { var s = 0.0; for (var i = 0; i < _semLen; i++) s += v[i] * v[i]; s = Math.Sqrt(s); if (s <= 1e-9) return false; for (var i = 0; i < _semLen; i++) v[i] /= s; return true; }

    // ─────────────────────────────────────────────────────────────────────────────── Recognition hierarchy (M3)
    /// <summary>The outcome of <see cref="Recognize"/>: the recognized/composed symbol, a confidence, whether it was
    /// a WHOLE hit (remembered directly) vs composed fresh, and how much of it was built from KNOWN components.</summary>
    public readonly record struct Recognition(string Symbol, double Confidence, bool WholeHit, int KnownComponents, int TotalComponents);

    /// <summary>
    /// Recognize-highest-first (PLATONIC_THEORY.md §6 recognize; the user's "hit word level first"). (1) Match the
    /// WHOLE input as an existing composite hub (or a number via the homomorphism) — the fast "remember". (2) On a
    /// miss, DECOMPOSE to components (text→words, word→chars) and measure how many are already known. (3) COMPOSE a
    /// new hub from those components and STORE it (G3 generative) — so the same input is a whole hit next time. This
    /// is the accumulating, scaling path: novelty is absorbed by recombining reused parts, never by re-stamping.
    /// </summary>
    public Recognition Recognize(string input)
    {
        var key = Normalize(input);
        if (FaceCodec.IsNumeric(key))
            return new Recognition(key, 1.0, WholeHit: true, 0, 0);                 // recognized via the homomorphism
        if (_concepts.TryGet(key, out var existing) && !existing.Archived && existing.Kind != ElementKind.Atom)
            return new Recognition(key, 1.0, WholeHit: true, existing.Components.Length, existing.Components.Length); // remembered whole

        var (known, total) = ComponentCoverage(key);
        GetOrCreateConcept(key);   // compose-and-store: builds the hub over its components (and any novel parts)
        var conf = total > 0 ? (double)known / total : 0.5;
        return new Recognition(key, conf, WholeHit: false, known, total);
    }

    /// <summary>How many of an input's ▷ components are already KNOWN (text→words, word→char atoms).</summary>
    private (int Known, int Total) ComponentCoverage(string key)
    {
        if (HasWhitespace(key))
        {
            var words = key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            return (words.Count(w => _concepts.Contains(w)), words.Length);
        }
        return (key.Count(c => _concepts.Contains("atom:" + c)), key.Length);
    }

    // ─────────────────────────────────────────────────────────────────────────────── Recognize / query (contract)
    public IReadOnlyList<PlatonicNeighbor> GetNeighbors(
        string concept, PlatonicNeighborhoodType type = PlatonicNeighborhoodType.Any, int maxNeighbors = 16, double minConfidence = 0.0)
    {
        var self = Normalize(concept);
        var merged = new Dictionary<string, PlatonicNeighbor>(StringComparer.Ordinal);
        void Consider(PlatonicNeighbor n)
        {
            if (n.Concept.Equals(self, StringComparison.Ordinal)) return;
            if (n.Confidence < minConfidence) return;
            if (!merged.TryGetValue(n.Concept, out var ex) || n.Confidence > ex.Confidence) merged[n.Concept] = n;
        }

        if (type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Relational && _adjacency.TryGetValue(self, out var adj))
            foreach (var nb in adj)
            {
                var rel = _relations.TryGetValue(Key(self, nb), out var r) ? r : null;
                Consider(new PlatonicNeighbor(nb, rel?.Strength ?? 0.5, PlatonicNeighborhoodType.Relational, rel?.ObservationCount ?? 0));
            }

        if (type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Semantic)
            foreach (var (sym, dist) in GetNearestConcepts(self, null, maxNeighbors))
                Consider(new PlatonicNeighbor(sym, 1.0 / (1.0 + Math.Max(0.0, dist)), PlatonicNeighborhoodType.Semantic, 0));

        return merged.Values.OrderByDescending(n => n.Confidence).Take(Math.Clamp(maxNeighbors, 1, 64)).ToArray();
    }

    public bool TryRelationElementNeighbour(string concept, out string neighbour, out double strength)
    {
        neighbour = string.Empty; strength = 0.0;
        var self = Normalize(concept);
        if (!_adjacency.TryGetValue(self, out var adj)) return false;
        foreach (var nb in adj)
        {
            var s = _relations.TryGetValue(Key(self, nb), out var r) ? r.Strength : 0.0;
            if (s > strength) { strength = s; neighbour = nb; }
        }
        return neighbour.Length > 0;
    }

    public PlatonicSpaceMemory.PlatonicQueryResult QueryConceptChain(IReadOnlyList<string> anchorConcepts, int maxHops = 2, int beamWidth = 2)
        => QueryConceptChain(anchorConcepts, maxHops, beamWidth, out _);

    public PlatonicSpaceMemory.PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts, int maxHops, int beamWidth, out IReadOnlyList<PlatonicEvidence> evidence)
    {
        var ev = new List<PlatonicEvidence>();
        evidence = ev;
        var anchors = (anchorConcepts ?? Array.Empty<string>()).Select(Normalize)
            .Where(c => c.Length > 0 && !IsOperationToken(c)).ToArray();
        if (anchors.Length == 0)
            return new PlatonicSpaceMemory.PlatonicQueryResult(string.Empty, 0.0, 0, 0);

        // Greedy strongest-edge walk from the best anchor — the relational traversal (concept-chain route).
        var best = string.Empty; var bestConf = 0.0; var hops = 0;
        foreach (var anchor in anchors)
        {
            var current = anchor; var conf = 1.0; var localHops = 0;
            var visited = new HashSet<string>(StringComparer.Ordinal) { current };
            for (var h = 0; h < Math.Max(1, maxHops); h++)
            {
                if (!TryRelationElementNeighbour(current, out var nb, out var strength) || strength <= 0.0) break;
                if (!visited.Add(nb)) break;
                ev.Add(new PlatonicEvidence(current, nb, strength, h + 1));
                current = nb; conf *= strength; localHops++;
            }
            if (localHops > 0 && conf > bestConf && !IsOperationToken(current))
            {
                best = current; bestConf = conf; hops = localHops;
            }
        }
        return new PlatonicSpaceMemory.PlatonicQueryResult(best, bestConf, hops, best.Length > 0 ? 1 : 0);
    }

    public IReadOnlyList<RelationElement> GetRelationElements()
    {
        var result = new List<RelationElement>(_relations.Count);
        foreach (var r in _relations.Values)
        {
            var emb = Array.Empty<double>();
            if (TryGetConceptFace(r.Left, out var lf) && TryGetConceptFace(r.Right, out var rf) && lf.Length == rf.Length && lf.Length > 0)
            {
                emb = new double[lf.Length];
                for (var i = 0; i < lf.Length; i++) emb[i] = 0.5 * (lf[i] + rf[i]);
            }
            result.Add(new RelationElement(r.Left, r.Right, r.Strength, emb));
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────────── Composition / chunks (M1 deepens)
    public void MineChunk(string tag, string chunk)
    {
        if (string.IsNullOrEmpty(tag) || string.IsNullOrEmpty(chunk)) return;
        if (!_chunks.TryGetValue(tag, out var byChunk)) { byChunk = new Dictionary<string, int>(StringComparer.Ordinal); _chunks[tag] = byChunk; }
        byChunk[chunk] = byChunk.TryGetValue(chunk, out var c) ? c + 1 : 1;
    }

    public bool TryGetTopChunk(string tag, out string chunk)
    {
        chunk = string.Empty;
        if (!_chunks.TryGetValue(tag, out var byChunk) || byChunk.Count == 0) return false;
        chunk = byChunk.OrderByDescending(kv => kv.Value).First().Key;
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────────── Function / word elements
    public IReadOnlyList<PlatonicElement> FunctionElements => _functions.Elements;
    public PlatonicElement RegisterFunctionElement(string name, IReadOnlyList<string>? references = null) => _functions.Register(name, references);
    public bool TryGetFunctionElement(string name, out PlatonicElement element) => _functions.TryGet(name, out element);

    // Word elements have NO external production consumer (PLATONIC_THEORY.md §10) — the dialectic does not stamp a
    // per-concept identity, so these are intentionally inert in the new core.
    public IReadOnlyList<PlatonicElement> WordElements => Array.Empty<PlatonicElement>();
    public PlatonicElement RegisterWordElement(string concept)
        => new(Id: 0, Kind: Core.ElementKind.Object, Embedding: PlatonicFaceComposer.GetFreshEmbedding(Normalize(concept), _dim),
               Symbol: Normalize(concept), GeneratedAtTick: 0, RelatedTo: System.Collections.Immutable.ImmutableArray<int>.Empty,
               GenerationPath: "word");
    public bool TryGetWordElement(string concept, out PlatonicElement element) { element = default!; return false; }
    public IReadOnlyList<PlatonicElement> DecomposeWordElement(string concept) => Array.Empty<PlatonicElement>();

    // ─────────────────────────────────────────────────────────────────────────────── Geometry summary (diagnostics)
    public PlatonicSpaceMemory.GeometrySummary SummarizePushPullGeometry(int maxConcepts = 600, int unrelatedPerNode = 8, int seed = 1234)
    {
        var rng = new Random(seed);
        var mutable = _concepts.All.Where(e => !e.Archived && !FaceCodec.IsNumeric(e.Symbol) && e.Kind != ElementKind.Atom).ToArray();
        var sample = mutable.Length <= maxConcepts ? mutable : mutable.OrderBy(_ => rng.Next()).Take(maxConcepts).ToArray();
        var related = new List<double>();
        var unrelated = new List<double>();
        foreach (var node in sample)
        {
            var nf = FullFace(node.Symbol, node);
            if (_adjacency.TryGetValue(node.Symbol, out var adj))
                foreach (var rn in adj)
                    if (_concepts.TryGet(rn, out var other) && !FaceCodec.IsNumeric(rn))
                        related.Add(FaceAwareDistance(nf, FullFace(rn, other)));
            for (var s = 0; s < unrelatedPerNode && mutable.Length > 1; s++)
            {
                var other = mutable[rng.Next(mutable.Length)];
                if (ReferenceEquals(other, node)) continue;
                if (_adjacency.TryGetValue(node.Symbol, out var a) && a.Contains(other.Symbol)) continue;
                unrelated.Add(FaceAwareDistance(nf, FullFace(other.Symbol, other)));
            }
        }
        static (double Mean, double Min, double Max) St(List<double> xs) => xs.Count == 0 ? (0, 0, 0) : (xs.Average(), xs.Min(), xs.Max());
        var r = St(related); var u = St(unrelated);
        return new PlatonicSpaceMemory.GeometrySummary(_concepts.ActiveCount, mutable.Length, related.Count, unrelated.Count,
            r.Mean, r.Min, r.Max, u.Mean, u.Min, u.Max);
    }

    // ─────────────────────────────────────────────────────────────────────────────── Maintenance (G6) — M0 no-op
    public PlatonicSpaceMemory.SpaceMaintenanceResult ApplyMaintenance(PlatonicSpaceMemory.SpaceMaintenanceRequest request)
        => new(0, 0, 0); // archival/eviction not needed for correctness; deepened later

    // ─────────────────────────────────────────────────────────────────────────────── Snapshots (checkpoint compat)
    public PlatonicMemorySnapshot ExportSnapshot()
    {
        var nodes = _concepts.All.Where(e => !FaceCodec.IsNumeric(e.Symbol) && e.Kind != ElementKind.Atom).Select(e =>
        {
            var pf = FullFace(e.Symbol, e);
            return new PlatonicNodeSnapshot(e.Symbol, pf, FaceCodec.Negate(pf), (int)e.ObservationCount);
        }).ToArray();
        var rels = _relations.Values.Select(r => new PlatonicRelationSnapshot(
            r.Left, r.Right, r.Thesis, r.LastObserved, r.Synthesis, r.ObservationCount, r.UseCount, r.SuccessCount, r.FailureCount, r.LastUsedStep)).ToArray();
        var chunks = _chunks.SelectMany(t => t.Value.Select(c => new PlatonicChunkSnapshot(t.Key, c.Key, c.Value))).ToArray();
        return new PlatonicMemorySnapshot(_dim, nodes, rels, chunks, _operationTokens.ToArray());
    }

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        if (snapshot == null) return;
        foreach (var n in snapshot.Nodes ?? Array.Empty<PlatonicNodeSnapshot>())
        {
            if (FaceCodec.IsNumeric(Normalize(n.Name))) continue;
            var e = GetOrCreateConcept(n.Name);
            e.ObservationCount = n.ObservationCount;
            if (n.PositiveFace is { Length: > 0 }) // restore the orbital slice
                for (var i = 0; i < _semLen && _semStart + i < n.PositiveFace.Length; i++)
                    e.SemanticFace[i] = n.PositiveFace[_semStart + i];
        }
        foreach (var r in snapshot.Relations ?? Array.Empty<PlatonicRelationSnapshot>())
        {
            var a = Normalize(r.Left); var b = Normalize(r.Right);
            if (a == b) continue;
            _relations[Key(a, b)] = new Relation { Left = a, Right = b, Thesis = r.ThesisContradiction,
                LastObserved = r.LastObservedContradiction, Synthesis = r.SynthesisContradiction,
                ObservationCount = r.ObservationCount, UseCount = r.UseCount, SuccessCount = r.SuccessCount,
                FailureCount = r.FailureCount, LastUsedStep = r.LastUsedStep };
            AddAdjacency(a, b);
        }
        foreach (var c in snapshot.Chunks ?? Array.Empty<PlatonicChunkSnapshot>())
            for (var i = 0; i < c.Count; i++) MineChunk(c.Tag, c.Chunk);
        foreach (var t in snapshot.OperationTokens ?? Array.Empty<string>())
            RegisterOperationToken(t);
        _lattice.MarkEmbeddingsDirty(); // imported orbitals were written directly → force a rebuild from live faces
    }
}
