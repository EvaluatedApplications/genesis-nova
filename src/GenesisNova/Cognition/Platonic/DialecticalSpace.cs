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
    // Distinct Merge LABELS seen this session (seeded with the fact label "is") — the candidate role/label vocab the
    // address-space structure decoder (PlatonicFaceDecoder.DecodeStructure via TryDecodeCoordinate) resolves a slot's role code against.
    private readonly HashSet<string> _mergeLabels = new(StringComparer.Ordinal) { "is" };

    /// <summary>The LEARNED number-word lexicon (de-hardcoding #5). Lives on the SPACE (not the engine) so it is SHARED
    /// between the training engine and the inference engine and persists with the substrate snapshot. A word→value
    /// annotation, NOT a κ relation edge — numbers-never-edge is respected.</summary>
    public GenesisNova.Cognition.NumberWordLexicon NumberWords { get; } = new();

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

    // RELEVANCE-DECAY MAINTENANCE — a FORGETTING CURVE, not a size cap. The store only ever ADDS, so without decay an
    // unbounded input stream (or a long run minting endless distinct phrases/facts) grows the active set without limit
    // (we measured 182 MB / a pegged core). The fix is NOT an arbitrary ceiling (that churns reinforced structure and
    // broke warming). Instead EVERY element decays: its survival GRACE grows with its UTILITY — chiefly how much it
    // CONTRIBUTED TO CORRECT ANSWERS (relation SuccessCount, credited by ReinforceEvidence), with mere observation as a
    // WEAK floor and failures as a penalty. An element is released only once it has gone STALE beyond the grace its
    // utility earned. So a useful edge/concept is effectively immortal (it keeps earning answers long before the grace
    // expires) while a merely-co-observed one decays out. "Helped answer" outranks "was seen often"; everything decays.
    // The decay holds the relevant, recently-useful structure; ON TOP of it a HARD MaxActiveConcepts cap is the pure-
    // overflow safety net (a corpus stream re-observes its long tail → nothing goes stale → decay alone can't bound it),
    // evicting the lowest-utility excess by the SAME metric. (See DischargeIrrelevant / EnforceActiveConceptCap / RelationUtility.)
    // BASE grace (observation-steps); utility MULTIPLIES it. Sized to comfortably exceed the FocusedCurriculum ROTATION
    // gap — a muscle/the foundation is only re-observed during its focus turn, so the gap between turns is large; if the
    // grace is shorter than that gap, learned vocab decays BETWEEN turns (catastrophic forgetting — an overnight run
    // measured 6000→523 nodes, foundation acc 0.70→0.08 at the old 8_192). A run does millions of observations, so this is
    // deliberately large: reinforced/recurring structure survives the whole run; only genuinely-abandoned hapax decays.
    public long DischargeStalenessWindow { get; set; } = 1_048_576;
    public long DischargeInterval { get; set; } = 4_096;        // sweep once per ~this many observations
    // HARD ACTIVE-CONCEPT CAP (the pure-overflow safety net the grace curve alone cannot provide). Relevance-decay
    // releases STALE noise, but a corpus stream (e.g. a 100% Wikipedia prebake) keeps re-observing the same long-tail
    // vocab, so every word's LastSeenStep stays current → nothing ever goes stale → the active set grows unbounded
    // (the checkpoint bloated to ~2 GB). So ON TOP OF decay we enforce a ceiling by distributional VALUE: when the
    // active concept count exceeds this cap, the LOWEST-UTILITY excess are evicted down to the cap (same success-
    // weighted utility the grace path uses, no word lists). Protected and never counted against / evicted by the cap:
    // ∘-anchors, atoms, Functions, numbers, and ▷-referenced components. 0 disables the cap (unbounded, legacy).
    public int MaxActiveConcepts { get; set; } = 12_000;
    private long _observeStep;                                  // monotone observation clock (drives recency)
    private long _lastDischargeStep;

    // GENERATIVE ATOMS (gated, default OFF = legacy eager char-decomposition, byte-identical). When ON, a token is stored
    // as its OWN biggest-chunk atom (Object) and is DECOMPOSED ON DEMAND (Decompose) into candidate sub-atoms — characters
    // and short n-grams — which then COMPETE via relevance-decay: a sub-atom that recurs across tokens AND contributes to
    // correct answers (a morpheme like "hel") survives, a one-off substring discharges. So useful granularity is DISCOVERED
    // (and is language-agnostic — a CJK char is its own atom, a Latin word breaks down only where it earns its keep), rather
    // than eagerly fixed at single chars. This is the first brick of the genesis create→decompose→select→store loop.
    public bool GenerativeAtoms { get; set; }

    // DECODE-FROM-THE-VOID RECOVERY (gated, default OFF = byte-identical). When ON, an evicted / latent / never-materialised
    // concept is RE-MATERIALISED from its COORDINATE on demand (RecoverFromCoordinate): the frozen identity bands [0,416)
    // are a deterministic invertible codec, so the IDENTITY survives eviction even though the learned orbital + store slot
    // were freed (G6 via the latent address). The materialised space becomes a CACHE over the conserved decodable void —
    // the navigator's walk (TryLand) and reasoning (Reason) decode a missing concept back when they reach its address.
    public bool RecoverFromVoid { get; set; }
    private const double RecoverDecodeConfidence = 0.6;  // a coordinate must decode at least this confidently to count as a
                                                         // VALID identity worth materialising (below = between addresses / noise)
    private const double RecoverRoundTripTolerance = 0.5; // and its recovered canonical face must re-match the input coordinate's
                                                         // FROZEN identity [42,416) within this — the junk-rejection guard

    private const int MaxDecomposeGram = 4;            // longest candidate n-gram (morpheme) generated per token
    private const long DecomposeMinObservations = 3;   // only PROVEN tokens (reinforced) are decomposed — never noise
    private readonly HashSet<string> _decomposed = new(StringComparer.Ordinal); // decompose each token at most once

    // BATCHED-GPU CLOUD RECOMPUTE (gated, default OFF = per-observation scalar RecomputeCloud, byte-identical). When ON,
    // ObserveContradiction does NOT recompute clouds inline; it marks both endpoints DIRTY and defers. Every CloudFlushInterval
    // observations (or before any cloud read) the whole dirty set is recomputed in ONE batched op (GpuCloudBatcher, Cloud = A·T
    // on CUDA). This both dedupes redundant recompute (a hub touched N×/flush → recomputed once) and moves the semLen-wide
    // multiply-accumulate onto the GPU — the substrate's measured bottleneck (638 obs/s scalar). Clouds carry BOUNDED staleness
    // between flushes; read paths flush first (EnsureCloudsFresh), so retrieval always sees current geometry.
    public bool BatchedCloudGpu { get; set; }
    public long CloudFlushInterval { get; set; } = 8_192;       // observations between batched flushes (larger = more dedup)
    private GpuCloudBatcher? _batcher;
    private readonly HashSet<string> _dirtyCloud = new(StringComparer.Ordinal);
    private long _obsSinceFlush;

    public DialecticalSpace(int faceDimension, int seed = 42)
    {
        _dim = Math.Max(4, faceDimension);
        _semStart = FaceCodec.SemanticStart(_dim);
        _semLen = FaceCodec.SemanticLength(_dim);
        _functions = new FunctionElementRegistry(_dim);
        // Index the ACTIVE non-atom concepts by their FULL face; the lattice slices the semantic region
        // [WordFaceStart..dim) internally — the SAME offset FaceCodec.SemanticStart (and CloudOf) use, so the
        // VP-tree's neighbourhood and the relaxation's ranking live in one subspace.
        // EXCLUDE atoms (the reusable base) AND Functions (reserved operation elements — a route, not a semantic
        // target) from the retrieval index, so an op never surfaces as a nearest-concept answer.
        _lattice = new PlatonicLattice(
            nodeNames: () => _concepts.All.Where(IsRetrievable).Select(e => e.Symbol),
            nodeFaces: () => _concepts.All.Where(IsRetrievable)
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
        public long LastSeenStep; // observation recency (drives relevance-decay discharge), distinct from inference LastUsedStep
        public double Strength => Math.Clamp(1.0 - Synthesis, 0.0, 1.0);
    }

    private static string Normalize(string v) => v.Trim().ToLowerInvariant();

    // A retrievable concept: live, and neither an atom (reusable base) nor a Function (reserved operation element — a
    // first-class, decodable route over the address space, never a semantic-retrieval target). One predicate so the
    // lattice index and every concept scan agree on what an op is NOT.
    private static bool IsRetrievable(Element e) => !e.Archived && e.Kind != ElementKind.Atom && e.Kind != ElementKind.Function;

    // RESERVED symbol form for a Function/op element — the "∘fn:" namespace. Distinct from the intent/op anchors
    // (∘add/∘qst…) and Merge label anchors (∘<label>) so an op token (even "add") never collides with one via
    // GetOrCreate idempotency; still ∘-prefixed, so every existing reserved-symbol filter (IsReservedConcept,
    // discharge, Decompose) already covers it and keeps it out of retrieval.
    private const string OpSymbolPrefix = "∘fn:";
    private static string OpSymbol(string opToken) => OpSymbolPrefix + opToken;
    private static string Key(string a, string b) => string.CompareOrdinal(a, b) <= 0 ? a + "" + b : b + "" + a;
    private static double Clamp01(double x) => x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;

    // ─────────────────────────────────────────────────────────────────────────────── Core (counts / config / glue)
    public bool UseInfoNceRepulsion { get; set; }
    // INTENTIONALLY IGNORED in this core: the dialectical core is per-aspect BY CONSTRUCTION — RecomputeCloud superposes
    // neighbour tokens per-dimension (cloud[i] = token(self)[i] + Σ(1−2κ)·token(n)[i]), so shared aspects reinforce and
    // contradicting ones cancel WITHOUT an explicit per-dim κ gate. This flag drives ONLY the LEGACY PlatonicSpaceMemory's
    // MessagePassUpdate gate; it is kept here solely to satisfy IPlatonicSpace and is never read in this class.
    public bool DimensionalContradiction { get; set; } = true;
    // Concept counts EXCLUDE atoms — atoms are reusable sub-lexical components, not user concepts (keeps NodeCount's
    // meaning identical to the legacy store while the atom layer lives underneath).
    public int NodeCount => _concepts.ActiveConceptCount; // O(1), concurrency-safe (read live by /status during training)
    public int RelationCount => _relations.Count;
    public int ArchivedNodeCount => _concepts.TotalCount - _concepts.ActiveCount;
    public int ArchivedRelationCount => 0;
    public int FaceDimension => _dim;
    public int NumericDimensions => Math.Min(_dim / 2, 21);
    public bool ContainsConcept(string concept)
        => _concepts.TryGet(Normalize(concept), out var e) && IsRetrievable(e);
    // G5 — "a Function is itself an element": registering an op both records the cue token AND realises a live, first-
    // class Function element so the operation has a decodable coordinate (kind=Function + op band) like every other band.
    public void RegisterOperationToken(string token)
    {
        if (string.IsNullOrWhiteSpace(token)) return;
        var op = Normalize(token);
        _operationTokens.Add(op);
        GetOrCreateFunction(op);
    }

    /// <summary>
    /// (G5) Realise an OPERATION as a live, first-class <see cref="ElementKind.Function"/> element under its RESERVED
    /// <see cref="OpSymbolPrefix"/> symbol — born with a token orbital like any element, its op-code written into the
    /// frozen op band [400,416) by <see cref="WriteOpBand"/> at face-assembly time. Idempotent (same op ⇒ same element).
    /// NOT registered in the lattice and NOT counted as a concept (it is a route, not a retrieval target); reserved
    /// (∘-prefixed) so it never enters concept retrieval / NodeCount / discharge.
    /// </summary>
    private Element GetOrCreateFunction(string opToken)
        => _concepts.GetOrCreate(OpSymbol(opToken), ElementKind.Function, () => FaceCodec.Token(OpSymbol(opToken), _dim));
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
        else if (GenerativeAtoms)
        {
            components = Array.Empty<int>();   // the token IS its own (biggest-chunk) atom — broken down on demand, not eagerly
            kind = ElementKind.Object;
        }
        else
        {
            components = key.Select(c => GetOrCreateAtom(c).Id).ToArray(); // legacy: eager char-composition (bounded reuse)
            kind = ElementKind.Composition;
        }
        // Born as its OWN token: a concept with no relations means just itself; meaning EMERGES as it accumulates
        // context (so even a single direct relation big↔large makes them cluster — they share each other's token).
        var created = _concepts.GetOrCreate(key, kind, () => FaceCodec.Token(key, _dim), components);
        _lattice.RegisterNode(key); // a new non-atom concept entered the active set
        created.LastSeenStep = _observeStep; // born fresh — decays only if it never gets reinforced (see DischargeIrrelevant)
        return created;
    }

    /// <summary>A reusable CHAR ATOM (Law S): one element per character, referenced (▷) by every word that uses it,
    /// so the atom set stays bounded (~the alphabet) no matter how large the vocabulary grows.</summary>
    private Element GetOrCreateAtom(char c)
        => _concepts.GetOrCreate("atom:" + c, ElementKind.Atom, () => new double[_semLen]);

    private static bool HasWhitespace(string s) { foreach (var c in s) if (char.IsWhiteSpace(c)) return true; return false; }

    // ── MERGE (PLATONIC_THEORY G5 — "a Relation is itself an element"; Chomsky's binary recursive composition; the plan
    //    in [[nova-merge-substrate-plan]]). THE ATOM OF STRUCTURE: bind two elements a, b under a LABEL into a NEW element
    //    m — kind Relation, positioned at the BLEND of their meanings, holding both as ▷ Components. m is itself a
    //    first-class element, so it can be an endpoint of the NEXT Merge → recursion + hierarchy from one operation (the
    //    role head and the copula-pivot were flat special cases of this). Endpoints may be atoms OR prior Merge outputs.
    //    The label TYPES the edge via the existing ∘-anchor pattern (reserved, never retrieved as an answer).
    public string Merge(string a, string b, string label)
    {
        var ea = GetOrCreateConcept(a);   // a/b may be atoms, words, or prior Merge symbols (then this returns them)
        var eb = GetOrCreateConcept(b);
        var sym = MergeSymbol(ea.Symbol, eb.Symbol, label);
        // POSITION = the centroid of the two meanings (the same template GetRelationElements uses), stored as the merged
        // element's ORBITAL so m carries a composed MEANING, not just a structural link. Read back via CloudOf later.
        var ca = CloudOf(ea); var cb = CloudOf(eb);
        var m = _concepts.GetOrCreate(sym, ElementKind.Relation, () => FaceCodec.Token(sym, _dim), new[] { ea.Id, eb.Id });
        _lattice.RegisterNode(sym);
        if (!string.IsNullOrWhiteSpace(label)) { ObserveContradiction(sym, "∘" + Normalize(label), 0.0); _mergeLabels.Add(Normalize(label)); } // TYPE the bind + remember the label as decode vocab
        // POSITION written LAST: a Merge's meaning is the BLEND of its parts (Laws C/S — "position derived from its
        // parts"), authoritative over the token-cloud that GetOrCreate/ObserveContradiction's RecomputeCloud would set.
        var dst = m.SemanticFace;
        for (var i = 0; i < _semLen && i < dst.Length && i < ca.Length && i < cb.Length; i++) dst[i] = 0.5 * (ca[i] + cb[i]);
        return sym;
    }

    /// <summary>Deterministic, content-addressed symbol for a Merge — the SAME (a, label, b) is the SAME element
    /// (idempotent), and nesting composes because a/b may themselves be merge symbols. Reserved bracket form (kept out of
    /// retrieval like the other structural symbols).</summary>
    private static string MergeSymbol(string a, string b, string label)
        => "⟨" + a + "·" + (string.IsNullOrWhiteSpace(label) ? "" : Normalize(label) + "·") + b + "⟩";

    /// <summary>Re-create a MERGE element from its content-addressed ⟨a·label·b⟩ symbol (snapshot reload): rebuild the
    /// CHILDREN first (recursively — a child may itself be a Merge subtree), then <see cref="Merge"/> them so the parent
    /// is born with its ▷ Components, exactly as the original assertion built it. Idempotent (content-addressed), so the
    /// shared sub-expressions of different facts converge. Returns the (normalized) symbol of the rebuilt element; for a
    /// non-bracket leaf it just ensures the concept exists. Used only by <see cref="ImportSnapshot"/>.</summary>
    private string ReconstructMerge(string symbol)
    {
        if (!symbol.StartsWith("⟨", StringComparison.Ordinal) || !symbol.EndsWith("⟩", StringComparison.Ordinal))
            return GetOrCreateConcept(symbol).Symbol; // a leaf endpoint — ensure it exists, no structure to rebuild
        var parts = SplitTopLevelMerge(symbol.Substring(1, symbol.Length - 2));
        if (parts.Count is not (2 or 3)) return GetOrCreateConcept(symbol).Symbol; // malformed — fall back to a plain token
        var a = ReconstructMerge(parts[0]);                       // child A built first (with its own components)
        var b = ReconstructMerge(parts[^1]);                      // child B built first
        return Merge(a, b, parts.Count == 3 ? parts[1] : "");     // now the parent is born with [aId, bId]
    }

    /// <summary>Split a Merge body on its TOP-LEVEL "·" separators (tracking ⟨⟩ nesting depth), so a nested child
    /// ⟨x·np·y⟩ stays one part. Yields [a, b] (no label) or [a, label, b].</summary>
    private static List<string> SplitTopLevelMerge(string body)
    {
        var parts = new List<string>(3);
        var depth = 0; var start = 0;
        for (var i = 0; i < body.Length; i++)
        {
            var c = body[i];
            if (c == '⟨') depth++;
            else if (c == '⟩') depth--;
            else if (c == '·' && depth == 0) { parts.Add(body.Substring(start, i - start)); start = i + 1; }
        }
        parts.Add(body.Substring(start));
        return parts;
    }

    /// <summary>DIAGNOSTIC: the raw element behind a symbol (for Merge / structure tests). Null if absent or archived.</summary>
    public Element? GetElement(string symbol)
        => _concepts.TryGet(Normalize(symbol), out var e) && !e.Archived ? e : null;

    // ── FACT MEMORY via MERGE (M1, [[nova-merge-substrate-plan]]). A fact is a Merge typed "is" — the KEY bound to the
    //    VALUE. It is made findable from the key (the key→fact edge); the fact→value link IS the Merge's components, so
    //    recall is structural tree-traversal (key → fact → the non-key component), NOT the RetrievalFrame/∘ret gate that
    //    kept mis-firing. The copula-pivot collapses into this single labelled Merge; key/value may be Merge subtrees.
    private const string FactLabel = "is";

    public string LearnFact(string key, string value)
    {
        var fact = Merge(key, value, FactLabel);            // ⟨key·is·value⟩ — the typed bind, positioned at the blend
        var k = Normalize(key);
        // BELIEF REVISION (G2 / free-energy): a fresh assertion makes `value` the key's CURRENT belief — weaken the key's
        // prior fact edges so recall returns the new truth, not a stale one (G6: weakened toward dormancy, not destroyed).
        foreach (var n in GetNeighbors(k, PlatonicNeighborhoodType.Relational, 16, 0.0).ToList())
            if (n.Concept.StartsWith("⟨", StringComparison.Ordinal) && !n.Concept.Equals(fact, StringComparison.Ordinal))
                DisruptAssociation(k, n.Concept);
        ObserveContradiction(k, fact, 0.0);                 // index the fact by its KEY (agreement → strong edge)
        return fact;
    }

    public bool TryRecallFact(string key, out string value)
    {
        value = string.Empty;
        var k = Normalize(key);
        if (!_concepts.TryGet(k, out var ke) || ke.Archived) return false;
        // FOLLOW the strongest fact edge from the key to its fact element (a ⟨…⟩ Merge), then read the VALUE component
        // (the part that is not the key) — recall = walking the Merge structure.
        var fact = GetNeighbors(k, PlatonicNeighborhoodType.Relational, 16, 0.0)
            .Where(n => n.Concept.StartsWith("⟨", StringComparison.Ordinal))
            .OrderByDescending(n => n.Confidence).Select(n => n.Concept).FirstOrDefault();
        if (fact is null || !_concepts.TryGet(fact, out var fe) || fe.Components.Length != 2) return false;
        foreach (var cid in fe.Components)
        {
            var ce = _concepts.ById(cid);
            if (ce.Id != ke.Id && !ce.Archived) { value = ce.Symbol; return true; } // the non-key component is the value
        }
        return false;
    }

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
        // ADDRESS SPACE (A): a COMPOSITE (Merge ⟨a·label·b⟩) carries a DECODABLE identity in the FROZEN structure band
        // [208,400) — the ordered child coordinates + label. AssemblePositiveFace only overlays the orbital onto the
        // tail [OrbitalStart,dim), so it never clobbers the structure band; we mirror the element's ▷ children into it
        // here so the composite decodes from its frozen face (PlatonicFaceDecoder.DecodeStructure), not just its symbol/lossy orbital.
        // The band lies OUTSIDE FaceAwareDistance's [_semStart,dim) tail, so semantic retrieval is unaffected.
        if (element is not null) WriteStructureBand(face, element);
        // ADDRESS SPACE (op band): a FUNCTION element (a realised operation) carries its op-code in the FROZEN op band
        // [400,416) + the kind band = Function, so the operation decodes from its coordinate (TryDecodeCoordinate). The
        // op band lies BELOW the orbital tail, so the learned orbital never clobbers it (mirrors WriteStructureBand).
        if (element is not null) WriteOpBand(face, element);
        return face;
    }

    /// <summary>
    /// (Op band) Write a FUNCTION element's op-code into the FROZEN op band [400,416) via
    /// <see cref="PlatonicFaceComposer.EncodeOp"/>, and stamp the kind band = Function (overriding the symbol's
    /// best-effort Object code), so an operation's coordinate decodes to (kind=Function, op=token) — the 7th frozen
    /// band. The op token = the element's reserved symbol minus the <see cref="OpSymbolPrefix"/>. No-op for a
    /// non-Function element or below address-space dims (legacy faces have no op band).
    /// </summary>
    private void WriteOpBand(double[] face, Element e)
    {
        if (!Core.FaceLayout.IsAddressSpace(_dim) || e.Kind != ElementKind.Function
            || !e.Symbol.StartsWith(OpSymbolPrefix, StringComparison.Ordinal)) return;
        var op = e.Symbol.Substring(OpSymbolPrefix.Length);
        PlatonicFaceComposer.EncodeKind(face, PlatonicKind.Function, _dim);
        PlatonicFaceComposer.EncodeOp(face, op, _dim);
    }

    /// <summary>
    /// (A) Mirror a COMPOSITE's ▷ children into the FROZEN structure band [208,400) of its assembled face via
    /// <see cref="PlatonicFaceComposer.EncodeStructure"/>, so the composite's identity is decodable from the
    /// coordinate itself. Only Merge composites (⟨…⟩) are mirrored — atoms / words / numbers carry no structure.
    /// The structure band lies BELOW the orbital tail, so the learned orbital never clobbers it. No-op below
    /// address-space dims (legacy faces have no structure band).
    /// </summary>
    private void WriteStructureBand(double[] face, Element e)
    {
        if (!Core.FaceLayout.IsAddressSpace(_dim) || e.Components.Length == 0
            || !e.Symbol.StartsWith("⟨", StringComparison.Ordinal)) return;
        var childCoords = new double[e.Components.Length][];
        for (var i = 0; i < e.Components.Length; i++)
        {
            var ce = _concepts.ById(e.Components[i]);
            childCoords[i] = FullFace(ce.Symbol, ce); // recurse: a child may itself be a Merge (bounded by nesting depth)
        }
        PlatonicFaceComposer.EncodeStructure(face, childCoords, MergeLabelOf(e), _dim);
    }

    /// <summary>Recover a Merge's LABEL from its content-addressed symbol ⟨a·label·b⟩, using the authoritative ▷
    /// component symbols (a, b) to strip the surrounding parts — robust to nested ⟨…⟩ children. Empty when unlabeled.</summary>
    private string MergeLabelOf(Element e)
    {
        if (e.Components.Length < 2 || !e.Symbol.StartsWith("⟨", StringComparison.Ordinal)
            || !e.Symbol.EndsWith("⟩", StringComparison.Ordinal)) return string.Empty;
        var inner = e.Symbol.Substring(1, e.Symbol.Length - 2); // strip ⟨ ⟩
        var aSym = _concepts.ById(e.Components[0]).Symbol;
        var bSym = _concepts.ById(e.Components[1]).Symbol;
        if (inner.StartsWith(aSym + "·", StringComparison.Ordinal)) inner = inner.Substring(aSym.Length + 1);
        if (inner.EndsWith(bSym, StringComparison.Ordinal)) inner = inner.Substring(0, inner.Length - bSym.Length);
        return inner.Trim('·');
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
        EnsureCloudsFresh();
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

    // IDENTITY / ADDRESSING distance (D) over the FROZEN address bands [KindStart,OrbitalStart) = kind+spelling+
    // structure+op. These dims are a drift-free, codec-derived pure function of the symbol, so two coordinates with
    // the SAME identity are exactly equal here regardless of how their learned tails wandered. Used only as a
    // TIE-BREAK behind the semantic-tail distance (keeps exact-identity matches stable). Zero below address-space
    // dims (no frozen address bands there → no secondary signal, legacy ordering preserved).
    private double FrozenIdentityDistance(IReadOnlyList<double> q, IReadOnlyList<double> c)
        => Core.FaceLayout.IsAddressSpace(_dim)
            ? RangeDistance(q, c, Core.FaceLayout.KindStart, Core.FaceLayout.OrbitalStart)
            : 0.0;
    private const double FrozenTieEpsilon = 1e-6; // tail distances within this are TIES → broken by frozen identity

    // The pure-codec FROZEN identity face of a symbol (kind+spelling+arith, zero orbital, NO structure band) — the
    // address to score an arbitrary/decoded coordinate against (TryDecodeCoordinate confidence).
    private double[] FrozenAddressOf(string symbol) => FaceCodec.AssemblePositiveFace(symbol, new double[_semLen], _dim);

    // The pure-codec FROZEN face of a FUNCTION/op coordinate: the reserved symbol's address with the kind band stamped
    // Function and the op band [400,416) written — the address a DECODED op coordinate is scored against (confidence).
    private double[] FrozenFunctionAddressOf(string opToken)
    {
        var face = FrozenAddressOf(OpSymbol(opToken));
        PlatonicFaceComposer.EncodeKind(face, PlatonicKind.Function, _dim);
        PlatonicFaceComposer.EncodeOp(face, opToken, _dim);
        return face;
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

        var scored = new List<(string Sym, double Tail, double Frozen)>();
        foreach (var cand in pool)
        {
            if (cand.Equals(self, StringComparison.Ordinal)) continue;
            if (!_concepts.TryGet(cand, out var ce) || ce.Archived || ce.Kind == ElementKind.Atom) continue;
            var cf = FullFace(cand, ce);
            scored.Add((cand, FaceAwareDistance(q, cf), FrozenIdentityDistance(q, cf)));
        }
        // PRIMARY ordering = the semantic-tail distance (unchanged, routing degraded-by-design). At address-space dims
        // add the IDENTITY/ADDRESSING tie-break (D): when two candidates are within FrozenTieEpsilon on the tail, the
        // one whose FROZEN address [42,416) is closer to the query wins, so exact-identity matches stay stable and
        // drift-free. Below address-space dims keep the stable OrderBy (byte-identical legacy ordering).
        if (!Core.FaceLayout.IsAddressSpace(_dim))
            return scored.OrderBy(x => x.Tail).Take(limit).Select(x => (x.Sym, x.Tail)).ToArray();
        scored.Sort((x, y) =>
        {
            var d = x.Tail - y.Tail;
            if (d > FrozenTieEpsilon) return 1;
            if (d < -FrozenTieEpsilon) return -1;
            return x.Frozen.CompareTo(y.Frozen);
        });
        return scored.Take(limit).Select(x => (x.Sym, x.Tail)).ToArray();
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

    // ──────────────────────────────────────────────────────── Navigation SEAMS for the NN-navigator (C; do NOT build a
    //   navigator here). Reasoning-as-a-situated-walk ([[nova-nn-navigator-vision]]) is an ENGINE-layer concern: the
    //   SELF-object the walker carries, the HALT action, and the STEP policy all live in the navigator, NOT in the space.
    //   The space only ever provides two primitives a walker steps through: (1) decode an arbitrary POSITION (realised OR
    //   latent void) to the element it addresses, and (2) report the egocentric NEIGHBOURHOOD ("what is around me").

    /// <summary>
    /// (C·1) Decode an ARBITRARY coordinate — realised OR latent void, with NO stored Element — to the element it
    /// addresses, proving "every coordinate is an element". Pure function of the face + the L1 decoders: numeric
    /// (poly/log, zero-storage) → composite (structure band) → spelling (char slots), selected by the kind code /
    /// numeric signal. Below address-space dims there is no decodable address → false.
    /// </summary>
    public bool TryDecodeCoordinate(double[] face, out PlatonicKind kind, out string symbol, out double confidence)
    {
        kind = PlatonicKind.None; symbol = string.Empty; confidence = 0.0;
        if (face is null || face.Length == 0 || !Core.FaceLayout.IsAddressSpace(_dim)) return false;
        kind = PlatonicFaceDecoder.DecodeKind(face, _dim);
        // (1) NUMBER — read straight off the poly/log bands (exact, zero storage). Numbers carry the all-zero kind code.
        if (kind == PlatonicKind.None)
        {
            var (val, q, faceSel) = PlatonicFaceDecoder.DecodeNumericFromPrediction(face, _dim);
            if (faceSel != "none" && q > 0.3)
            {
                symbol = val.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture);
                confidence = q;
                return true;
            }
        }
        // (1b) FUNCTION / OP — decode WHICH operation off the frozen op band [400,416), nearest registered op-code.
        // Works for a LATENT op coordinate (built from the codec, never stored) — the same "decode the void" property
        // the number band has: the operation is a route over the address space, decodable from its coordinate alone.
        if (kind == PlatonicKind.Function)
        {
            var op = PlatonicFaceDecoder.DecodeOp(face, _dim, _operationTokens);
            if (!string.IsNullOrEmpty(op))
            {
                symbol = op;
                confidence = 1.0 / (1.0 + FrozenIdentityDistance(face, FrozenFunctionAddressOf(op)));
                return true;
            }
        }
        // (2) COMPOSITE — decode the ordered child digests + label from the structure band (a best-effort symbol; a
        // realised composite is reconciled against the authoritative ▷ children in Element.Components below).
        if (kind is PlatonicKind.Relation or PlatonicKind.Composition)
        {
            var dec = PlatonicFaceDecoder.DecodeStructure(face, _dim, _mergeLabels);
            if (dec.Children.Count > 0)
            {
                var parts = new List<string>();
                if (dec.Label.Length > 0) parts.Add(dec.Label);
                parts.AddRange(dec.Children);
                symbol = "⟨" + string.Join("·", parts) + "⟩";
                confidence = 0.5 + 0.5 * Math.Min(1.0, dec.Children.Count / (double)Core.FaceLayout.StructureSlots);
                return true;
            }
        }
        // (3) WORD / ATOM — read the spelling band back to its token; confidence = how exactly the decoded symbol's
        // frozen address re-matches the coordinate (1 for an on-address point, lower out in the void).
        var spelled = PlatonicFaceDecoder.CharSlotDecode(face, _dim);
        if (!string.IsNullOrEmpty(spelled))
        {
            symbol = spelled;
            confidence = 1.0 / (1.0 + FrozenIdentityDistance(face, FrozenAddressOf(spelled)));
            if (kind == PlatonicKind.None) kind = PlatonicFaceComposer.KindForSymbol(spelled);
            return true;
        }
        return false;
    }

    /// <summary>
    /// (C·2) The egocentric "what is around me" sensor: the nearest concepts to <paramref name="atSymbol"/> (via the
    /// existing fresh lattice/scan), each annotated with its relation DEGREE so the walker can spot hub / landmark
    /// nodes. A thin wrapper over <see cref="GetNearestConceptsFresh"/> + <see cref="GetRelationDegree"/>; carries no
    /// policy of its own (the navigator decides where to step).
    /// </summary>
    public IReadOnlyList<(string Symbol, double Distance, int Degree)> Neighborhood(string atSymbol, int k)
        => GetNearestConceptsFresh(atSymbol, seeds: null, maxNeighbors: Math.Clamp(k, 1, 32))
            .Select(n => (n.Symbol, n.Distance, GetRelationDegree(n.Symbol)))
            .ToArray();

    // ──────────────────────────────────────── NAVIGATOR MOTION PRIMITIVES (STEP 1, PLATONIC_NAVIGATOR.md §5.1/§5.2/§9).
    //   The substrate seams a future NN walker STEPS WITH — no NN, no policy, no training here; just the moves. Every
    //   navigator action (STEP-near / FOLLOW-edge / COMPUTE-jump / TOWARD-landmark) reduces to ONE primitive: emit a
    //   target coordinate, then let the lattice LAND the foot on the nearest decodable coordinate (TryLand). The walk
    //   may also WRITE — a useful passed-through latent coordinate is committed (Materialise, the genesis tick). It SENSES
    //   over the FROZEN ADDRESS (FrozenIdentityDistance), the drift-free identity, not the live tail.

    // Tunables for the motion primitives (kept local — these are substrate seams, not policy).
    private const double LandDecodeConfidence = 0.55; // below this the target is "between addresses" → SNAP to a real concept
    private const int    LandSnapPool = 64;           // lattice harvest size for the snap rescore

    /// <summary>
    /// (NAV·1) MOTION PRIMITIVE — "the lattice lands the step" (§5.1). Every action emits/computes a target coordinate;
    /// this resolves where the foot actually falls — the nearest DECODABLE coordinate (realised OR latent). Two regimes:
    ///   (a) DECODE-FIRST (O(1), zero index cost): if the target IS already a decodable address (TryDecodeCoordinate
    ///       fires confidently), the foot lands EXACTLY there. A COMPUTE-jump to GetFreshNumericEmbedding(141) lands on
    ///       "141" with NO lattice query, even though "141" was never stored — the void is on-address terrain.
    ///   (b) SNAP: otherwise the target is off-address ("between" addresses); harvest a candidate pool from the lattice
    ///       (rides-the-lattice-for-speed) and RESCORE by <see cref="FrozenIdentityDistance"/> — the drift-free address
    ///       distance — returning the nearest REAL concept. "Head this direction, snap to the nearest decodable coord."
    /// <paramref name="landedFace"/> is the canonical codec face of the landing (so the walker can re-sense from it).
    /// </summary>
    public bool TryLand(double[] targetFace, out string symbol, out PlatonicKind kind, out double[] landedFace, out double confidence)
    {
        symbol = string.Empty; kind = PlatonicKind.None; landedFace = Array.Empty<double>(); confidence = 0.0;
        if (targetFace is null || targetFace.Length == 0 || !Core.FaceLayout.IsAddressSpace(_dim)) return false;

        // (a) DECODE-FIRST — a clean address (realised or latent void) lands exactly, no index cost.
        if (TryDecodeCoordinate(targetFace, out kind, out symbol, out confidence)
            && confidence >= LandDecodeConfidence && !string.IsNullOrEmpty(symbol))
        {
            // DECODE-FROM-VOID RECOVERY (gated): the foot is landing on a decodable identity with NO active element
            // (evicted / latent). Re-materialise the conserved identity from its coordinate so the walk lands on (and
            // re-senses from) the recovered element — not just a transient decode. Guarded: RecoverFromCoordinate
            // materialises nothing unless the coordinate cleanly decodes to a valid identity. Default-off = byte-identical.
            if (RecoverFromVoid && !ContainsConcept(symbol)) RecoverFromCoordinate(targetFace);
            landedFace = CanonicalFace(symbol);
            return true;
        }

        // (b) SNAP — off-address target → nearest REAL decodable concept by FROZEN-address distance (rides the lattice).
        EnsureCloudsFresh();
        string? best = null; var bestDist = double.PositiveInfinity;
        foreach (var cand in HarvestCandidates(targetFace, LandSnapPool))
        {
            if (!_concepts.TryGet(cand, out var ce) || ce.Archived || ce.Kind == ElementKind.Atom) continue;
            var d = FrozenIdentityDistance(targetFace, FullFace(cand, ce));
            if (d < bestDist) { bestDist = d; best = cand; }
        }
        if (best is null) { symbol = string.Empty; kind = PlatonicKind.None; confidence = 0.0; return false; }
        symbol = best;
        kind = PlatonicFaceComposer.KindForSymbol(best);
        landedFace = CanonicalFace(best);
        confidence = 1.0 / (1.0 + bestDist);
        return true;
    }

    /// <summary>
    /// (NAV·2) GENESIS-TICK WRITE-PATH (§5.2). Materialise a passed-through LATENT coordinate: decode it to its symbol
    /// and commit a REALISED element (an orbital is born). A no-op if the coord does not decode. After the call the
    /// symbol is realised (<see cref="ContainsConcept"/> true). LastSeenStep is bumped so relevance-decay keeps it ONLY
    /// if the walk later reinforces it — trails that never pay off decay back to latent. Navigation is not read-only.
    /// </summary>
    public void Materialise(double[] coord)
    {
        if (coord is null || coord.Length == 0) return;
        if (!TryDecodeCoordinate(coord, out _, out var symbol, out _) || string.IsNullOrEmpty(symbol)) return;
        Materialise(symbol);
    }

    /// <summary>(NAV·2, overload) Materialise a symbol directly (the decoded landing): realise the element and bump its
    /// recency so relevance-decay keeps it only if reinforced. Idempotent (re-materialising just refreshes recency).</summary>
    public Element Materialise(string symbol)
    {
        var e = GetOrCreateConcept(symbol);
        e.LastSeenStep = _observeStep; // bump recency — kept by decay only if the walk keeps re-touching it
        return e;
    }

    /// <summary>
    /// DECODE-FROM-THE-VOID RECOVERY (gated by <see cref="RecoverFromVoid"/>; G6 via the latent address). Re-materialise an
    /// EVICTED / LATENT / never-materialised concept from its COORDINATE alone: decode the frozen identity bands to a
    /// symbol+kind (<see cref="TryDecodeCoordinate"/>) and — only if that is a CONFIDENT, VALID, recoverable identity —
    /// realise the element and RECONSTRUCT its orbital from KNOWN PARTS so it lands near its family (else identity-only).
    /// The materialised space is a CACHE over the conserved decodable void: eviction freed the learned orbital + the store
    /// slot, this decodes the identity back when the walk / retrieval reaches its address.
    /// <para>
    /// THE GUARD (the danger is materialising junk from a coordinate that does not cleanly decode). A "confident valid
    /// decode" means ALL of: the decode fired with a non-empty symbol; its <b>confidence ≥ <see cref="RecoverDecodeConfidence"/></b>
    /// (for a WORD/op the confidence IS the frozen round-trip 1/(1+frozen-distance), so this alone pins it on its address);
    /// it is a <b>recoverable concept kind</b> — NOT a bare number and NOT a ∘-operation (both are zero-storage
    /// void-decodable already, the codec re-derives them on demand, so materialising them would only pollute the store),
    /// and not a reserved/op token; and a final <b>frozen round-trip</b> — the recovered element's own canonical (codec)
    /// face must re-decode to the SAME symbol AND sit within <see cref="RecoverRoundTripTolerance"/> of the input's frozen
    /// identity [42,416). The round-trip is what rejects a STRUCTURED noise coordinate (whose structural confidence can be
    /// high but whose clean reconstructed structure band does not re-match the noise) — it is materialised then ROLLED
    /// BACK. Returns the re-materialised (or already-active) element, or <c>null</c> when nothing valid decodes.
    /// </para>
    /// </summary>
    public Element? RecoverFromCoordinate(double[] face)
    {
        if (!RecoverFromVoid || face is null || face.Length == 0 || !Core.FaceLayout.IsAddressSpace(_dim)) return null;
        // (1) DECODE the frozen identity off the coordinate.
        if (!TryDecodeCoordinate(face, out var kind, out var symbol, out var confidence) || string.IsNullOrEmpty(symbol))
            return null;
        if (confidence < RecoverDecodeConfidence) return null;
        var key = Normalize(symbol);
        // (2) GUARD — only a recoverable CONCEPT identity. Numbers + ∘-operations are zero-storage void-decodable already
        //     (the codec/homomorphism re-derives them on demand), so they are never materialised into the store here.
        if (kind == PlatonicKind.Function || FaceCodec.IsNumeric(key) || IsReservedConcept(key) || IsOperationToken(key))
            return null;
        // (3) ALREADY ACTIVE → nothing was evicted; hand back the live element (idempotent).
        if (_concepts.TryGet(key, out var live) && IsRetrievable(live)) return live;
        // (4) MATERIALISE the conserved identity (frozen bands re-derived deterministically by the codec) and RECONSTRUCT
        //     its orbital from known parts. We remember whether it pre-existed (archived) so a junk roll-back only ever
        //     removes something WE created.
        var existedBefore = _concepts.TryGet(key, out _);
        var e = GetOrCreateConcept(key);
        ReconstructRecoveredOrbital(key, e);
        // (5) VERIFY the round-trip on the element's OWN canonical face — the junk-rejection guard (see remarks).
        var canonical = FullFace(key, e);
        var roundTrips = TryDecodeCoordinate(canonical, out _, out var back, out _)
            && string.Equals(Normalize(back), key, StringComparison.Ordinal)
            && FrozenIdentityDistance(face, canonical) <= RecoverRoundTripTolerance;
        if (!roundTrips)
        {
            if (!existedBefore) DischargeConcept(key); // materialise NOTHING from a coordinate that doesn't cleanly decode
            return null;
        }
        e.LastSeenStep = _observeStep; // recency bump — kept by decay only if the walk/retrieval reinforces it
        return e;
    }

    // Reconstruct a recovered concept's ORBITAL from its KNOWN parts so it lands near its family (the generative-atoms
    // payoff): a multi-word whole blends its known WORD components' clouds; a single token blends its known MORPHEMES'
    // clouds (ComposeMeaningFromKnownParts). With no known parts the orbital is left as the bare identity token (neutral).
    private void ReconstructRecoveredOrbital(string key, Element e)
    {
        if (HasWhitespace(key))
        {
            var acc = new double[_semLen]; var n = 0;
            foreach (var w in key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (!_concepts.TryGet(w, out var part) || part.Archived || part.Kind == ElementKind.Atom) continue;
                var c = CloudOf(part);
                for (var i = 0; i < _semLen && i < c.Length; i++) acc[i] += c[i];
                n++;
            }
            if (n == 0) return;
            var dst = e.SemanticFace;
            for (var i = 0; i < _semLen && i < dst.Length; i++) dst[i] = acc[i] / n;
            _lattice.MarkEmbeddingsDirty();
        }
        else if (GenerativeAtoms && ComponentCoverage(key).Known > 0)
        {
            ComposeMeaningFromKnownParts(key, e);
        }
    }

    // The canonical (codec) face of a decoded symbol: a realised concept carries its learned cloud; a latent symbol
    // (never stored — a computed number or an unseen word) gets its pure address face. Either decodes back to itself.
    private double[] CanonicalFace(string symbol)
    {
        var key = Normalize(symbol);
        return _concepts.TryGet(key, out var e) && !e.Archived ? FullFace(key, e) : FullFace(key, null);
    }

    // RIDES-THE-LATTICE-FOR-SPEED harvest: below LatticeMinNodes the exact scan over all active concepts is cheap AND
    // always fresh; at scale the VP-tree harvests a bounded near pool in O(log N). LIMITATION (PLATONIC_NAVIGATOR.md §9):
    // the lattice ranks by the SEMANTIC tail, so at scale a frozen-near concept whose cloud differs can be MISSED from
    // the pool — the harvest only BOUNDS which candidates we then rescore (always by FrozenIdentityDistance), it never
    // changes HOW they score. A dedicated FROZEN-ADDRESS VP-tree is the future optimization; rescoring a lattice-
    // harvested pool by FrozenIdentityDistance is the correct first cut (exact at the <384-node test scale).
    private IEnumerable<string> HarvestCandidates(double[] queryFace, int poolSize)
        => _concepts.ActiveCount >= LatticeMinNodes
            ? _lattice.GetSemanticNeighbors(queryFace, Math.Clamp(poolSize, 16, 512), null).Select(n => n.Name)
            : _concepts.All.Where(e => !e.Archived && e.Kind != ElementKind.Atom).Select(e => e.Symbol);

    public double[] ComputeRoutePerception(string anchor, double transformReliability = 0.0)
    {
        EnsureCloudsFresh();
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
        var now = ++_observeStep;                                 // advance the observation clock
        ea.ObservationCount++; eb.ObservationCount++;
        ea.LastSeenStep = eb.LastSeenStep = now;                  // both endpoints are REINFORCED + made recent → kept

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
        rel.LastSeenStep = now;

        // DISTRIBUTIONAL MEANING (PLATONIC_NUCLEUS.md; validated in LargeFaceMeaningTests). A concept's large-face
        // cloud = token(self) + Σ over its relational context of affinity·token(neighbour) (agree adds, contradict
        // subtracts) — computed PRESENCE-based (each neighbour once, NOT per-observation, so repeated training never
        // drowns the self-token). Related concepts share context → clouds overlap; unrelated go orthogonal, no
        // repulsion tuning. Both endpoints' neighbour sets just changed, so recompute both.
        if (BatchedCloudGpu)
        {
            _dirtyCloud.Add(a); _dirtyCloud.Add(b);             // defer — recompute once per flush (dedup), on GPU
            if (++_obsSinceFlush >= CloudFlushInterval) FlushClouds();
        }
        else
        {
            RecomputeCloud(ea);
            RecomputeCloud(eb);
        }
        if (now - _lastDischargeStep >= DischargeInterval)
        {
            _lastDischargeStep = now;                 // set FIRST so re-entrant observes (from Decompose) don't re-trigger
            DischargeIrrelevant();
            if (GenerativeAtoms) DecomposeReinforced(); // break PROVEN tokens into candidate sub-atoms; they compete via decay
        }
    }

    /// <summary>
    /// DISCHARGE IRRELEVANCE (the live maintenance — NOT a size cap). Release concepts and relations that are NOISE: barely
    /// observed (≤ <see cref="DischargeObservations"/>) AND gone stale (not seen for <see cref="DischargeStalenessWindow"/>
    /// observation-steps). Anything reinforced (observed more) or recently active is ALWAYS kept, so reinforced signals like
    /// the function-word geometry are never disturbed — that is what a numeric cap got wrong. Atoms (the bounded reusable
    /// base) and structural ∘-anchors are never discharged; nor is a concept still referenced as a ▷-component of a retained
    /// whole. A discharged element re-forms from observation if it ever recurs (it was only noise). O(active) but runs only
    /// once per <see cref="DischargeInterval"/> observations, and the active set stays bounded by RELEVANCE, not fiat.
    /// </summary>
    private void DischargeIrrelevant()
    {
        var now = _observeStep;

        // RELATIONS — release an edge only once it has gone STALE beyond the grace its UTILITY earned. Utility is led by
        // SuccessCount (it contributed to correct answers), with observation as a weak floor and failures as a penalty.
        // A useful edge keeps getting re-used/credited (LastUsedStep/LastSeenStep advance) long before its grace expires.
        List<Relation>? deadRels = null;
        foreach (var r in _relations.Values)
        {
            if (r.Left.StartsWith('∘') || r.Right.StartsWith('∘')) continue;     // structural ∘-type edges
            var lastActive = Math.Max(r.LastSeenStep, r.LastUsedStep);
            if (now - lastActive <= GraceFor(RelationUtility(r))) continue;       // still within earned grace
            (deadRels ??= new()).Add(r);
        }
        if (deadRels is not null) foreach (var r in deadRels) DropRelation(r);

        // CONCEPTS — a concept is RELEVANT to the degree its edges earned correct answers; observation is the floor.
        // Its utility = the best SuccessCount among its relations (precomputed once) + a log-damped observation floor.
        Dictionary<string, int>? bestSuccess = null;
        HashSet<int>? referenced = null;
        List<string>? deadConcepts = null;
        foreach (var e in _concepts.All)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || e.Symbol.StartsWith('∘')) continue;
            bestSuccess ??= BuildBestRelationSuccess();
            var succ = bestSuccess.TryGetValue(e.Symbol, out var s) ? s : 0;
            var util = 2.0 * succ + Math.Log(1.0 + e.ObservationCount);
            if (now - e.LastSeenStep <= GraceFor(util)) continue;                 // still within earned grace
            referenced ??= BuildReferencedComponentIds();
            if (referenced.Contains(e.Id)) continue;                             // a part of a retained whole (keeps ▷ / ById valid)
            (deadConcepts ??= new()).Add(e.Symbol);
        }
        if (deadConcepts is not null) foreach (var s in deadConcepts) DischargeConcept(s);

        // HARD CAP — the overflow safety net. Decay above only releases STALE concepts; a corpus stream re-observes its
        // long tail so nothing goes stale and the active set can still blow past any reasonable size. Enforce the ceiling
        // by distributional VALUE regardless of recency, so the useless long tail is dropped even when "recently seen".
        EnforceActiveConceptCap();
    }

    /// <summary>
    /// Enforce the <see cref="MaxActiveConcepts"/> ceiling (the pure-overflow safety net). After the grace-decay prune,
    /// if the active CONCEPT count still exceeds the cap, evict the LOWEST-UTILITY excess concepts down to the cap —
    /// using the SAME success-weighted distributional utility the grace path uses (<c>2·bestSuccess + log(1+obs)</c>),
    /// no word lists. PROTECTED (never evicted): ∘-anchors, atoms, Functions, numbers, and a concept still referenced as
    /// a ▷-component of a retained whole — so arithmetic, operations and relation structure are untouched. Unlike the
    /// grace path this ignores recency, so it holds even when NOTHING has decayed (every word "recently seen").
    /// </summary>
    private void EnforceActiveConceptCap()
    {
        var cap = MaxActiveConcepts;
        if (cap <= 0) return;
        var overflow = _concepts.ActiveConceptCount - cap;
        if (overflow <= 0) return;

        var bestSuccess = BuildBestRelationSuccess();
        var referenced = BuildReferencedComponentIds();
        List<(string Symbol, double Util)>? candidates = null;
        foreach (var e in _concepts.All)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || e.Kind == ElementKind.Function) continue;
            if (e.Symbol.StartsWith('∘') || FaceCodec.IsNumeric(e.Symbol)) continue;  // anchors + numbers are protected
            if (referenced.Contains(e.Id)) continue;                                  // a part of a retained whole (keeps ▷ / ById valid)
            var succ = bestSuccess.TryGetValue(e.Symbol, out var s) ? s : 0;
            var util = 2.0 * succ + Math.Log(1.0 + e.ObservationCount);
            (candidates ??= new()).Add((e.Symbol, util));
        }
        if (candidates is null) return;
        candidates.Sort((a, b) => a.Util.CompareTo(b.Util));                          // ascending — weakest first
        var drop = Math.Min(overflow, candidates.Count);
        for (var i = 0; i < drop; i++) DischargeConcept(candidates[i].Symbol);
    }

    // RELEVANCE = success-weighted contribution to CORRECT ANSWERS (strong), with log-damped observation as a weak floor
    // and failures as a penalty. "Helped answer" outranks "was merely seen". Clamped ≥ 0.
    private static double RelationUtility(Relation r)
        => Math.Max(0.0, 2.0 * r.SuccessCount + Math.Log(1.0 + r.ObservationCount) - r.FailureCount);

    // FORGETTING CURVE: the base staleness window, EXTENDED in proportion to utility. Utility 0 → one window of grace;
    // high utility → many windows (effectively immortal while it keeps earning answers). Everything decays; utility buys grace.
    private double GraceFor(double utility) => (long)(DischargeStalenessWindow * (1.0 + Math.Max(0.0, utility)));

    // Best SuccessCount over each concept's incident relations — how much its edges have CONTRIBUTED TO CORRECT ANSWERS.
    private Dictionary<string, int> BuildBestRelationSuccess()
    {
        var best = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var r in _relations.Values)
        {
            if (r.SuccessCount <= 0) continue;
            if (!best.TryGetValue(r.Left, out var l) || r.SuccessCount > l) best[r.Left] = r.SuccessCount;
            if (!best.TryGetValue(r.Right, out var rr) || r.SuccessCount > rr) best[r.Right] = r.SuccessCount;
        }
        return best;
    }

    // ── GENERATIVE DECOMPOSITION (the tick's break-down op, [[nova-nn-directed-generative-tick]]) ─────────────────────
    /// <summary>
    /// Break a token into candidate SUB-ATOMS — its characters and short adjacent n-grams — each created as a first-class
    /// atom and OBSERVED against the parent (a ▷ "agrees-with-its-part" edge). The candidates then COMPETE via the relevance
    /// decay: a sub-atom that RECURS across tokens and CONTRIBUTES TO CORRECT ANSWERS (a morpheme like "hel") accrues
    /// utility and survives, while a one-off substring goes stale and discharges. So useful granularity is DISCOVERED, not
    /// fixed at char. Idempotent (the SAME sub-atom across tokens is the same element — that's how it recurs and wins).
    /// </summary>
    public IReadOnlyList<string> Decompose(string symbol)
    {
        var key = Normalize(symbol);
        if (key.Length < 2 || FaceCodec.IsNumeric(key) || key.StartsWith('∘') || HasWhitespace(key)) return Array.Empty<string>();
        _decomposed.Add(key);
        var parts = CandidateGrams(key);
        foreach (var p in parts)
            if (!string.Equals(p, key, StringComparison.Ordinal)) ObserveContradiction(key, p, 0.05); // token ▷ part (creates + tracks it)
        return parts;
    }

    // The candidate sub-atoms of a token: its characters + adjacent n-grams up to MaxDecomposeGram (the candidate morphemes).
    private static IReadOnlyList<string> CandidateGrams(string key)
    {
        var parts = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        void Add(string s) { if (seen.Add(s)) parts.Add(s); }
        foreach (var c in key) Add(c.ToString());
        for (var n = 2; n <= Math.Min(MaxDecomposeGram, key.Length - 1); n++)
            for (var i = 0; i + n <= key.Length; i++) Add(key.Substring(i, n));
        return parts;
    }

    // The tick driver: decompose the PROVEN tokens (reinforced ≥ DecomposeMinObservations) not yet broken down. Collect
    // first, then decompose — Decompose mutates the store, so we never decompose while iterating it. Budgeted per sweep.
    private void DecomposeReinforced()
    {
        var budget = 32;
        List<string>? todo = null;
        foreach (var e in _concepts.All)
        {
            if (budget <= 0) break;
            if (e.Archived || e.Kind == ElementKind.Atom || e.Symbol.Length < 2) continue;
            if (e.ObservationCount < DecomposeMinObservations || _decomposed.Contains(e.Symbol)) continue;
            if (FaceCodec.IsNumeric(e.Symbol) || e.Symbol.StartsWith('∘') || HasWhitespace(e.Symbol)) continue;
            (todo ??= new()).Add(e.Symbol); budget--;
        }
        if (todo is not null) foreach (var s in todo) Decompose(s);
    }

    private HashSet<int> BuildReferencedComponentIds()
    {
        var set = new HashSet<int>();
        foreach (var e in _concepts.All)
            foreach (var c in e.Components) set.Add(c);
        return set;
    }

    // Release a discharged (noise) concept: drop it from the store + spatial index + its dangling edges. No orbital is
    // preserved — it was never reinforced enough to be a real distinction; it re-creates fresh from observation if it recurs.
    private void DischargeConcept(string symbol)
    {
        if (_adjacency.TryGetValue(symbol, out var nbrs))
        {
            foreach (var n in nbrs.ToArray())
            {
                _relations.Remove(Key(symbol, n));
                if (_adjacency.TryGetValue(n, out var back)) back.Remove(symbol);
            }
            _adjacency.Remove(symbol);
        }
        _lattice.UnregisterNode(symbol);
        _concepts.Remove(symbol);
    }

    // Remove an edge from the active index (both the relation record and the symmetric adjacency). Unlike a concept, a
    // pruned co-occurrence edge has no learned orbital to preserve — it simply re-forms if the pair is observed again.
    private void DropRelation(Relation r)
    {
        _relations.Remove(Key(r.Left, r.Right));
        if (_adjacency.TryGetValue(r.Left, out var sa)) sa.Remove(r.Right);
        if (_adjacency.TryGetValue(r.Right, out var sb)) sb.Remove(r.Left);
    }

    /// <summary>
    /// Recompute a concept's large-face cloud as the PRESENCE-based superposition of its relational context:
    /// token(self) + Σ_{neighbour n} (1−2κ)·token(n). Each neighbour contributes ONCE (not per-observation), so the
    /// self-token keeps weight 1 and repeated training cannot drown it — the fix for directly-related-but-context-poor
    /// pairs (big↔large) going orthogonal. Agree (low κ) adds the neighbour's token (clouds overlap); contradict (high
    /// κ) subtracts it. Drift-free: always reflects the current relation set. The arithmetic face is untouched.
    /// </summary>
    // The deterministic identity token of a concept (FaceCodec.Token), MEMOIZED on the element. Pure function of
    // (symbol, dim) → caching is bit-identical. The returned array is SHARED/read-only — clone before mutating.
    private double[] TokenOf(Element e) => e.TokenVector ??= FaceCodec.Token(e.Symbol, _dim);

    private double[] TokenOf(string symbol)
        => _concepts.TryGet(symbol, out var e) ? TokenOf(e) : FaceCodec.Token(symbol, _dim);

    private void RecomputeCloud(Element e)
    {
        var cloud = (double[])TokenOf(e).Clone(); // self-token, weight 1 (clone — the cached token is shared/read-only)
        if (_adjacency.TryGetValue(e.Symbol, out var nbrs))
        {
            foreach (var n in nbrs)
            {
                var aff = 1.0 - 2.0 * Clamp01(GetContradiction(e.Symbol, n));
                if (Math.Abs(aff) < 1e-9) continue;
                var t = TokenOf(n);                 // cached deterministic token (bit-identical to FaceCodec.Token)
                for (var i = 0; i < _semLen; i++) cloud[i] += aff * t[i];
            }
        }
        var dst = e.SemanticFace;
        for (var i = 0; i < _semLen && i < cloud.Length && i < dst.Length; i++) dst[i] = cloud[i];
        _lattice.MarkEmbeddingsDirty(); // the semantic face moved → drift toward a VP-tree rebuild
    }

    // BATCHED-GPU PATH (BatchedCloudGpu). Recompute every DIRTY concept's cloud in one GpuCloudBatcher op, writing the raw
    // (unnormalized) cloud back into each element's SemanticFace — identical to RecomputeCloud's definition up to float32.
    private void FlushClouds()
    {
        _obsSinceFlush = 0;
        if (_dirtyCloud.Count == 0) return;
        _batcher ??= new GpuCloudBatcher(_semLen, preferCuda: true);
        var selves = new List<string>(_dirtyCloud.Count);
        foreach (var s in _dirtyCloud)
            if (_concepts.TryGet(s, out var e) && !e.Archived) selves.Add(s);
        _dirtyCloud.Clear();
        if (selves.Count == 0) return;
        _batcher.Flush(selves, TokenOf, EntriesFor, (r, cloud) =>
        {
            if (!_concepts.TryGet(selves[r], out var e)) return;
            var dst = e.SemanticFace;
            for (var i = 0; i < _semLen && i < cloud.Length && i < dst.Length; i++) dst[i] = cloud[i];
        });
        _lattice.MarkEmbeddingsDirty();
    }

    // A concept's neighbour entries for the batched recompute: (neighbour, affinity) — the SAME affinity the scalar path uses.
    private IReadOnlyList<(string nbr, double weight)> EntriesFor(string self)
    {
        if (!_adjacency.TryGetValue(self, out var nbrs) || nbrs.Count == 0) return Array.Empty<(string, double)>();
        var list = new List<(string, double)>(nbrs.Count);
        foreach (var n in nbrs) list.Add((n, 1.0 - 2.0 * Clamp01(GetContradiction(self, n))));
        return list;
    }

    // Read paths call this first so retrieval/relaxation never sees a stale deferred cloud.
    private void EnsureCloudsFresh() { if (BatchedCloudGpu && _dirtyCloud.Count > 0) FlushClouds(); }

    /// <summary>Force any deferred (batched-GPU) cloud recomputes to complete now — call before a checkpoint/inspect snapshot,
    /// or to fold the final partial batch into a measurement. No-op unless <see cref="BatchedCloudGpu"/> is on with work pending.</summary>
    public void FlushCloudBatch() => EnsureCloudsFresh();

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

    /// <summary>The relation degree counting ONLY STRONG edges (relation Strength ≥ <paramref name="minStrength"/>) — the
    /// substrate's is-a-parent signal that IGNORES weak edges. The trainer's discriminative-coupling REPELS a leaf's
    /// nearest distractors (high-contradiction ⇒ Strength≈0.1 edges), which inflates raw <see cref="GetRelationDegree"/>
    /// and eventually makes a leaf look like a hub — breaking a degree-based ancestor climb. Ranking the climb by STRONG
    /// degree (planted/reinforced is-a edges have Strength≈1.0) makes the taxonomy robust to that observe-path noise.</summary>
    public int StrongRelationDegree(string concept, double minStrength = 0.5)
    {
        var c = Normalize(concept);
        if (!_adjacency.TryGetValue(c, out var adj)) return 0;
        var n = 0;
        foreach (var nb in adj)
            if (_relations.TryGetValue(Key(c, nb), out var r) && r.Strength >= minStrength) n++;
        return n;
    }

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

    // Below this strength a thoroughly-contradicted CUE→anchor routing relation is DROPPED (not just weakened): a
    // routing cue is BINARY at use (a token either selects a route or not — see ResolveLearnedIntent/ResolveLearnedOp,
    // which read the TOP relation regardless of absolute strength), so merely lowering strength can never stop a bad
    // cue from hijacking. Dropping the edge is the only thing that flips routing — the world UNLEARNS the association.
    private const double CueDropStrength = 0.2;

    /// <summary>
    /// SELF-HEAL a mis-firing routing CUE (the substrate half of "learn from a wrong route"). A learned cue→anchor
    /// relation (e.g. an operator symbol "-" wrongly related to the compare anchor "∘cmp" by corpus contamination)
    /// that drove a WRONG answer is CONTRADICTED here; once it is essentially all-contradiction (strength ≤
    /// <see cref="CueDropStrength"/>) the edge is DROPPED so the cue stops selecting that route. It re-forms only if
    /// genuinely re-observed. Gradual (one contradiction per wrong outcome) so a single mis-grade can't nuke a good
    /// cue; a persistently-wrong cue is unlearned over a few outcomes. Returns true when the edge was dropped. No-op
    /// if the relation is absent (nothing to unlearn). See <see cref="DisruptAssociation"/> (the orbital-repel sibling).
    /// </summary>
    public bool DisruptCueRelation(string cue, string anchor, double perHitContradiction = 0.25)
    {
        var a = Normalize(cue); var b = Normalize(anchor);
        if (!_relations.TryGetValue(Key(a, b), out var rel)) return false;
        rel.FailureCount++;
        rel.Synthesis = Clamp01(rel.Synthesis + Math.Max(0.0, perHitContradiction));
        rel.LastObserved = rel.Synthesis;
        if (rel.Strength <= CueDropStrength) { DropRelation(rel); return true; }
        return false;
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
            rel.LastUsedStep = _observeStep; // recency-of-USE — keeps a useful edge fresh (drives the relevance grace)
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
        EnsureCloudsFresh();
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
                // RETRIEVAL-MISS RECOVERY (gated, conservative): the anchor has NO active element. If its FROZEN
                // coordinate decodes to a CONFIDENT VALID identity (an evicted/latent concept), re-materialise it so we
                // relax from its reconstructed MEANING (near its family) rather than the bare token. Guarded inside
                // RecoverFromCoordinate (confident decode + frozen round-trip only) so a non-decoding anchor — or any
                // number/op — falls through to the raw-token path unchanged. Default-off = byte-identical.
                var rec = RecoverFromVoid && !FaceCodec.IsNumeric(key) ? RecoverFromCoordinate(FrozenAddressOf(key)) : null;
                if (rec is not null)
                {
                    var cloud = CloudOf(rec);
                    for (var i = 0; i < _semLen; i++) q[i] += cloud[i];
                }
                else
                {
                    var t = FaceCodec.Token(key, _dim);
                    for (var i = 0; i < _semLen; i++) q[i] += t[i];
                }
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

    /// <summary>
    /// MEANING-TRANSFORM / ANALOGY in the LARGE (word) face — generative reasoning over the distributional cloud, NOT
    /// the numeric homomorphism. From example pairs it forms the relation vector Δ = avg(cloud(to) − cloud(from)) and
    /// applies it to the query's MEANING: target = cloud(query) + Δ, then retrieves the nearest concept. "Paris is to
    /// France as Tokyo is to ___", as vector arithmetic in meaning space — the same T(f) the numeric faces use, on the
    /// large face. Endpoints + query must be known concepts (they need an accumulated cloud).
    /// </summary>
    public Thought Analogy(IReadOnlyList<(string From, string To)> pairs, string query, double settleThreshold = 0.3)
    {
        if (pairs is null || pairs.Count == 0) return new Thought(string.Empty, 0.0, false, 0);
        var delta = new double[_semLen];
        var n = 0;
        var exclude = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (from, to) in pairs)
        {
            exclude.Add(Normalize(from)); exclude.Add(Normalize(to));
            var cf = SemanticVectorOf(from); var ct = SemanticVectorOf(to);
            if (cf is null || ct is null) continue;
            for (var i = 0; i < _semLen; i++) delta[i] += ct[i] - cf[i];
            n++;
        }
        var cq = SemanticVectorOf(query);
        if (n == 0 || cq is null) return new Thought(string.Empty, 0.0, false, 0);
        var target = new double[_semLen];
        for (var i = 0; i < _semLen; i++) target[i] = cq[i] + delta[i] / n;
        if (!NormalizeVec(target)) return new Thought(string.Empty, 0.0, false, 0);
        exclude.Add(Normalize(query));

        var best = string.Empty; var bestSim = double.NegativeInfinity;
        void Consider(Element e)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || FaceCodec.IsNumeric(e.Symbol)) return;
            if (IsReservedConcept(e.Symbol) || IsOperationToken(e.Symbol) || exclude.Contains(e.Symbol)) return;
            var sim = Dot(target, CloudOf(e));
            if (sim > bestSim) { bestSim = sim; best = e.Symbol; }
        }
        // The analogy answer is the nearest concept to `target` in meaning space — a kNN query, so use the VP-tree
        // (O(log N)) instead of scanning the whole space, then LIVE-rescore the harvested candidates (tree distances
        // discarded). Same hybrid as GetNearestConcepts/Reason: below LatticeMinNodes the exact scan is cheap and fresh.
        if (_concepts.ActiveCount >= LatticeMinNodes)
        {
            var tFull = new double[_dim];
            for (var i = 0; i < _semLen && _semStart + i < _dim; i++) tFull[_semStart + i] = target[i];
            foreach (var (sym, _) in _lattice.GetSemanticNeighbors(tFull, 64, null))
                if (_concepts.TryGet(sym, out var e)) Consider(e);
        }
        else
        {
            foreach (var e in _concepts.All) Consider(e);
        }
        return new Thought(best, Math.Clamp(bestSim, 0.0, 1.0), bestSim >= settleThreshold, 1);
    }

    // Reserved internal symbols (the codec's face anchors, "∴"-prefixed reflexive elements, and "∘"-prefixed operation
    // anchors that learned cue→op relations point at) — observed/related by the mind, never retrieved as an answer.
    private static bool IsReservedConcept(string s)
        => s.StartsWith("face:", StringComparison.Ordinal) || s.StartsWith("∴", StringComparison.Ordinal)
        || s.StartsWith("∘", StringComparison.Ordinal) || s.StartsWith("⟨", StringComparison.Ordinal); // ⟨ = a Merge element
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
        EnsureCloudsFresh();
        var key = Normalize(concept);
        if (IsOperationToken(key) || FaceCodec.IsNumeric(key)) return null;
        return _concepts.TryGet(key, out var e) && !e.Archived && e.Kind != ElementKind.Atom ? CloudOf(e) : null;
    }

    /// <summary>The width of a semantic-face vector (<see cref="SemanticVectorOf"/> / the self-field) — so the mind can
    /// allocate its persistent self in the same space the field reasons in.</summary>
    public int SemanticLength => _semLen;
    private double Dot(double[] a, double[] b) { var d = 0.0; for (var i = 0; i < _semLen; i++) d += a[i] * b[i]; return d; }
    private bool NormalizeVec(double[] v) { var s = 0.0; for (var i = 0; i < _semLen; i++) s += v[i] * v[i]; s = Math.Sqrt(s); if (s <= 1e-9) return false; for (var i = 0; i < _semLen; i++) v[i] /= s; return true; }

    // ── LEARNED function-word signal (replaces the hardcoded stopword list). A function/framing word (the/of/for/with…)
    //    co-occurs with EVERYTHING, so its meaning-CLOUD averages toward the global centroid; a content word points at
    //    its own cluster, AWAY from the centroid — even a POPULAR content word, which is exactly where relation-DEGREE
    //    fails (it conflates a high-degree hub like "fruit" with filler). So "function-likeness" = how CENTRAL a cloud
    //    is. Measured (FunctionWordResearch) in a gym-warmed space: function 0.70–0.82 vs content 0.0–0.42, a clean gap
    //    once training spreads function words across many contexts. GYM SKILLS ARE WARM-START — the field abilities that
    //    consult this only ever run in a trained space, so the distribution exists. In a cold/tiny or low-variance space
    //    the signal SELF-ABSTAINS (returns false → nothing is filtered), so it can never over-filter while bootstrapping.
    private double _fnMean, _fnStd; // distribution of NEIGHBOUR-COHERENCE over sufficiently-connected concepts
    private double _fnOtsu;         // VALLEY cut between the glue mode and the content mode (Otsu's between-class split)
    private int _fnStamp = -1;
    private bool _fnReady;
    private const int FnMinWarm = 48;     // below this the body's distribution isn't real yet — don't filter anything
    private const int FnMinDegree = 6;    // need ≥ this many neighbours to read diversity reliably
    private const int FnMinSamples = 8;   // need ≥ this many qualifying concepts to read a 2-class split
    private const double FnCohCeil = 0.80;// absolute guard: a function word's neighbours are GENUINELY diverse
    // RETENTION threshold (G6 conservation). A word becomes CONSERVED-function once it has been OBSERVED bridging (live
    // coherence ≤ the Otsu cut, on a warm real distribution) on at least this many INDEPENDENT stats recomputations. Two
    // is the smallest count that requires CONFIRMATION: a single reading can be a transient during graph churn, but a
    // distinction seen twice across independent recomputes is real — and two is cheap to reach in a warm/trained space
    // (stats recompute repeatedly as the graph grows). Once crossed, the classification is RETAINED for good (G6) even
    // as the live coherence later drifts ABOVE the cut — growth only ADDS edges, it can never UNMAKE the distinction.
    private const double FnEvidenceRetain = 2.0;
    // Evidence accrues only on a CONFIDENT bridging reading — a clear low-coherence OUTLIER (coh ≤ mean − σ), not a
    // borderline word hovering near the body mean. This is what keeps a noisy/immature early graph (where content
    // clusters haven't formed yet, so content coherence is transiently low) from PERMANENTLY establishing a content
    // word: a genuine function word is a strong low outlier on essentially every mature recompute and crosses retention
    // easily, while a content word's occasional near-the-mean dip never qualifies. Conservation without over-fixation.
    private const double FnEvidenceSigma = 1.0;

    // NEIGHBOURHOOD CLUSTERING (the (b) signal, measured on the GRAPH not the clouds): of a concept's neighbours, what
    // fraction of PAIRS are themselves connected? A CONTENT word / category hub's neighbours are RELATED (same cluster →
    // connected to each other → HIGH clustering); a FUNCTION word bridges otherwise-UNRELATED words (its neighbours come
    // from many clusters and are NOT connected to each other → LOW clustering). This is immune to the cloud entanglement
    // that inverted the cloud-coherence metric (every content word shares a glue component). Low clustering + enough
    // neighbours ⇒ function-like. Bounded cost: sample up to 24 neighbours (≤276 pair lookups).
    private double NeighbourCoherence(string concept)
    {
        if (!_adjacency.TryGetValue(Normalize(concept), out var nbrs) || nbrs.Count < 2) return 1.0;
        var list = nbrs.Count <= 24 ? new List<string>(nbrs) : nbrs.Take(24).ToList();
        int links = 0, pairs = 0;
        for (var i = 0; i < list.Count; i++)
            for (var j = i + 1; j < list.Count; j++)
            {
                pairs++;
                if (_adjacency.TryGetValue(list[i], out var ni) && ni.Contains(list[j])) links++;
            }
        return pairs == 0 ? 1.0 : (double)links / pairs; // ∈ [0,1]: 1 = neighbours all interconnected, → 0 = a bridge
    }

    private void EnsureFunctionStats()
    {
        var count = _concepts.ActiveCount;
        if (_fnReady && Math.Abs(count - _fnStamp) < Math.Max(16, count / 16)) return; // amortize the O(N·deg) rebuild
        double sum = 0, sumSq = 0; var n = 0;
        var vals = new List<double>();
        var samples = new List<(Element E, double Coh)>(); // keep the elements so we can RECORD evidence below
        foreach (var e in _concepts.All)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || FaceCodec.IsNumeric(e.Symbol)) continue;
            if (GetRelationDegree(e.Symbol) < FnMinDegree) continue; // only words with enough neighbours to judge diversity
            var coh = NeighbourCoherence(e.Symbol);
            sum += coh; sumSq += coh * coh; n++;
            vals.Add(coh);
            samples.Add((e, coh));
        }
        var nn = Math.Max(1, n);
        _fnMean = sum / nn;
        _fnStd = Math.Sqrt(Math.Max(0.0, sumSq / nn - _fnMean * _fnMean));
        _fnOtsu = OtsuValleyThreshold(vals); // the VALLEY between the glue mode and the content mode (tracks the bimodal split as the graph grows)
        _fnStamp = count; _fnReady = n >= FnMinSamples;

        // G6 GENERATE-BY-OBSERVATION → RETAIN. The live coherence reading is now only the EVIDENCE SOURCE (an OBSERVATION
        // of bridging, G3). On a WARM, real distribution (enough warmth + a found valley), a word that reads as a CONFIDENT
        // bridge — below the Otsu valley, below the diversity ceiling, AND a clear low OUTLIER (≤ mean − σ) — ACCUMULATES
        // one unit of conserved FunctionEvidence. Monotonic — only ever increases (G6) — so the distinction, once accrued,
        // is never re-measured away when the graph's later growth drifts the live coherence above the cut. The σ-outlier
        // bar is what prevents an immature early graph from PERMANENTLY establishing a content word (whose dips are
        // near-the-mean, not deep in the glue tail), so conservation never hardens an early misclassification.
        if (_fnReady && _fnOtsu > 0.0 && count >= FnMinWarm)
        {
            var outlierCut = _fnMean - FnEvidenceSigma * _fnStd; // a CONFIDENT low outlier, not a borderline near-body dip
            foreach (var (e, coh) in samples)
                if (coh <= _fnOtsu && coh <= FnCohCeil && coh <= outlierCut)
                    e.FunctionEvidence += 1.0;
        }
    }

    // OTSU'S METHOD over the coherence histogram: the cut that maximizes BETWEEN-class variance (equivalently minimizes
    // within-class variance) — it lands in the natural VALLEY between the low-coherence GLUE mode and the higher-coherence
    // CONTENT mode. This is the right estimator for a skewed/bimodal-with-tail distribution where a mean−σ cut collapses
    // INTO the low tail as the graph grows (the tail fattens, the mean sinks, the σ-cut sinks with it → the separation
    // degrades over time). Otsu instead tracks the valley wherever it sits, so it does NOT drift into the tail at scale.
    private static double OtsuValleyThreshold(List<double> values)
    {
        if (values.Count < FnMinSamples) return 0.0; // too few to read a split → 0 filters nothing (self-abstain)
        const int Bins = 64;
        var hist = new int[Bins];
        foreach (var v in values) hist[(int)(Math.Clamp(v, 0.0, 0.9999999) * Bins)]++;
        int total = values.Count;
        double sumAll = 0;
        for (int i = 0; i < Bins; i++) sumAll += (i + 0.5) / Bins * hist[i];
        double sumB = 0; int wB = 0; double maxVar = -1; int bestBin = -1;
        for (int t = 0; t < Bins; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break; // no content class above the cut → stop
            sumB += (t + 0.5) / Bins * hist[t];
            double mB = sumB / wB, mF = (sumAll - sumB) / wF;
            double between = (double)wB * wF * (mB - mF) * (mB - mF); // between-class variance (×total², monotone proxy)
            if (between > maxVar) { maxVar = between; bestBin = t; }
        }
        return bestBin < 0 ? 0.0 : (bestBin + 1.0) / Bins; // upper edge of the chosen bin = the cut: coh ≤ this ⇒ glue/function
    }

    /// <summary>Is this concept a FILLER / function word — does it co-occur with MANY MUTUALLY-UNRELATED words (low
    /// neighbour-coherence), as opposed to a content word (even a category hub) whose neighbours are related? Learned
    /// from the live distribution, NOT a hardcoded list. Self-abstains (false) cold / with too few neighbours.</summary>
    public bool IsFunctionLike(string concept)
    {
        if (_concepts.ActiveCount < FnMinWarm) return false;
        var key = Normalize(concept);
        if (GetRelationDegree(key) < FnMinDegree) return false; // needs enough neighbours for a real diversity reading
        if (DistributionalFnWord)
        {
            // DISTRIBUTIONAL signal (PMI): a function word co-occurs with everything BY FREQUENCY (weak associations,
            // PMI≈0); content co-occurs SELECTIVELY with its cluster (a strong PMI). So a low strongest-association = a
            // function word — degree-robust where the graph clustering estimator is noisy (e.g. possessives).
            EnsureAssocStats();
            if (!_assocReady || _assocStd < 1e-6) return false;
            return AssociationStrength(key) <= _assocMean - AssocSigma * _assocStd; // a LOW-association outlier
        }
        EnsureFunctionStats(); // amortized — also ACCRUES the conserved FunctionEvidence below
        // CONSERVED classification (G6): once a word has accrued enough bridging evidence it is RETAINED as function-like
        // FOREVER — regardless of how its live coherence later drifts ABOVE the cut as the graph only ADDS edges (a
        // distinction once made is never unmade). This is the fix for "the prebake gets worse over time": the live reading
        // is re-measured and DRIFTS, but the world-knowledge it generated is conserved.
        if (_concepts.TryGet(key, out var ke) && ke.FunctionEvidence >= FnEvidenceRetain) return true;
        if (!_fnReady || _fnOtsu <= 0.0) return false; // no valley found (cold / unimodal / too few) → filter nothing
        // LIVE evidence path — a currently-bridging word reads function-like immediately (and is accruing toward retention),
        // so this never regresses behaviour before the conserved threshold is reached; it only ever ADDS a classification.
        var coh = NeighbourCoherence(key);
        return coh <= _fnOtsu && coh <= FnCohCeil; // below the bimodal VALLEY (and genuinely diverse) ⇒ function-like
    }

    /// <summary>DIAGNOSTIC: the NEIGHBOUR-COHERENCE of a concept (1=aligned/content, →0=diverse/function) and the active
    /// thresholds — so a curriculum/test can SEE why IsFunctionLike fires. Tuple reused: Centrality=coherence,
    /// Threshold=the Otsu VALLEY cut (function if coherence ≤ it), Floor=the coherence ceiling, MinWarm=degree.
    /// Evidence=the CONSERVED FunctionEvidence (function-like once it ≥ the retention threshold, regardless of live drift).</summary>
    public (double Centrality, double Mean, double Std, double Threshold, double Floor, int Active, int MinWarm, double Evidence) FunctionStats(string concept)
    {
        EnsureFunctionStats();
        var key = Normalize(concept);
        var evidence = _concepts.TryGet(key, out var e) ? e.FunctionEvidence : 0.0;
        return (NeighbourCoherence(key), _fnMean, _fnStd, _fnOtsu, FnCohCeil, _concepts.ActiveCount, GetRelationDegree(key), evidence);
    }

    // ── DISTRIBUTIONAL function-word signal (theory #2): PMI from the co-occurrence COUNTS, not graph topology ────────
    // A function word co-occurs with everything roughly BY CHANCE (count(a,b) ≈ obs(a)·obs(b)/N ⇒ PMI ≈ 0); a content
    // word co-occurs SELECTIVELY with its cluster (count ≫ chance ⇒ high PMI). So a concept's STRONGEST associations are
    // weak for a function word, strong for content — degree-robust (PMI normalises by frequency), where the graph
    // clustering estimator gets noisy on low-degree words (possessives). Gated by DistributionalFnWord (default off).
    public bool DistributionalFnWord { get; set; }
    private const double AssocSigma = 0.5; // function = a LOW strongest-association OUTLIER below the body's mean
    private double _assocMean, _assocStd; private int _assocStamp = -1; private bool _assocReady;

    /// <summary>The mean of a concept's TOP-3 PMI associations — how STRONGLY/selectively it co-occurs with its partners.
    /// High = content (a real cluster bond); low = function (it co-occurs by frequency, no selective bond). 0 if too sparse.</summary>
    public double AssociationStrength(string concept)
    {
        var key = Normalize(concept);
        if (!_adjacency.TryGetValue(key, out var nbrs) || nbrs.Count < 2 || !_concepts.TryGet(key, out var ea)) return 0.0;
        var n = (double)Math.Max(1, _observeStep);
        var oa = (double)Math.Max(1, ea.ObservationCount);
        var pmis = new List<double>(nbrs.Count);
        foreach (var nb in nbrs)
        {
            if (!_relations.TryGetValue(Key(key, nb), out var r) || r.ObservationCount <= 0) continue;
            if (!_concepts.TryGet(nb, out var eb)) continue;
            pmis.Add(Math.Log((r.ObservationCount * n) / (oa * Math.Max(1, eb.ObservationCount)))); // PMI(a,b)
        }
        if (pmis.Count == 0) return 0.0;
        pmis.Sort();
        var k = Math.Min(3, pmis.Count);                          // the STRONGEST associations (robust max)
        double s = 0; for (var i = pmis.Count - k; i < pmis.Count; i++) s += pmis[i];
        return s / k;
    }

    /// <summary>DIAGNOSTIC (Inspect tab): a concept's strongest-association strength + the population threshold, so the
    /// PMI verdict (function if Assoc ≤ Threshold) can be shown SIDE-BY-SIDE with the graph metric on the same space.</summary>
    public (double Assoc, double Mean, double Std, double Threshold) AssociationStats(string concept)
    {
        EnsureAssocStats();
        return (AssociationStrength(Normalize(concept)), _assocMean, _assocStd, _assocMean - AssocSigma * _assocStd);
    }

    private void EnsureAssocStats()
    {
        var count = _concepts.ActiveCount;
        if (_assocReady && Math.Abs(count - _assocStamp) < Math.Max(16, count / 16)) return;
        double sum = 0, sumSq = 0; var n = 0;
        foreach (var e in _concepts.All)
        {
            if (e.Archived || e.Kind == ElementKind.Atom || FaceCodec.IsNumeric(e.Symbol)) continue;
            if (GetRelationDegree(e.Symbol) < FnMinDegree) continue;
            var a = AssociationStrength(e.Symbol);
            sum += a; sumSq += a * a; n++;
        }
        var nn = Math.Max(1, n);
        _assocMean = sum / nn;
        _assocStd = Math.Sqrt(Math.Max(0.0, sumSq / nn - _assocMean * _assocMean));
        _assocStamp = count; _assocReady = n > 0;
    }

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
        var e = GetOrCreateConcept(key);   // compose-and-store: builds the hub over its components (and any novel parts)
        if (GenerativeAtoms && !HasWhitespace(key) && known > 0)
            ComposeMeaningFromKnownParts(key, e); // GENERALISE: position the novel token from its known morphemes
        var conf = total > 0 ? (double)known / total : 0.5;
        return new Recognition(key, conf, WholeHit: false, known, total);
    }

    // GENERALISATION (the payoff of generative atoms): a novel token's meaning is the BLEND of its KNOWN MORPHEMES' clouds,
    // so "helix" lands near the words that share "hel" — recognised though never seen. Only reinforced multi-char morphemes
    // contribute (single chars are near-centroid noise; the competition already filtered the useless chunks).
    private void ComposeMeaningFromKnownParts(string key, Element e)
    {
        var acc = new double[_semLen]; var n = 0;
        foreach (var g in CandidateGrams(key))
        {
            if (g.Length < 2 || string.Equals(g, key, StringComparison.Ordinal)) continue;
            if (!_concepts.TryGet(g, out var part) || part.Archived || part.Kind == ElementKind.Atom) continue;
            var c = CloudOf(part);
            for (var i = 0; i < _semLen && i < c.Length; i++) acc[i] += c[i];
            n++;
        }
        if (n == 0) return;
        var dst = e.SemanticFace;
        for (var i = 0; i < _semLen && i < dst.Length; i++) dst[i] = acc[i] / n;
        _lattice.MarkEmbeddingsDirty();
    }

    /// <summary>How many of an input's ▷ components are already KNOWN (text→words, word→char atoms).</summary>
    private (int Known, int Total) ComponentCoverage(string key)
    {
        if (HasWhitespace(key))
        {
            var words = key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            return (words.Count(w => _concepts.Contains(w)), words.Length);
        }
        if (GenerativeAtoms)
        {
            // Coverage by KNOWN MORPHEMES (the surviving n-grams ≥2) — single chars are uninformative (near-centroid).
            var grams = CandidateGrams(key).Where(g => g.Length >= 2).ToList();
            return grams.Count == 0 ? (0, 0) : (grams.Count(_concepts.Contains), grams.Count);
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

    // ──────────────────────────────────────────────────────────── Maintenance — discharge by relevance + enforce the cap
    /// <summary>
    /// The LIVE/per-epoch maintenance path: release stale noise by relevance-decay AND enforce the hard
    /// <see cref="MaxActiveConcepts"/> ceiling (both run inside <see cref="DischargeIrrelevant"/>). Decay only touches
    /// barely-observed, stale concepts (an in-progress lesson and the reinforced function-word geometry stay safe); the
    /// cap then drops the lowest-utility overflow so a corpus stream that keeps everything "recently seen" still cannot
    /// grow the active set without bound. Anchors/atoms/Functions/numbers/referenced components are protected in both.
    /// </summary>
    public PlatonicSpaceMemory.SpaceMaintenanceResult ApplyMaintenance(PlatonicSpaceMemory.SpaceMaintenanceRequest request)
    {
        var before = _concepts.ActiveCount;
        DischargeIrrelevant();        // grace-decay prune + EnforceActiveConceptCap (the hard ceiling)
        _lastDischargeStep = _observeStep;
        return new(0, Math.Max(0, before - _concepts.ActiveCount), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────────── Snapshots (checkpoint compat)
    public PlatonicMemorySnapshot ExportSnapshot()
    {
        EnsureCloudsFresh(); // fold any deferred batched-GPU clouds into SemanticFace before snapshotting (else stale)
        // FAITHFUL: capture EVERY non-numeric element (INCLUDING atoms) with its learned ORBITAL + kind + lifecycle.
        // The orbital (semantic-face slice) is the ONLY mutable state; identity faces are codec-derived from the
        // symbol and ▷-components are re-derived from the symbol, so storing the orbital + kind round-trips exactly.
        // Numeric elements are codec-exact and never relate, so they're recreated on demand (omitted). Nodes are left
        // EMPTY — Elements supersedes the old lossy full-face node path (which only ever restored the orbital anyway).
        // Persist the ACTIVE set only — NOT archived (evicted) elements. In-session dormancy (G6 reactivation) keeps
        // archived concepts in the live store, but serializing the evicted long tail would re-bloat the checkpoint
        // unboundedly under corpus scale (the storage half of the blow-up) and defeat the eviction. An evicted concept
        // that recurs after a reload is simply re-created from observation — exactly as it was first learned.
        // Function elements (reserved operations) are OMITTED — they are codec-derived from their op token (no learned
        // orbital that matters) and recreated on import via RegisterOperationToken from the persisted OperationTokens
        // list, exactly like numeric elements are recreated on demand. Persisting them would also collide with that
        // recreate (a Function symbol round-tripped through GetOrCreateConcept would come back the wrong kind).
        var elements = _concepts.All
            .Where(e => !e.Archived && !FaceCodec.IsNumeric(e.Symbol) && e.Kind != ElementKind.Function)
            .Select(e => new DialecticalElementSnapshot(e.Symbol, (int)e.Kind, (double[])e.SemanticFace.Clone(), e.ObservationCount, e.Archived, e.FunctionEvidence))
            .ToArray();
        var rels = _relations.Values.Select(r => new PlatonicRelationSnapshot(
            r.Left, r.Right, r.Thesis, r.LastObserved, r.Synthesis, r.ObservationCount, r.UseCount, r.SuccessCount, r.FailureCount, r.LastUsedStep)).ToArray();
        var chunks = _chunks.SelectMany(t => t.Value.Select(c => new PlatonicChunkSnapshot(t.Key, c.Key, c.Value))).ToArray();
        var numberWords = NumberWords.Export().Select(a => new NumberWordAtomSnapshot(a.Word, a.Value)).ToArray();
        return new PlatonicMemorySnapshot(_dim, Array.Empty<PlatonicNodeSnapshot>(), rels, chunks, _operationTokens.ToArray(),
            elements, numberWords, PlatonicMemorySnapshot.CurrentLayoutVersion); // stamp the address-space layout version (L2)
    }

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        if (snapshot == null) return;
        // LAYOUT VERSION GATE (B, BREAKING — fresh train, no migration). The orbital tail moved + shrank (310→96 at
        // dim 512), so a checkpoint stamped below CurrentLayoutVersion (old/absent = 0) holds layout-INCOMPATIBLE
        // element orbitals — writing them into the relocated [OrbitalStart,dim) tail would be mismatched garbage. When
        // incompatible we SKIP element orbitals (re-learned fresh) but still restore the layout-INDEPENDENT data
        // (element identity + counters, relations, chunks, op-tokens, number-words). No migration is attempted.
        var layoutCompatible = snapshot.LayoutVersion >= PlatonicMemorySnapshot.CurrentLayoutVersion;
        if (snapshot.Elements is { Length: > 0 })
        {
            // FAITHFUL path: recreate each element through the SAME creation φ the live space uses (so kind + ▷-
            // components match exactly), then overwrite its orbital + counters. GetOrCreate is idempotent, so atom
            // entries and the concepts that reference them converge regardless of order.
            foreach (var el in snapshot.Elements)
            {
                var key = Normalize(el.Symbol);
                if (FaceCodec.IsNumeric(key)) continue;
                // A MERGE element ⟨a·label·b⟩ (Kind.Relation) is NOT re-derivable by GetOrCreateConcept — that parses the
                // bracket as a plain token (empty ▷ Components, wrong kind), so a persisted FACT (⟨key·is·value⟩) would
                // reload component-less and TryRecallFact would fail (recall walks the Components). REBUILD it structurally
                // from its content-addressed symbol via Merge so the binding (and any nested subtree) is restored exactly.
                var e = (ElementKind)el.Kind == ElementKind.Atom
                    ? _concepts.GetOrCreate(el.Symbol, ElementKind.Atom, () => new double[_semLen])
                    : el.Symbol.StartsWith("⟨", StringComparison.Ordinal)
                        ? _concepts.TryGet(ReconstructMerge(el.Symbol), out var me) ? me : GetOrCreateConcept(el.Symbol)
                        : GetOrCreateConcept(el.Symbol);
                // Orbital is the ONLY layout-dependent state — copy it ONLY when the checkpoint's layout matches the
                // current band layout AND the stored width matches this dim's SemanticLength (defensive); else ignore it.
                if (layoutCompatible && el.Orbital is { Length: > 0 } && el.Orbital.Length == _semLen)
                    for (var i = 0; i < _semLen && i < el.Orbital.Length && i < e.SemanticFace.Length; i++)
                        e.SemanticFace[i] = el.Orbital[i];
                e.ObservationCount = el.ObservationCount;
                // CONSERVED world-knowledge — layout-INDEPENDENT (a scalar count), so restore it UNCONDITIONALLY (unlike
                // the orbital, which is dropped on a layout mismatch). Max-merge so a re-observed element never LOSES
                // accrued evidence (G6 monotonic): the reload only ever raises it.
                e.FunctionEvidence = Math.Max(e.FunctionEvidence, el.FunctionEvidence);
                if (el.Archived && !e.Archived) _concepts.Archive(el.Symbol); // restore G6 dormancy (keeps ActiveCount correct)
            }
        }
        else foreach (var n in snapshot.Nodes ?? Array.Empty<PlatonicNodeSnapshot>()) // LEGACY checkpoints (no Elements)
        {
            if (FaceCodec.IsNumeric(Normalize(n.Name))) continue;
            var e = GetOrCreateConcept(n.Name);
            e.ObservationCount = n.ObservationCount;
            // The full-face slice is layout-dependent (it was sliced at the OLD SemanticStart); restore it only when
            // the layout matches — legacy node checkpoints are always pre-address-space (version 0) → orbital dropped.
            if (layoutCompatible && n.PositiveFace is { Length: > 0 }) // restore the orbital slice
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
        if (snapshot.NumberWords is { Length: > 0 } nw)
            NumberWords.Import(nw.Select(a => (a.Word, a.Value)));
        _lattice.MarkEmbeddingsDirty(); // imported orbitals were written directly → force a rebuild from live faces
    }
}
