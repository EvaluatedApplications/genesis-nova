using System.Collections.Generic;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// The carrier E of Π (PLATONIC_THEORY.md §1): the monotone set of elements. Ids are handed out monotonically and
/// never reused. G6 (irreversibility) holds NOT because elements are never deleted — live eviction
/// (<see cref="Remove"/>) genuinely frees them — but because the address space is a LATENT coordinate system: an
/// element's identity is a deterministic decodable ADDRESS (a pure function of its symbol), so a deleted symbol
/// re-observed re-derives the EXACT same frozen face. Deletion only DE-MATERIALISES (frees working memory); the
/// distinction is conserved by the coordinate, not by the live entry. <see cref="Archive"/> (dormant) is NOT the
/// evictor — its only live use is restoring G6-dormancy on snapshot import.
///
/// This is deliberately a plain, law-shaped store: a symbol→element index (the O(1) access role <c>_nodes</c> played
/// in the old memory) plus a bounded atom registry (the reusable base for composition, Law S). It holds no geometry
/// and no force constants — positioning (Law D) and composition (Laws C/S) live in their own units and operate over
/// these elements. Not thread-safe; the space is driven from a single training/inference loop.
/// </summary>
public sealed class ElementStore
{
    private readonly Dictionary<string, Element> _bySymbol = new(System.StringComparer.Ordinal);
    // Keyed by monotone Id (never reused), NOT a List, so deeply-dormant elements can be PURGED to keep the store bounded
    // (see Remove). Ids stay dense-enough for ById; gaps from purges are fine (ById is a dictionary lookup).
    private readonly Dictionary<int, Element> _byId = new();
    private int _nextId;
    private int _activeCount;    // tracked incrementally so ActiveCount is O(1) — it gates per-query hot paths
    private int _activeConcepts; // active USER concepts (= NodeCount). EXCLUDES the bounded structural base (atoms) AND
                                 // reserved Functions (operations — a route over the address space, not a retrievable
                                 // concept), so registering an op never moves NodeCount. O(1), safe to read concurrently
                                 // from /status without enumerating the live store (which races with training).

    // A user CONCEPT for NodeCount purposes: not an atom (the reusable sub-lexical base) and not a Function (a reserved
    // operation element — first-class + decodable, but a route, never a semantic-retrieval target). Kept in one place so
    // every count mutation (create / reactivate / archive / remove) stays consistent.
    private static bool CountsAsConcept(ElementKind kind) => kind != ElementKind.Atom && kind != ElementKind.Function;

    /// <summary>Live (non-archived) element count. O(1) (maintained on create/reactivate/archive).</summary>
    public int ActiveCount => _activeCount;

    /// <summary>Live non-atom element count (= concept NodeCount). O(1) and concurrency-safe (a plain int read).</summary>
    public int ActiveConceptCount => _activeConcepts;

    /// <summary>Elements currently retained (active + archived-but-not-yet-purged). Bounded by the caller's archive budget.</summary>
    public int TotalCount => _byId.Count;

    /// <summary>The next Id to be assigned — i.e. one past the highest Id ever created. Monotone; survives purges (unlike
    /// <see cref="TotalCount"/>), so it is the correct basis for "most-recently-created" recency windows.</summary>
    public int NextId => _nextId;

    public IReadOnlyCollection<Element> All => _byId.Values;

    public bool TryGet(string symbol, out Element element) => _bySymbol.TryGetValue(symbol, out element!);

    /// <summary>Resolve an element by its monotone id (ids are dense and never reused, so id == insertion index).</summary>
    public Element ById(int id) => _byId[id];

    public bool Contains(string symbol) => _bySymbol.TryGetValue(symbol, out var e) && !e.Archived;

    /// <summary>
    /// Get an existing element by symbol or create it (G3 generative observation). A previously-archived element is
    /// REACTIVATED rather than duplicated (G6 — the distinction was never destroyed). The caller supplies the initial
    /// semantic (orbital) face — neutral for a fresh concept, so its position settles from κ (Law D), never stamped.
    /// </summary>
    public Element GetOrCreate(string symbol, ElementKind kind, System.Func<double[]> freshSemanticFace, int[]? components = null)
    {
        if (_bySymbol.TryGetValue(symbol, out var existing))
        {
            if (existing.Archived) { existing.Archived = false; _activeCount++; if (CountsAsConcept(existing.Kind)) _activeConcepts++; } // reactivate (G6)
            return existing;
        }

        var element = new Element(_nextId++, kind, symbol, freshSemanticFace(), components);
        _bySymbol[symbol] = element;
        _byId[element.Id] = element;
        _activeCount++;
        if (CountsAsConcept(kind)) _activeConcepts++;
        return element;
    }

    /// <summary>Mark an element G6-dormant (reactivatable via <see cref="GetOrCreate"/>). Its only live caller is
    /// snapshot import, restoring the dormancy an element had when it was saved — this is NOT the evictor
    /// (live eviction genuinely deletes via <see cref="Remove"/>).</summary>
    public void Archive(string symbol)
    {
        if (_bySymbol.TryGetValue(symbol, out var e) && !e.Archived)
        {
            e.Archived = true;
            _activeCount--;
            if (CountsAsConcept(e.Kind)) _activeConcepts--;
        }
    }

    /// <summary>
    /// The EVICTOR: permanently free an element so the store stays bounded under corpus-scale churn. The live caller
    /// (<c>DischargeIrrelevant → DischargeConcept</c>) evicts relevance-decayed concepts — never one still referenced as a
    /// ▷-component (that would break ▷ / ById). This DELETES the live entry, yet stays G6-faithful because the address
    /// space is a LATENT coordinate system: a removed symbol re-observed re-derives the EXACT same frozen face, so the
    /// delete only DE-MATERIALISES it (frees working memory + every O(TotalCount) scan), it does not unmake the
    /// distinction. The non-derivable learned tail (orbital + FunctionEvidence) re-accumulates on re-observation, exactly
    /// as first learned. Consistent with the checkpoint persisting the active set only.
    /// </summary>
    public void Remove(string symbol)
    {
        if (!_bySymbol.TryGetValue(symbol, out var e)) return;
        _bySymbol.Remove(symbol);
        _byId.Remove(e.Id);
        if (!e.Archived) { _activeCount--; if (CountsAsConcept(e.Kind)) _activeConcepts--; } // an evicted ACTIVE concept frees its slot
    }
}
