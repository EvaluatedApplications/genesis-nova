using System.Collections.Generic;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// The carrier E of Π (PLATONIC_THEORY.md §1): the monotone set of elements. G6 (irreversibility) is structural —
/// <see cref="GetOrCreate"/> only ever ADDS, and removal is <see cref="Archive"/> (dormant), never a delete. Ids are
/// handed out monotonically and never reused, so a distinction once made is permanent.
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

    /// <summary>G6: dim an element to dormant. It is retained (reactivatable via <see cref="GetOrCreate"/>), never deleted.</summary>
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
    /// PERMANENTLY forget a deeply-dormant element so the store stays bounded under corpus-scale churn. G6 keeps an
    /// evicted distinction reactivatable, but an effectively-infinite stream of once-seen tokens would otherwise grow the
    /// store (and every O(TotalCount) scan / the RAM footprint) without limit. The caller purges only the OLDEST archived
    /// elements beyond an archive budget, and never one still referenced as a ▷-component — so this is the bounded tail of
    /// dormancy, consistent with the checkpoint already persisting the active set only. A purged symbol re-creates fresh
    /// from observation, exactly as first learned.
    /// </summary>
    public void Remove(string symbol)
    {
        if (!_bySymbol.TryGetValue(symbol, out var e)) return;
        _bySymbol.Remove(symbol);
        _byId.Remove(e.Id);
        if (!e.Archived) { _activeCount--; if (CountsAsConcept(e.Kind)) _activeConcepts--; } // callers only purge archived; stay honest
    }
}
