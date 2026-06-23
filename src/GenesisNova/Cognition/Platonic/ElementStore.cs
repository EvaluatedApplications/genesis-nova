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
    private readonly List<Element> _byId = new();
    private int _nextId;

    /// <summary>Live (non-archived) element count.</summary>
    public int ActiveCount
    {
        get
        {
            var n = 0;
            foreach (var e in _byId)
                if (!e.Archived) n++;
            return n;
        }
    }

    /// <summary>Total elements ever created, including archived (G6 — the space only grows).</summary>
    public int TotalCount => _byId.Count;

    public IReadOnlyList<Element> All => _byId;

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
            existing.Archived = false; // reactivate (G6)
            return existing;
        }

        var element = new Element(_nextId++, kind, symbol, freshSemanticFace(), components);
        _bySymbol[symbol] = element;
        _byId.Add(element);
        return element;
    }

    /// <summary>G6: dim an element to dormant. It is retained (reactivatable via <see cref="GetOrCreate"/>), never deleted.</summary>
    public void Archive(string symbol)
    {
        if (_bySymbol.TryGetValue(symbol, out var e))
            e.Archived = true;
    }
}
