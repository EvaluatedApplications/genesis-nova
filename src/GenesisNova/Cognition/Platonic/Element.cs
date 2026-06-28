namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// The kind of a platonic element (PLATONIC_THEORY.md §1, genesis 02-PLATONIC-EMERGENCE). Atoms are
/// irreducible (characters, digit-places); the rest are built. The new dialectical core is built around this
/// emergent-kind ontology rather than a flat bag of stamped vectors.
/// </summary>
public enum ElementKind
{
    /// <summary>Irreducible, reusable base: a single character, a digit place. The bounded set every composite reuses.</summary>
    Atom,
    /// <summary>A concept / vocabulary point — defined by its differences (κ) to everything else (Law M).</summary>
    Object,
    /// <summary>A positioned link between elements (κ web). Itself an element (G5 — can be related/composed).</summary>
    Relation,
    /// <summary>A transform/operation as a first-class element.</summary>
    Function,
    /// <summary>A hub over reused components (▷ part-of); position derived from its parts (Laws C/S).</summary>
    Composition,
}

/// <summary>
/// One element of the platonic space Π = (E, κind, π, ¬, κ, ▷) — see PLATONIC_THEORY.md.
///
/// Realizes the genesis DUAL-FACE invariant (03-SYMMETRY-BRIDGE Corr.7, 10-UNIFIED): an element is an immutable
/// IDENTITY (the "nucleus" — its arithmetic/char faces, derived verbatim from the kept codec, never stored here so
/// it can never drift) bound by conservation (G4) to an updatable SEMANTIC face (the "orbital" — the only mutable
/// state, born NEUTRAL and settling from observed contradiction, Law D). We therefore store ONLY the orbital plus
/// the structural part-of edges; the immutable identity is recomputed from <see cref="Symbol"/> on demand.
///
/// G6 (irreversibility): an element is never destroyed — <see cref="Archived"/> dims it instead, so a distinction,
/// once made, is retained (and can be reactivated). The complement ¬ (G4) is the exact negation of the assembled
/// face (NegativeFace = −PositiveFace), enforced at assembly time, not stored redundantly.
/// </summary>
public sealed class Element
{
    public Element(int id, ElementKind kind, string symbol, double[] semanticFace, int[]? components = null)
    {
        Id = id;
        Kind = kind;
        Symbol = symbol;
        SemanticFace = semanticFace;
        Components = components ?? System.Array.Empty<int>();
    }

    /// <summary>Stable identity within the store (monotone — assigned once, never reused; G6).</summary>
    public int Id { get; }

    public ElementKind Kind { get; }

    /// <summary>Canonical key. The IMMUTABLE identity (arithmetic + char faces) is derived from this via the kept
    /// codec — never stored, so it cannot drift (the "nucleus is fixed by element identity").</summary>
    public string Symbol { get; }

    /// <summary>The mutable ORBITAL — the semantic/free face region only. Born neutral; settles from κ (Law D).
    /// This is the ONLY learned state on an element; identity is codec-derived, structure is in <see cref="Components"/>.</summary>
    public double[] SemanticFace { get; set; }

    /// <summary>▷ part-of: ids of the COMPONENT elements this one is a hub over (chars for a word, words for a
    /// text, digit-places for a value). Empty for atoms/objects. Distinct from the κ (contradiction) web — ▷ points
    /// DOWN to parts, κ points SIDEWAYS to meaning (PLATONIC_THEORY.md §4). Shared: one atom is referenced by many
    /// composites (reuse ⇒ bounded growth, Law S).</summary>
    public int[] Components { get; }

    /// <summary>G6: dormant rather than deleted. Archived elements are retained (reactivatable), never destroyed.</summary>
    public bool Archived { get; set; }

    /// <summary>How many times this element has participated in an observation (drives plasticity / lifecycle).</summary>
    public long ObservationCount { get; set; }

    /// <summary>The observation-step at which this element was last seen — its RECENCY. With <see cref="ObservationCount"/>
    /// it drives relevance-decay: a barely-observed element that has gone stale (not seen for a long window) is DISCHARGED
    /// as noise, while a reinforced or recently-active one is retained. Runtime signal — not persisted (decay restarts per
    /// session). Distinct from a size cap: the space holds as much RELEVANT structure as exists; only noise is released.</summary>
    public long LastSeenStep { get; set; }
}
