namespace GenesisNova.Core;

/// <summary>
/// Non-overlapping face layout constants for the platonic embedding vector.
/// Ported from the genesis-engine source of truth (Genesis.Shared.FaceLayout) so that
/// nova's face geometry conforms exactly to the canonical platonic space.
/// <para>
/// The embedding vector is partitioned into functional regions (faces):
/// <list type="bullet">
///   <item>Polynomial face: [0..NumericDims) — add/sub homomorphism, embed[i] = value × 10^-(i+1)</item>
///   <item>Logarithmic face: [NumericDims..PolyFaceMax) — mul/div homomorphism, embed[i] = ln(|value|) × 10^-(i+1)</item>
///   <item>Character face: [CharFaceStart..WordFaceStart) — clean per-char slots</item>
///   <item>Word face: [WordFaceStart..dim) — whole strings / phrases live here intact</item>
/// </list>
/// </para>
/// <para>
/// <c>PolyFaceMax</c> (42) is the total arithmetic face width: 21 poly dims + 21 log dims.
/// For dim &lt; 42, all semantic content is packed into [dim/2..dim) for compatibility.
/// The arithmetic faces pin their algebraic identity; every dimension OUTSIDE a face's
/// identity range is left free to "wiggle" (learnable).
/// </para>
/// </summary>
public static class FaceLayout
{
    /// <summary>
    /// Maximum combined width of the polynomial + logarithmic faces (21 + 21 = 42).
    /// Character-slot face begins at this offset when dim ≥ 84.
    /// </summary>
    public const int PolyFaceMax = 42;

    /// <summary>
    /// Number of dimensions per numeric face (polynomial or logarithmic).
    /// Formula: min(dim/2, 21). Each face gets up to 21 dims.
    /// </summary>
    public static int NumericDims(int dim) => Math.Min(dim / 2, 21);

    /// <summary>
    /// Start index of the polynomial face (always 0).
    /// </summary>
    public static int PolyFaceStart => 0;

    /// <summary>
    /// Start index of the logarithmic face: equals the polynomial width.
    /// </summary>
    public static int LogFaceStart(int dim) => NumericDims(dim);

    /// <summary>
    /// One-past-the-end of the arithmetic (poly + log) block. This entire block is the
    /// algebraic identity that must stay exact for the homomorphism to hold.
    /// </summary>
    public static int ArithmeticFaceEnd(int dim) => Math.Min(2 * NumericDims(dim), dim);

    /// <summary>
    /// Start index of the character-slot face.
    /// Fixed at <see cref="PolyFaceMax"/> (42) once dim ≥ 84, otherwise dim/2.
    /// For dim &lt; 42 all semantic content is packed into [dim/2..dim) for compat.
    /// </summary>
    public static int CharFaceStart(int dim) => Math.Min(PolyFaceMax, dim / 2);

    /// <summary>
    /// Start index of the word-slot face.
    /// Fixed at 202 once dim &gt; 202, otherwise equals dim (no word face).
    /// </summary>
    public static int WordFaceStart(int dim) => dim > 202 ? 202 : dim;

    /// <summary>
    /// Width of the character-slot face: from CharFaceStart up to WordFaceStart (capped at 160).
    /// </summary>
    public static int CharFaceDims(int dim) => Math.Min(160, WordFaceStart(dim) - CharFaceStart(dim));

    /// <summary>
    /// Width of the word-slot face: from 202 to end (capped at 320 = 40 slots × 8 dims).
    /// Zero when dim ≤ 202.
    /// </summary>
    public static int WordFaceDims(int dim) => dim > 202 ? Math.Min(320, dim - 202) : 0;

    /// <summary>
    /// Number of semantic dimensions allocated per character slot.
    /// Formula: max(4, semanticDims/12) — tighter packing than /8, more slots.
    /// Minimum 4 dims/slot ensures adequate char discrimination in the hash-seeded space.
    /// </summary>
    public static int SlotDims(int semanticDims) => Math.Max(4, semanticDims / 12);

    /// <summary>
    /// Number of semantic dimensions per chunk slot.
    /// Formula: max(4, semanticDims/10) — slightly more compact than char slots.
    /// </summary>
    public static int ChunkSlotDims(int semanticDims) => Math.Max(4, semanticDims / 10);

    /// <summary>
    /// Number of semantic dimensions per word slot. Fixed at 8 dims/slot → 40 slots at 320 dims.
    /// </summary>
    public static int WordSlotDims(int semanticDims) => 8;

    /// <summary>
    /// Maximum number of clean character slots available at this dimension.
    /// </summary>
    public static int MaxCharSlots(int dim)
    {
        var charDims = CharFaceDims(dim);
        var slot = SlotDims(charDims);
        return slot > 0 ? Math.Max(0, charDims / slot) : 0;
    }

    /// <summary>
    /// Maximum number of word slots available at this dimension.
    /// </summary>
    public static int MaxWordSlots(int dim)
    {
        var wordDims = WordFaceDims(dim);
        var slot = WordSlotDims(wordDims);
        return slot > 0 ? Math.Max(0, wordDims / slot) : 0;
    }

    // ============================================================================================
    // ADDRESS-SPACE LAYOUT (PLATONIC_NUCLEUS.md §1, PLATONIC_THEORY.md §9.4) — FIXED offsets,
    // active ONLY when dim >= AddressSpaceDim (512). Below that, the legacy CharFace/WordFace layout
    // above remains in force so small-dim callers/tests are unaffected. The whole region
    // [0, OrbitalStart) is a FROZEN, codec-derived, invertible address; only [OrbitalStart, dim) is
    // the learned, materialised-only meaning tail.
    //
    //   poly      [0,21)     number value (add/sub) — byte-identical to the legacy numeric face
    //   log       [21,42)    number value (mul/div) — byte-identical
    //   kind      [42,48)    6-dim deterministic kind code
    //   spelling  [48,208)   16 char-slots × 10 dims; slot i = deterministic atom of s[i] (decodable)
    //   structure [208,400)  6 child-slots × 32 = (child-digest 24 + role/label 8)
    //   op        [400,416)  16-dim deterministic op code
    //   orbital   [416,dim)  LEARNED meaning tail — the ONLY mutable region. At the production face dim 1024
    //                        this is [416,1024) = 608 dims (the frozen address bands ≤416 are fixed regardless of dim).
    // ============================================================================================

    /// <summary>Production dimension at which the fixed-offset address-space layout activates.</summary>
    public const int AddressSpaceDim = 512;

    /// <summary>True when the fixed-offset address-space band layout is in force (dim ≥ 512).</summary>
    public static bool IsAddressSpace(int dim) => dim >= AddressSpaceDim;

    /// <summary>Start of the kind band; 6-dim deterministic per-kind code.</summary>
    public const int KindStart = 42;
    /// <summary>Width of the kind band.</summary>
    public const int KindDims = 6;

    /// <summary>Start of the spelling band — the authoritative, decodable identity for text.</summary>
    public const int SpellingStart = 48;
    /// <summary>Number of character slots in the spelling band.</summary>
    public const int SpellingSlots = 16;
    /// <summary>Dimensions per spelling char-slot.</summary>
    public const int SpellingSlotDims = 10;
    /// <summary>Total spelling band width (16 × 10 = 160) → [48,208).</summary>
    public const int SpellingDims = SpellingSlots * SpellingSlotDims;

    /// <summary>Start of the structure band — ordered child digests + role/label codes.</summary>
    public const int StructureStart = 208;
    /// <summary>Number of child slots in the structure band.</summary>
    public const int StructureSlots = 6;
    /// <summary>Dimensions per structure child-slot (24 child-digest + 8 role/label).</summary>
    public const int StructureSlotDims = 32;
    /// <summary>Dimensions of the child digest within a structure slot.</summary>
    public const int StructureChildDigestDims = 24;
    /// <summary>Dimensions of the role/label code within a structure slot.</summary>
    public const int StructureRoleDims = 8;
    /// <summary>Total structure band width (6 × 32 = 192) → [208,400).</summary>
    public const int StructureDims = StructureSlots * StructureSlotDims;

    /// <summary>Start of the op band; 16-dim deterministic op code.</summary>
    public const int OpStart = 400;
    /// <summary>Width of the op band.</summary>
    public const int OpDims = 16;

    /// <summary>Start of the learned orbital tail and one-past-the-end of the frozen address.</summary>
    public const int OrbitalStart = 416;
}

/// <summary>
/// Deterministic kind code written into the address-space <see cref="FaceLayout.KindStart"/> band.
/// <c>None</c> (numbers, read off poly/log) is the all-zero code; every other kind is one-hot.
/// </summary>
public enum PlatonicKind
{
    None = 0,
    Atom = 1,
    Object = 2,
    Relation = 3,
    Function = 4,
    Composition = 5
}
