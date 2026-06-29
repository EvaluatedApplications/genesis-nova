using System.Globalization;
using GenesisNova.Core;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// The bridge to the KEPT, PROVEN codec (PLATONIC_THEORY.md §10 "must port verbatim"). The new core delegates ALL
/// identity construction to <see cref="PlatonicFaceComposer"/> — it invents no face math — so the number
/// homomorphism (<c>embed(a+b)=embed(a+b)</c>) and the generative char codec are preserved bit-for-bit.
///
/// It realizes the DUAL-FACE split (§4): an assembled positive face = an IMMUTABLE identity nucleus (arithmetic
/// poly/log for numbers, char-composed spelling for text — straight from the codec) ⊕ the element's emergent
/// SEMANTIC orbital (the only mutable region, born neutral, settling from κ via Law D). Numbers return their pure
/// homomorphic face (no stored orbital) so arithmetic stays exact; their associations are carried by relations.
/// </summary>
public static class FaceCodec
{
    /// <summary>
    /// Start of the mutable semantic (orbital) region. At address-space dims (≥512) this is the fixed
    /// orbital tail <see cref="FaceLayout.OrbitalStart"/> (416) — the ONLY mutable region; everything
    /// below it is frozen, decodable address. Below 512 it falls back to the legacy word face
    /// <see cref="FaceLayout.WordFaceStart"/> so small-dim callers/tests are unaffected.
    /// </summary>
    public static int SemanticStart(int dim)
        => FaceLayout.IsAddressSpace(dim) ? FaceLayout.OrbitalStart : FaceLayout.WordFaceStart(dim);

    /// <summary>Width of the semantic orbital an element stores.</summary>
    public static int SemanticLength(int dim) => System.Math.Max(0, dim - SemanticStart(dim));

    /// <summary>
    /// A NEUTRAL (spawn-spread) orbital: tiny ±0.01 deterministic, symbol-stable seed noise — the legitimate
    /// symmetry-breaking birth (`SeedLearnableDims`), NOT the arbitrary unit-norm <c>AddWordIdentity</c> stamp. The
    /// noise is MEANINGLESS (no assigned structure); a concept's position still EMERGES from its contradictions
    /// (Law D). Zeros would never break symmetry — attraction from the origin stays at the origin — so a faint,
    /// settled-from-κ spread is the correct neutral. Reuses the proven seeder so births match the old codec.
    /// </summary>
    public static double[] NeutralSemantic(string symbol, int dim)
    {
        var full = PlatonicFaceComposer.GetFreshEmbedding(symbol, dim); // identity + seed noise, NO stamp
        var start = SemanticStart(dim);
        var n = SemanticLength(dim);
        var orbital = new double[n];
        for (var i = 0; i < n; i++)
            orbital[i] = full[start + i];
        return orbital;
    }

    /// <summary>True if the symbol is a number (routed through the homomorphism, never a stored orbital).</summary>
    public static bool IsNumeric(string symbol)
        => double.TryParse(symbol, NumberStyles.Float, CultureInfo.InvariantCulture, out _);

    /// <summary>
    /// The stable per-concept TOKEN — a deterministic, symbol-seeded unit vector over the large/word face. This is
    /// the BASIS of distributional meaning (PLATONIC_NUCLEUS.md, [[nova-distributional-large-face]]): a concept's
    /// large-face cloud is the superposition of its OWN token plus its relational context's tokens. Distinct
    /// concepts get ≈orthogonal tokens (so unrelated stay apart), and shared context makes related clouds overlap.
    /// NOT a stamped position (that was AddWordIdentity) — it is the atom the cloud is built FROM.
    /// </summary>
    public static double[] Token(string symbol, int dim)
    {
        var n = SemanticLength(dim);
        var t = new double[n];
        if (n == 0) return t;
        var rng = new System.Random(StableHash(symbol));
        var s = 0.0;
        for (var i = 0; i < n; i++) { t[i] = rng.NextDouble() * 2.0 - 1.0; s += t[i] * t[i]; }
        var inv = s > 1e-12 ? 1.0 / System.Math.Sqrt(s) : 0.0;
        for (var i = 0; i < n; i++) t[i] *= inv;
        return t;
    }

    private static int StableHash(string s)
    {
        var h = 2166136261u;
        foreach (var c in s) { h ^= c; h *= 16777619u; }
        return unchecked((int)h);
    }

    /// <summary>
    /// Assemble an element's full positive face π(e). NUMBERS: the pure homomorphic face, recomputed exactly from the
    /// codec (identity, exact, generalizing — never carries a learned orbital, so arithmetic is bit-exact). TEXT:
    /// the codec's immutable identity (char-composed spelling) with the mutable semantic orbital written into the
    /// word face [SemanticStart, dim).
    /// </summary>
    public static double[] AssemblePositiveFace(string symbol, double[] semanticOrbital, int dim)
    {
        double[] face;
        if (IsNumeric(symbol))
        {
            double.TryParse(symbol, NumberStyles.Float, CultureInfo.InvariantCulture, out var value);
            face = PlatonicFaceComposer.GetFreshNumericEmbedding(value, dim); // arithmetic face exact (homomorphism)
        }
        else
        {
            face = PlatonicFaceComposer.GetFreshEmbedding(symbol, dim); // immutable char identity (+ codec seed)
        }
        // Overlay the emergent SEMANTIC orbital onto [SemanticStart, dim). For TEXT this replaces the codec seed; for
        // a NUMBER this gives it a semantic position (so "5" can settle near "five") WITHOUT touching its arithmetic
        // face [0,42) — arithmetic stays bit-exact while the number gains a place in the relational/semantic space.
        var start = SemanticStart(dim);
        var n = System.Math.Min(semanticOrbital.Length, dim - start);
        for (var i = 0; i < n; i++)
            face[start + i] = semanticOrbital[i];
        return face;
    }

    /// <summary>G4 conservation: the complement face ¬e = −e, exact on every dimension.</summary>
    public static double[] Negate(double[] face)
    {
        var neg = new double[face.Length];
        for (var i = 0; i < face.Length; i++)
            neg[i] = -face[i];
        return neg;
    }
}
