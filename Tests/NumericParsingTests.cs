using System;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Numeric-CONCEPT classification must be STRICT and consistent with
/// <c>PlatonicSpaceMemory.TryParseNumber</c>: a token is composed into the polynomial/log
/// (arithmetic) face only when it is a genuine plain signed decimal. Malformed tokens like
/// "0+" / "5-" must NOT be treated as numbers — historically <c>NumberStyles.Any</c> accepted
/// trailing-sign garbage ("5-" → 5) and silently gave such tokens a numeric face.
///
/// Behavioural probe: <see cref="PlatonicFaceComposer.GetFreshNumericEmbedding"/> writes
/// poly[0] = value * 10^-1, so a real number lands a clearly-large value at index 0
/// (e.g. "5" → 0.5, "3.5" → 0.35), whereas a text/char token leaves index 0 untouched except
/// for tiny (~0.01) learnable-dim seeding noise. We therefore assert numeric poly[0] > 0.2 and
/// non-numeric poly[0] < 0.05 — a wide, unambiguous gap.
/// </summary>
public sealed class NumericParsingTests
{
    private static readonly int Dim = ProductionDims.FaceDimension;

    private readonly ITestOutputHelper _out;
    public NumericParsingTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData("0", 0.0)]    // value*10^-1 = 0.0 — special-cased below (numeric but poly[0]==0)
    [InlineData("1", 0.1)]
    [InlineData("5", 0.5)]
    [InlineData("-3", -0.3)]
    [InlineData("3.5", 0.35)]
    public void GenuineNumbers_GetPolynomialFace(string token, double expectedPoly0)
    {
        var embedding = PlatonicFaceComposer.GetFreshEmbedding(token, Dim);
        _out.WriteLine($"'{token}' -> poly[0]={embedding[0]:R} (expected ~{expectedPoly0:R})");

        // poly[0] = value * 10^-1; must match the homomorphic identity, not noise.
        Assert.Equal(expectedPoly0, embedding[0], 6);
    }

    [Theory]
    [InlineData("5")]
    [InlineData("3.5")]
    [InlineData("-3")]
    public void NonZeroNumbers_HaveLargePolyComponent(string token)
    {
        var embedding = PlatonicFaceComposer.GetFreshEmbedding(token, Dim);
        _out.WriteLine($"'{token}' -> |poly[0]|={Math.Abs(embedding[0]):R}");

        // A real non-zero number lands a clearly-large value at the arithmetic-face identity dim.
        Assert.True(Math.Abs(embedding[0]) > 0.2,
            $"expected '{token}' to be numeric (|poly[0]| > 0.2) but got {embedding[0]:R}");
    }

    [Theory]
    [InlineData("5-")]    // trailing sign — NumberStyles.Any wrongly parsed this as 5
    [InlineData("0+")]    // trailing sign — NumberStyles.Any wrongly parsed this as 0
    [InlineData("3.5x")]
    [InlineData("abc")]
    [InlineData("1+2")]   // an expression, not a single number
    [InlineData("(5)")]   // parentheses — NumberStyles.Any treats as -5
    [InlineData("1,000")] // thousands separator — not a genuine concept token
    public void MalformedTokens_AreNotComposedAsNumeric(string token)
    {
        var embedding = PlatonicFaceComposer.GetFreshEmbedding(token, Dim);
        _out.WriteLine($"'{token}' -> poly[0]={embedding[0]:R} (should be ~0, text-only)");

        // Treated as text (char face): the arithmetic-face identity dim stays near zero,
        // carrying only the tiny (~0.01) learnable-dim seeding noise — never a poly value.
        Assert.True(Math.Abs(embedding[0]) < 0.05,
            $"'{token}' must NOT get a numeric/polynomial face, but poly[0]={embedding[0]:R}");
    }

    // Direct contrast required by the spec: "5" numeric vs "5-" non-numeric, "3.5" vs "0+".
    [Fact]
    public void NumericVsMalformed_AreClearlyDistinguished()
    {
        var five = PlatonicFaceComposer.GetFreshEmbedding("5", Dim);
        var fiveBad = PlatonicFaceComposer.GetFreshEmbedding("5-", Dim);
        var threeHalf = PlatonicFaceComposer.GetFreshEmbedding("3.5", Dim);
        var zeroBad = PlatonicFaceComposer.GetFreshEmbedding("0+", Dim);

        _out.WriteLine($"'5' poly[0]={five[0]:R}, '5-' poly[0]={fiveBad[0]:R}");
        _out.WriteLine($"'3.5' poly[0]={threeHalf[0]:R}, '0+' poly[0]={zeroBad[0]:R}");

        Assert.True(Math.Abs(five[0]) > 0.2, "'5' should be numeric");
        Assert.True(Math.Abs(fiveBad[0]) < 0.05, "'5-' should NOT be numeric");
        Assert.True(Math.Abs(threeHalf[0]) > 0.2, "'3.5' should be numeric");
        Assert.True(Math.Abs(zeroBad[0]) < 0.05, "'0+' should NOT be numeric");
    }
}

/// <summary>
/// The SAME strict-numeric contract, enforced at the OTHER entry point: <see
/// cref="PlatonicSpaceMemory.TryGetConceptFace"/> (which gates on <c>PlatonicSpaceMemory.TryParseNumber</c>)
/// must treat a token as numeric iff it is a genuine plain signed decimal. This is a DISTINCT parser
/// from <see cref="PlatonicFaceComposer"/> above (the composer uses its own NumberStyles parse), so both
/// are pinned — they must agree, and a regression in either would let "5-" masquerade as a number and
/// pollute numeric relations / the glider's operand resolve. (Consolidated here from the former
/// RelationalEquivalenceTests, whose number-word-equivalence experiment was retired — that live carrier
/// is covered by corenova:number-word-equiv in CoreBootstrapRegimeTests / NumberWordEquivalenceProductionTests.)
/// </summary>
public sealed class MemoryNumericGateTests
{
    [Theory]
    [InlineData("0", true)]
    [InlineData("1", true)]
    [InlineData("-3", true)]
    [InlineData("3.5", true)]
    [InlineData("0+", false)]   // trailing sign — was wrongly parsed as 0 under NumberStyles.Any
    [InlineData("5-", false)]
    [InlineData("1+1", false)]  // glued expression — not a number
    [InlineData("one", false)]
    public void TryGetConceptFace_TreatsAsNumeric_OnlyForGenuineNumbers(string token, bool expectedNumeric)
    {
        var memory = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 1);
        var isNumeric = memory.TryGetConceptFace(token, out _); // fresh space: true ⟺ parses as a number
        Assert.Equal(expectedNumeric, isNumeric);
    }
}
