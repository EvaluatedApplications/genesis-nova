using System;
using System.Collections.Generic;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE VOID IS LATENT STRUCTURE. The core claim of the address-space layout (dim >= 512): a COORDINATE decodes to
/// its element from the coordinate ALONE — no stored Element, no relation, nothing written at that address. The
/// address space is dense and self-describing; "storing" a concept only marks an address as VISITED. These tests
/// build a coordinate purely from the frozen codec for a symbol that was NEVER added to the space, decode it THROUGH
/// a fresh empty DialecticalSpace, and assert recovery while asserting the symbol is genuinely absent (latent).
/// </summary>
public sealed class AddressSpaceVoidTests
{
    private const int Dim = 512; // address-space layout active only at dim >= 512
    private readonly ITestOutputHelper _out;
    public AddressSpaceVoidTests(ITestOutputHelper o) => _out = o;

    // 1. LATENT WORD — an in-vocab word that was never added decodes off its frozen spelling-band address.
    [Fact]
    public void LatentWord_DecodesFromVoid()
    {
        const string symbol = "zephyrium";
        var space = new DialecticalSpace(Dim);
        Assert.False(space.ContainsConcept(symbol), "precondition: symbol must NOT be stored (truly latent)");

        var coord = PlatonicFaceComposer.GetFreshEmbedding(symbol, Dim);
        var ok = space.TryDecodeCoordinate(coord, out var kind, out var decoded, out var conf);
        _out.WriteLine($"[1] latent word: ok={ok} kind={kind} symbol='{decoded}' conf={conf:F4}");

        Assert.True(ok, "void coordinate must decode");
        Assert.Equal(symbol, decoded); // exact recovery — report the actual string if it ever differs
        Assert.True(kind is PlatonicKind.Object or PlatonicKind.Atom, $"word kind expected Object/Atom, got {kind}");
        Assert.False(space.ContainsConcept(symbol), "decoding must NOT have stored it");
    }

    // 2. LATENT NUMBER — a value never added decodes off the poly/log homomorphism bands (exact-ish).
    [Fact]
    public void LatentNumber_DecodesFromVoid()
    {
        const double value = 8675309;
        var space = new DialecticalSpace(Dim);

        var coord = PlatonicFaceComposer.GetFreshNumericEmbedding(value, Dim);
        var ok = space.TryDecodeCoordinate(coord, out var kind, out var decoded, out var conf);
        _out.WriteLine($"[2] latent number: ok={ok} kind={kind} symbol='{decoded}' conf={conf:F4}");

        Assert.True(ok, "void numeric coordinate must decode");
        var parsed = double.Parse(decoded, System.Globalization.CultureInfo.InvariantCulture);
        Assert.True(Math.Abs(parsed - value) < 1.0, $"decoded {parsed} != {value} (poly/log precision)");
        // numbers carry the all-zero kind code (None) — value lives in the bands, not a kind tag.
        Assert.Equal(PlatonicKind.None, kind);
    }

    // 3. LATENT COMPOSITE, NUMERIC CHILDREN (exact) — encode <5 is 8> into a bare coord, decode children exactly.
    [Fact]
    public void LatentComposite_NumericChildren_DecodeExactly()
    {
        var face = new double[Dim];
        var child5 = PlatonicFaceComposer.GetFreshNumericEmbedding(5, Dim);
        var child8 = PlatonicFaceComposer.GetFreshNumericEmbedding(8, Dim);
        PlatonicFaceComposer.EncodeStructure(face, new[] { child5, child8 }, "is", Dim);

        var dec = PlatonicFaceDecoder.DecodeStructure(face, Dim, new[] { "is" });
        _out.WriteLine($"[3] latent composite (numeric): label='{dec.Label}' children=[{string.Join(",", dec.Children)}]");

        Assert.Equal("is", dec.Label);
        Assert.Equal(2, dec.Children.Count);
        Assert.Equal("5", dec.Children[0]); // numeric child digests decode EXACTLY (poly head)
        Assert.Equal("8", dec.Children[1]);
    }

    // 4. LATENT COMPOSITE, WORD CHILDREN (digest-limited boundary) — children decode only to a short spelling-digest
    //    prefix, per the documented structure-band limit (EncodeStructure keeps only the first 2 spelling slots /
    //    ~2 leading chars per child). We assert the LABEL decodes and the call succeeds; we do NOT assert full word
    //    recovery here — that is the known address-space boundary for composite WORD children.
    [Fact]
    public void LatentComposite_WordChildren_DigestPrefixBoundary()
    {
        var face = new double[Dim];
        var apple = PlatonicFaceComposer.GetFreshEmbedding("apple", Dim);
        var fruit = PlatonicFaceComposer.GetFreshEmbedding("fruit", Dim);
        PlatonicFaceComposer.EncodeStructure(face, new[] { apple, fruit }, "is", Dim);

        var dec = PlatonicFaceDecoder.DecodeStructure(face, Dim, new[] { "is" });
        _out.WriteLine($"[4] latent composite (word): label='{dec.Label}' children=[{string.Join(",", dec.Children)}]");

        Assert.Equal("is", dec.Label);          // role/label code decodes exactly
        Assert.Equal(2, dec.Children.Count);     // both child slots are present
        // children resolve only to a leading digest prefix of "apple"/"fruit" — documented limit, not a failure.
    }
}
