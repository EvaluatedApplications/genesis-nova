using System;
using System.Globalization;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE OP BAND + THE FUNCTION KIND — the last gap in the address-space layout (PLATONIC_NUCLEUS.md §1/§2). An
/// operation is a FIRST-CLASS, decodable element (genesis G5 — "a Function is itself an element"): its op-code lives
/// in the frozen op band [400,416) and its kind band = Function, so the operation decodes from its coordinate ALONE —
/// realised OR latent void — exactly like the number / spelling / structure bands. These tests run at the production
/// face dim (1024), where the address-space layout is active (orbital [416,1024)).
/// </summary>
public sealed class OpBandFunctionTests
{
    private const int Dim = 1024; // production face: address-space layout active (dim >= 512)
    private readonly ITestOutputHelper _out;
    public OpBandFunctionTests(ITestOutputHelper o) => _out = o;

    // 1. LATENT OP — a Function/op coordinate built PURELY from the codec (nothing stored) decodes to kind=Function and
    //    op="plus" off the frozen op band. "Decode the void" for the 7th band.
    [Fact]
    public void LatentOp_RoundTripsFromVoid()
    {
        var face = new double[Dim];
        PlatonicFaceComposer.EncodeKind(face, PlatonicKind.Function, Dim);
        PlatonicFaceComposer.EncodeOp(face, "plus", Dim);

        var kind = PlatonicFaceDecoder.DecodeKind(face, Dim);
        var op = PlatonicFaceDecoder.DecodeOp(face, Dim, new[] { "plus", "minus", "times" });
        _out.WriteLine($"[1] latent op: kind={kind} op='{op}'");

        Assert.Equal(PlatonicKind.Function, kind);
        Assert.Equal("plus", op); // nearest registered op-code, nothing stored
    }

    // 2. LIVE FUNCTION ELEMENT — RegisterOperationToken realises a kind=Function element whose FullFace decodes via
    //    TryDecodeCoordinate to (Function, "plus"); the bare token "plus" is NOT a retrievable concept (reserved), and
    //    NodeCount is unaffected (a Function is a route, not a concept — excluded from the concept count, like atoms).
    [Fact]
    public void RegisterOperationToken_CreatesLiveDecodableFunction()
    {
        var space = new DialecticalSpace(Dim);
        var before = space.NodeCount;

        space.RegisterOperationToken("plus");

        Assert.Equal(before, space.NodeCount);              // op is reserved/structural — NodeCount unaffected
        Assert.False(space.ContainsConcept("plus"));        // the bare cue token is NOT a normal retrievable concept

        var fn = space.GetElement("∘fn:plus");              // the live Function element under its reserved symbol
        Assert.NotNull(fn);
        Assert.Equal(GenesisNova.Cognition.Platonic.ElementKind.Function, fn!.Kind);

        Assert.True(space.TryGetConceptFace("∘fn:plus", out var fullFace)); // its assembled face (codec + op band)
        var ok = space.TryDecodeCoordinate(fullFace, out var kind, out var sym, out var conf);
        _out.WriteLine($"[2] live function: ok={ok} kind={kind} op='{sym}' conf={conf:F4} node={space.NodeCount}");

        Assert.True(ok);
        Assert.Equal(PlatonicKind.Function, kind);
        Assert.Equal("plus", sym);
        Assert.True(conf > 0.0);
    }

    // 3. OTHER-BANDS SANITY — adding the op band must not regress the other frozen bands at dim 1024.
    [Fact]
    public void OtherBands_StillDecode_NoRegression()
    {
        var space = new DialecticalSpace(Dim);

        // (a) NUMBER — latent value decodes off the poly/log homomorphism bands.
        var num = PlatonicFaceComposer.GetFreshNumericEmbedding(141, Dim);
        var (val, q, faceSel) = PlatonicFaceDecoder.DecodeNumericFromPrediction(num, Dim);
        _out.WriteLine($"[3a] number: value={val} q={q:F3} face={faceSel}");
        Assert.True(Math.Abs(val - 141.0) < 1.0, $"decoded {val} != 141");

        // (b) WORD — the spelling band round-trips a token exactly.
        var word = PlatonicFaceComposer.GetFreshEmbedding("cat", Dim);
        var spelled = PlatonicFaceDecoder.CharSlotDecode(word, Dim);
        _out.WriteLine($"[3b] word: '{spelled}'");
        Assert.Equal("cat", spelled);

        // (c) COMPOSITE — the structure-band label decodes.
        var face = new double[Dim];
        var child5 = PlatonicFaceComposer.GetFreshNumericEmbedding(5, Dim);
        var child8 = PlatonicFaceComposer.GetFreshNumericEmbedding(8, Dim);
        PlatonicFaceComposer.EncodeStructure(face, new[] { child5, child8 }, "is", Dim);
        var dec = PlatonicFaceDecoder.DecodeStructure(face, Dim, new[] { "is" });
        _out.WriteLine($"[3c] composite: label='{dec.Label}' children=[{string.Join(",", dec.Children)}]");
        Assert.Equal("is", dec.Label);
        Assert.Equal(2, dec.Children.Count);
        Assert.Equal("5", dec.Children[0]);
        Assert.Equal("8", dec.Children[1]);
    }
}
