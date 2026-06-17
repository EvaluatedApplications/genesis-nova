using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// WORD ELEMENTS (Model B): a concept's spelling-independent identity is a DISTINCT element (its own index,
/// like Function/Relation elements), positioned by a stable, ~orthogonal WHOLE-STRING code in the word face.
/// A multi-word element RELATES to its constituent word elements, so concat (compose → whole) and decompose
/// (read the parts) are element-native relations. char = spelling, word element = identity, relations =
/// meaning. Fast, deterministic, at production face dimension.
/// </summary>
public sealed class PlatonicFaceIdentityTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicFaceIdentityTests(ITestOutputHelper o) => _out = o;

    private static int Dim => ProductionDims.FaceDimension; // 512 — word face active (>202)
    private static PlatonicSpaceMemory Space() => new(faceDimension: Dim, seed: 7);

    private static double[] WordFace(double[] e)
    {
        var start = FaceLayout.WordFaceStart(Dim);
        var dims = FaceLayout.WordFaceDims(Dim);
        var slice = new double[dims];
        Array.Copy(e, start, slice, 0, dims);
        return slice;
    }

    private static double Norm(double[] v) { var s = 0.0; foreach (var x in v) s += x * x; return Math.Sqrt(s); }

    private static double Cosine(double[] a, double[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (na < 1e-12 || nb < 1e-12) ? 0.0 : dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    [Fact] // Every concept (single or multi word) registers a DISTINCT word element with a UNIT word-face identity.
    public void RegisterWordElement_CreatesDistinctElement_WithUnitWordIdentity()
    {
        Assert.True(Dim > 202, "test requires a real word face (production dim)");
        var space = Space();
        foreach (var c in new[] { "apple", "fruit", "red apple", "machine learning daemon" })
        {
            var e = space.RegisterWordElement(c);
            Assert.Equal(ElementKind.Composition, e.Kind);
            Assert.True(space.TryGetWordElement(c, out var got) && got.Id == e.Id);
            Assert.True(Math.Abs(Norm(WordFace(e.Embedding)) - 1.0) < 1e-6, $"'{c}' word element should have UNIT word-face identity");
        }
    }

    [Fact] // Identity is keyed by the WHOLE string → lexically-similar words are NOT confusable; concepts near-orthogonal.
    public void WordIdentity_IsWholeString_LexicallySimilar_AreDistinct()
    {
        var space = Space();
        var four = WordFace(space.RegisterWordElement("four").Embedding);
        var fruit = WordFace(space.RegisterWordElement("fruit").Embedding);
        var fort = WordFace(space.RegisterWordElement("fort").Embedding);
        _out.WriteLine($"|cos(four,fruit)|={Math.Abs(Cosine(four, fruit)):F3}  |cos(four,fort)|={Math.Abs(Cosine(four, fort)):F3}");
        Assert.True(Math.Abs(Cosine(four, fruit)) < 0.30, "lexically-similar words must be distinct identities");
        Assert.True(Math.Abs(Cosine(four, fort)) < 0.30);

        var redApple = WordFace(space.RegisterWordElement("red apple").Embedding);
        var red = WordFace(space.RegisterWordElement("red").Embedding);
        var apple = WordFace(space.RegisterWordElement("apple").Embedding);
        Assert.True(Math.Abs(Cosine(redApple, red)) < 0.30, "'red apple' must differ from 'red'");
        Assert.True(Math.Abs(Cosine(redApple, apple)) < 0.30, "'red apple' must differ from 'apple'");
    }

    [Fact] // A multi-word element RELATES to its constituent word elements (concat); decompose reads them back.
    public void MultiWord_RelatesToParts_AndDecomposes()
    {
        var space = Space();
        var whole = space.RegisterWordElement("red apple");
        Assert.Equal(2, whole.RelatedTo.Length);

        var parts = space.DecomposeWordElement("red apple");
        var symbols = parts.Select(p => p.Symbol).OrderBy(s => s).ToArray();
        Assert.Equal(new[] { "apple", "red" }, symbols);

        // The parts are themselves first-class word elements (registered, retrievable).
        Assert.True(space.TryGetWordElement("red", out _));
        Assert.True(space.TryGetWordElement("apple", out _));
        // An atomic word element decomposes to nothing.
        Assert.Empty(space.DecomposeWordElement("apple"));
    }

    [Fact] // Idempotent + own index: re-register returns the same element and adds nothing; concept store untouched.
    public void Register_IsIdempotent_AndLivesInItsOwnIndex()
    {
        var space = Space();
        var nodesBefore = space.NodeCount;
        var first = space.RegisterWordElement("daemon");
        var countAfterFirst = space.WordElements.Count;
        var again = space.RegisterWordElement("daemon");
        Assert.Equal(first.Id, again.Id);
        Assert.Equal(countAfterFirst, space.WordElements.Count);
        Assert.Equal(nodesBefore, space.NodeCount); // word elements never pollute concept retrieval
    }

    [Fact] // The word element is IDENTITY-only: decoupled from spelling (char face) and value (poly/log) — both ≈0.
    public void WordElement_IsIdentityOnly_DecoupledFromSpellingAndValue()
    {
        var space = Space();
        var e = space.RegisterWordElement("apple").Embedding;
        var charEnergy = 0.0;
        for (var d = FaceLayout.CharFaceStart(Dim); d < FaceLayout.WordFaceStart(Dim); d++) charEnergy += e[d] * e[d];
        var polyEnergy = 0.0;
        for (var d = 0; d < FaceLayout.ArithmeticFaceEnd(Dim); d++) polyEnergy += e[d] * e[d];
        Assert.True(charEnergy < 1e-9, "word element must NOT carry spelling (that's the char element's job)");
        Assert.True(polyEnergy < 1e-9, "word element must NOT carry numeric value");
    }
}
