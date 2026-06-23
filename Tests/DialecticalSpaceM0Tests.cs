using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M0 acceptance gates for the ground-up DialecticalSpace (PLATONIC_THEORY.md §9). EMPIRICAL — every law is a probe
/// that runs and prints its numbers, not an assumption. Exercises the new core THROUGH the IPlatonicSpace contract
/// (the same surface inference/training use), at production face dimension.
/// </summary>
public sealed class DialecticalSpaceM0Tests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceM0Tests(ITestOutputHelper o) => _out = o;

    private static IPlatonicSpace NewSpace() => new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);

    [Theory] // §9.1 + swap gate — the REAL production arithmetic path (PlatonicGliderInterpreter → R2 compose →
             // PlatonicFaceDecoder) runs through the NEW core and is exact, for every op. This is what inference's
             // gru-query route calls; it proves the homomorphism boundary holds end-to-end on DialecticalSpace.
    [InlineData(2, 3, "+", 5)]
    [InlineData(7, 4, "-", 3)]
    [InlineData(4, 5, "*", 20)]
    [InlineData(8, 2, "/", 4)]
    [InlineData(84, 57, "+", 141)] // unseen operands — derived, not retrieved (T-Generalization)
    public void ProductionArithmetic_Exact_ThroughNewCore(double a, double b, string op, double expected)
    {
        var gliderOp = op switch
        {
            "+" => GliderOp.Add, "-" => GliderOp.Subtract, "*" => GliderOp.Multiply, "/" => GliderOp.Divide,
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        IPlatonicSpace space = NewSpace();
        var interp = new PlatonicGliderInterpreter(space);
        var glider = new PlatonicGlider("arith", new Compute(gliderOp, new GliderBlock[] { new Operand(0), new Operand(1) }));
        var value = double.Parse(interp.Execute(glider, new[] { a.ToString(System.Globalization.CultureInfo.InvariantCulture), b.ToString(System.Globalization.CultureInfo.InvariantCulture) }), System.Globalization.CultureInfo.InvariantCulture);
        _out.WriteLine($"{a}{op}{b} through DialecticalSpace = {value} (expected {expected})");
        Assert.Equal(expected, value, 6);
    }

    [Fact] // §9.1 — the number homomorphism is exact: poly(a)+poly(b)=poly(a+b), ported verbatim.
    public void Homomorphism_PolyFace_IsExact()
    {
        var space = NewSpace();
        Assert.True(space.TryGetConceptFace("3", out var f3));
        Assert.True(space.TryGetConceptFace("5", out var f5));
        Assert.True(space.TryGetConceptFace("8", out var f8));
        var nd = space.NumericDimensions;
        var maxErr = 0.0;
        for (var i = 0; i < nd; i++)
            maxErr = Math.Max(maxErr, Math.Abs((f3[i] + f5[i]) - f8[i]));
        _out.WriteLine($"poly homomorphism max |f(3)+f(5) - f(8)| over {nd} dims = {maxErr:E3}");
        Assert.True(maxErr < 1e-9, $"poly face must be additive-exact; maxErr={maxErr:E3}");
    }

    [Fact] // §9.2/§9.3 — the large-face cloud is the DISTRIBUTIONAL superposition of relational context (not a
           // stamped point): a concept that SHARES context with another clusters with it, a concept that shares
           // nothing stays orthogonal. Meaning EMERGES from relations. (PLATONIC_NUCLEUS.md / distributional core.)
    public void Cloud_EmergesFromRelationalContext()
    {
        var space = NewSpace();
        var semStart = FaceLayout.WordFaceStart(space.FaceDimension);
        double Dist(double[] a, double[] b) { var s = 0.0; for (var i = semStart; i < a.Length; i++) { var d = a[i] - b[i]; s += d * d; } return Math.Sqrt(s); }

        space.ObserveContradiction("cat", "kitten", 0.1);
        space.ObserveContradiction("dog", "kitten", 0.1); // cat & dog SHARE context (kitten)
        space.ObserveContradiction("car", "wheel", 0.1);  // car shares nothing with cat

        Assert.True(space.TryGetConceptFace("cat", out var cat));
        Assert.True(space.TryGetConceptFace("dog", out var dog));
        Assert.True(space.TryGetConceptFace("car", out var car));
        var dCatDog = Dist(cat, dog);
        var dCatCar = Dist(cat, car);
        _out.WriteLine($"cat↔dog (shared context) {dCatDog:F3}   cat↔car (no shared context) {dCatCar:F3}");
        Assert.True(dCatDog < dCatCar, "concepts that share relational context cluster — meaning emerged, not stamped");
        Assert.True(dCatDog > 0.0, "distinct identities — cat and dog are not identical");
    }

    [Fact] // §9 retrieval — observation positions related concepts clearly CLOSER than unrelated (the dialectic +
           // contrastive repulsion). Without this, geometric retrieval has no signal. Mirrors the legacy gate.
    public void Observe_SeparatesRelatedFromUnrelated()
    {
        var space = NewSpace();
        const int C = 4, K = 5;
        var clusters = Enumerable.Range(0, C).Select(c => Enumerable.Range(0, K).Select(i => $"c{c}item{i}").ToArray()).ToArray();
        for (var epoch = 0; epoch < 40; epoch++)
            foreach (var cl in clusters)
                for (var i = 0; i < K; i++)
                    for (var j = i + 1; j < K; j++)
                        space.ObserveContradiction(cl[i], cl[j], 0.0); // agree → attract

        var g = space.SummarizePushPullGeometry();
        _out.WriteLine($"related(pull) {g.RelatedMean:F3}  unrelated(push) {g.UnrelatedMean:F3}  separation {g.Separation:F3}");
        Assert.True(g.RelatedPairs > 0 && g.UnrelatedPairs > 0, "need both related and unrelated pairs sampled");
        // Relative margin (robust to the tiny seed-scale magnitudes M0 positions at — M2 scales these up): related
        // must be clearly closer than unrelated. Here related is ~4.5× closer.
        Assert.True(g.Separation > 0.0 && g.RelatedMean < 0.6 * g.UnrelatedMean,
            $"related must sit clearly closer than unrelated; related={g.RelatedMean:F3} unrelated={g.UnrelatedMean:F3}");
    }

    // (The old per-aspect Dialectic_PreservesContradictingAspects test was removed: the per-aspect contradiction
    // mechanism it asserted was replaced by the distributional large-face cloud — ambiguity/superposition is now
    // covered by LargeFaceMeaningTests, clustering by Cloud_EmergesFromRelationalContext + the gym geometry guard.)

    [Fact] // Swap gate — the REAL GenesisTrainer training step (which writes via ObserveContradiction /
           // FineEditFromExample / ReinforceEvidence) runs through the new core without throwing and yields a finite
           // loss. Proves the cold-path write members are wired through the production training loop.
    public void GenesisTrainer_TrainStep_RunsThroughNewCore()
    {
        var tokenizer = new GenesisNova.Tokenization.WhitespaceGenesisTokenizer();
        var model = new GenesisNova.Model.GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05));
        IPlatonicSpace space = NewSpace();
        var trainer = new GenesisNova.Train.GenesisTrainer(tokenizer, model, space);

        var l1 = trainer.TrainStep(new GenesisNova.Data.GenesisExample("say one", "one"));
        var l2 = trainer.TrainStep(new GenesisNova.Data.GenesisExample("count two three", "five"));

        _out.WriteLine($"train losses through DialecticalSpace: {l1.TotalLoss:F4}, {l2.TotalLoss:F4}; nodes={space.NodeCount} rels={space.RelationCount}");
        Assert.True(l1.TotalLoss >= 0.0 && !double.IsNaN(l1.TotalLoss));
        Assert.True(l2.TotalLoss >= 0.0 && !double.IsNaN(l2.TotalLoss));
    }

    [Fact] // §9.5 — G4 conservation: ¬e = −e exactly; total charge ≡ 0.
    public void Conservation_G4()
    {
        var space = NewSpace();
        for (var i = 0; i < 10; i++) space.ObserveContradiction($"w{i}", $"v{i}", 0.2);
        Assert.Equal(0.0, space.TotalCharge(), 6);
        Assert.True(space.TryGetConceptFace("w0", out var f));
        var neg = FaceCodec.Negate(f);
        for (var i = 0; i < f.Length; i++) Assert.Equal(-f[i], neg[i], 9);
    }

    [Fact] // §9.6 — G6 monotone/archive: the store only grows; archive retains (reactivatable), never destroys.
    public void Irreversibility_G6_ArchiveRetains()
    {
        var store = new ElementStore();
        var a = store.GetOrCreate("alpha", GenesisNova.Cognition.Platonic.ElementKind.Object, () => new double[4]);
        store.GetOrCreate("beta", GenesisNova.Cognition.Platonic.ElementKind.Object, () => new double[4]);
        Assert.Equal(2, store.TotalCount);
        store.Archive("alpha");
        Assert.Equal(1, store.ActiveCount);
        Assert.Equal(2, store.TotalCount); // retained, not destroyed
        var a2 = store.GetOrCreate("alpha", GenesisNova.Cognition.Platonic.ElementKind.Object, () => new double[4]); // reactivates same element
        Assert.Same(a, a2);
        Assert.Equal(2, store.ActiveCount);
        Assert.Equal(2, store.TotalCount); // no duplicate id (monotone)
    }
}
