using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Targeted TDD + empirical research on how the model learns trivial maths. The platonic space
/// computes symbolic arithmetic EXACTLY; the open questions are (a) whether the model learns to ROUTE
/// to that exact answer and (b) whether the space can LEARN that a number-word ("one") and its digit
/// ("1") are the same — from their RELATIONSHIP in the data, with nothing hardcoded. That relational
/// equivalence is the whole point of the platonic space.
/// </summary>
public sealed class ArithmeticLearningTests
{
    private readonly ITestOutputHelper _out;
    public ArithmeticLearningTests(ITestOutputHelper output) => _out = output;

    // ── Phase 1: the platonic space computes symbolic arithmetic EXACTLY (no NN involved). ──
    [Theory]
    [InlineData(1, 1, "+", 2)]
    [InlineData(2, 3, "+", 5)]
    [InlineData(0, 0, "+", 0)]
    [InlineData(7, 4, "-", 3)]
    [InlineData(4, 5, "*", 20)]
    [InlineData(8, 2, "/", 4)]
    [InlineData(5, 5, "/", 1)]   // result == 1 → log slot ~0 (edge case for the decoder)
    [InlineData(1, 1, "*", 1)]
    public void PlatonicArithmetic_IsExact(double a, double b, string op, double expected)
    {
        const int dim = 32;
        var faceA = PlatonicFaceComposer.GetFreshNumericEmbedding(a, dim);
        var faceB = PlatonicFaceComposer.GetFreshNumericEmbedding(b, dim);
        var additive = op is "+" or "-";
        var sign = op is "-" or "/" ? -1.0 : 1.0;

        var blended = new double[dim];
        for (var i = 0; i < dim; i++)
            blended[i] = faceA[i] + (sign * faceB[i]);

        var preferFace = additive ? 1 : 2; // 1=poly (add/sub), 2=log (mul/div)
        var (value, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(blended, dim, preferFace);
        if (!additive && face == "log")
            value *= Math.Sign(a) * Math.Sign(b);

        _out.WriteLine($"{a}{op}{b}: value={value} expected={expected} quality={quality:F3} face={face}");
        Assert.Equal(expected, value, 6);
    }

    // ── Phase 2: the MODEL must LEARN to route arithmetic to the platonic answer + generalize. ──
    [Fact]
    public void Model_LearnsRoutesAndGeneralizesArithmetic()
    {
        var config = new GenesisNovaConfig(HiddenSize: 64, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null,
            trainer.FoldPathDiscovery, trainer.TransformLibrary, trainer.TransformAccumulator,
            enableDiagnosticFaceArithmeticShortcut: true);
        trainer.SetInferencePolicy(inference);

        var curriculum = new List<GenesisExample>();
        for (var a = 0; a <= 9; a++)
            for (var b = 0; b <= 9; b++)
                curriculum.Add(new GenesisExample($"{a} + {b}", $"{a + b}"));
        string[][] text =
        {
            new[] { "say hello", "hello" }, new[] { "say world", "world" },
            new[] { "name", "genesis" }, new[] { "greet", "hi" }, new[] { "color sky", "blue" },
        };
        foreach (var t in text)
            for (var k = 0; k < 4; k++)
                curriculum.Add(new GenesisExample(t[0], t[1]));

        Probe P(string input)
        {
            var r = model.PredictRoute(tokenizer.Encode(input));
            var g = inference.Generate(new GenerationRequest(input, 4));
            return new Probe(input, r.RouteId, r.Confidence, g.Output.Trim(),
                g.UsedPlatonicQuery, g.UsedNeuralFallback, g.DecisionPath);
        }

        const int epochs = 20;
        var rng = new Random(123);
        for (var e = 0; e < epochs; e++)
        {
            for (var i = curriculum.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (curriculum[i], curriculum[j]) = (curriculum[j], curriculum[i]);
            }
            foreach (var ex in curriculum)
                trainer.TrainStep(ex);
            if (e % 5 == 0 || e == epochs - 1)
                _out.WriteLine($"epoch {e,2}  1+1: {P("1 + 1")}");
        }

        var seen = P("1 + 1");
        var unseenSub = P("5 - 3"); // unseen operation
        var unseenMul = P("3 * 3");
        _out.WriteLine($"SEEN   1+1 : {seen}");
        _out.WriteLine($"UNSEEN 5-3 : {unseenSub}");
        _out.WriteLine($"UNSEEN 3*3 : {unseenMul}");

        Assert.Equal("2", seen.Output);
        Assert.Equal(1, seen.RouteId);
        Assert.True(seen.UsedPlatonic && !seen.UsedNeuralFallback,
            $"1+1 should be answered by the platonic path, not neural fallback. Got: {seen}");
        Assert.Equal("2", unseenSub.Output);
        Assert.Equal("9", unseenMul.Output);
    }

    // ── Phase 3: RELATIONAL number-word equivalence. NOTHING is hardcoded. We train on the
    // relationship ("one" answers "1", and vice-versa) and ask: does the platonic space LEARN that
    // "one" and "1" are the same value — i.e. does "one"'s value-face CONVERGE toward "1"'s, and end
    // up closer to its own digit than to an unrelated number? That convergence IS the platonic space
    // discovering the equivalence from data, which is its purpose. ──
    [Fact]
    public void PlatonicSpace_LearnsNumberWordEquivalence_FromRelationship()
    {
        var config = new GenesisNovaConfig(HiddenSize: 64, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);

        // Only the RELATIONSHIP is provided as data — no table maps "one" to 1.
        var curriculum = new List<GenesisExample>
        {
            new("one", "1"), new("1", "one"),
            new("two", "2"), new("2", "two"),
            new("three", "3"), new("3", "three"),
            new("four", "4"), new("4", "four"),
            new("five", "5"), new("5", "five"),
        };

        // Distance over the VALUE face (polynomial + logarithmic = the homomorphic identity).
        double ValueDist(string x, string y)
        {
            Assert.True(memory.TryGetConceptFace(x, out var fx), $"no face for '{x}'");
            Assert.True(memory.TryGetConceptFace(y, out var fy), $"no face for '{y}'");
            var nd = Math.Min(2 * memory.NumericDimensions, Math.Min(fx.Length, fy.Length));
            var s = 0.0;
            for (var i = 0; i < nd; i++)
            {
                var d = fx[i] - fy[i];
                s += d * d;
            }
            return Math.Sqrt(s);
        }

        foreach (var ex in curriculum)
            trainer.TrainStep(ex); // one pass: materialise the concept nodes

        var beforeNear = ValueDist("one", "1");
        var beforeFar = ValueDist("one", "5");
        _out.WriteLine($"baseline  dist(one,1)={beforeNear:F4}  dist(one,5)={beforeFar:F4}");

        for (var e = 0; e < 40; e++)
            foreach (var ex in curriculum)
                trainer.TrainStep(ex);

        var afterNear = ValueDist("one", "1");
        var afterFar = ValueDist("one", "5");
        _out.WriteLine($"trained   dist(one,1)={afterNear:F4}  dist(one,5)={afterFar:F4}");

        // It LEARNED the relationship: "one" moved toward its digit "1" in value-face space,
        Assert.True(afterNear < beforeNear,
            $"'one' should converge toward '1' from the relationship (was {beforeNear:F4}, now {afterNear:F4})");
        // and ended up closer to its own digit than to an unrelated number "5".
        Assert.True(afterNear < afterFar,
            $"'one' should be closer to '1' than to '5' (one~1={afterNear:F4}, one~5={afterFar:F4})");
    }

    // ── Phase 4 (GOAL / TDD spec): the space LEARNS "one"≡"1" from the RELATIONSHIP, and USES that to
    // do arithmetic on a worded number it never saw in any sum — and ONLY for words it actually learned
    // (proving the equivalence is learned/relational, not a hardcoded table). "one" appears ONLY in the
    // relation, never as an arithmetic operand, so answering "one + one" REQUIRES composing the learned
    // relation (one→1) with the learned symbolic arithmetic (1+1=2). This test defines the goal; it is
    // expected RED until the relational-resolution capability exists. ──
    //
    // SKIPPED: the bespoke worded-arithmetic parser/operator-table that made this pass was reverted as
    // over-hardcoded. The chosen direction is first-principles instead — clean, discriminable relational
    // geometry (see SpaceGeometryTests, contrastive repulsion) so "one≡1" is learned RELATIONALLY, then
    // consumed by the general path. Kept as the north-star spec; re-enable when that path can satisfy it.
    [Fact]
    public void GOAL_WordedArithmetic_ViaLearnedRelation_NotHardcoded()
    {
        var config = new GenesisNovaConfig(HiddenSize: 64, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null,
            trainer.FoldPathDiscovery, trainer.TransformLibrary, trainer.TransformAccumulator,
            enableDiagnosticFaceArithmeticShortcut: true);
        trainer.SetInferencePolicy(inference);

        var words = new[] { "zero", "one", "two", "three", "four", "five" };
        var curriculum = new List<GenesisExample>();
        // (a) The ONLY place words meet values: the number-word ↔ digit relationship.
        for (var n = 0; n <= 5; n++)
        {
            curriculum.Add(new GenesisExample(words[n], $"{n}"));
            curriculum.Add(new GenesisExample($"{n}", words[n]));
        }
        // (b) The ONLY arithmetic: SYMBOLIC, digit operands. No worded operand ever appears in a sum.
        for (var a = 0; a <= 5; a++)
            for (var b = 0; b <= 5; b++)
                curriculum.Add(new GenesisExample($"{a} + {b}", $"{a + b}"));

        var rng = new Random(7);
        for (var e = 0; e < 25; e++)
        {
            for (var i = curriculum.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (curriculum[i], curriculum[j]) = (curriculum[j], curriculum[i]);
            }
            foreach (var ex in curriculum)
                trainer.TrainStep(ex);
        }

        Probe Pr(string input)
        {
            var r = model.PredictRoute(tokenizer.Encode(input));
            var g = inference.Generate(new GenerationRequest(input, 4));
            return new Probe(input, r.RouteId, r.Confidence, g.Output.Trim(),
                g.UsedPlatonicQuery, g.UsedNeuralFallback, g.DecisionPath);
        }

        _out.WriteLine("one REL : " + string.Join(", ", memory
            .GetNeighbors("one", PlatonicNeighborhoodType.Relational, 16, 0.0)
            .Select(n => $"{n.Concept}@conf{n.Confidence:F2}×obs{n.ObservationCount}")));

        var learned = Pr("one + one");      // worded operand, HELD OUT of all sums
        var unlearned = Pr("zarp + zarp");  // never related to any number
        var baseline = Pr("5 + 5");         // digits — must always work
        _out.WriteLine($"GOAL    one + one : {learned}");
        _out.WriteLine($"PROOF   zarp+zarp : {unlearned}");
        _out.WriteLine($"BASE    5 + 5     : {baseline}");

        Assert.Equal("10", baseline.Output);                       // digit arithmetic still exact
        Assert.Equal("2", learned.Output);                         // GOAL: resolved one→1 via the learned relation
        Assert.NotEqual("2", unlearned.Output);                    // NOT hardcoded: zarp cannot resolve
    }

    private readonly record struct Probe(
        string Input, int RouteId, double Confidence, string Output,
        bool UsedPlatonic, bool UsedNeuralFallback, string Path)
    {
        public override string ToString()
            => $"route={RouteId} conf={Confidence:F3} out='{Output}' platonic={UsedPlatonic} " +
               $"fallback={UsedNeuralFallback} path={Path}";
    }
}
