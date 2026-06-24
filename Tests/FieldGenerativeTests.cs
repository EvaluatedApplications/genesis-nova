using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// THE CONSCIOUS FIELD'S GENERATIVE ARM, restored from the legacy ladder (PLATONIC_RECKONING.md keep-core). The field
// was limited to compute + one-shot retrieval; these prove it can now APPLY A LEARNED FUNCTION to a NOVEL operand by
// composition — genuine generalisation (a function learned from a few examples, used on inputs never seen), not the
// stored-pair lookup that made it feel like a calculator + dictionary.
public sealed class FieldGenerativeTests
{
    private readonly ITestOutputHelper _out;
    public FieldGenerativeTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_AppliesLearnedTransform_ToNovelOperand()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        // Teach "triple" = ×3 and "boost" = +10 from a handful of examples (pure vector arithmetic, no gradient).
        var transforms = new TransformAccumulator(config.FaceDimension);
        foreach (var x in new[] { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 })
        {
            transforms.Learn("triple", InputEmbeddingComposer.GetFreshNumericEmbedding(x, config.FaceDimension),
                                       InputEmbeddingComposer.GetFreshNumericEmbedding(x * 3, config.FaceDimension));
            transforms.Learn("boost", InputEmbeddingComposer.GetFreshNumericEmbedding(x, config.FaceDimension),
                                      InputEmbeddingComposer.GetFreshNumericEmbedding(x + 10, config.FaceDimension));
        }

        var mind = new GenesisInferenceEngine(tok, model, space, null, transformAccumulator: transforms) { ConsciousField = true };

        // Apply to operands NEVER taught — composing the learned transform, not recalling a stored pair.
        foreach (var (q, want) in new[] { ("triple 9", "27"), ("triple 11", "33"), ("boost 40", "50") })
        {
            var r = mind.Generate(new GenerationRequest(q, 8));
            _out.WriteLine($"{q} -> '{r.Output?.Trim()}' [{r.DecisionPath}]");
            Assert.Equal(want, r.Output?.Trim());
            Assert.StartsWith("field-transform", r.DecisionPath);
        }
    }

    // STAGE 1 of the generative tick: the field RUNS a query as a cascade — multi-step derivations that the one-shot
    // dispatch cannot reach are BUILT mid-inference, one manufactured intermediate element per tick.
    [Fact]
    public void FieldTick_ComposesLearnedTransforms_AcrossTicks()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        var transforms = new TransformAccumulator(config.FaceDimension);
        foreach (var x in new[] { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 })
        {
            transforms.Learn("double", InputEmbeddingComposer.GetFreshNumericEmbedding(x, config.FaceDimension),
                                       InputEmbeddingComposer.GetFreshNumericEmbedding(x * 2, config.FaceDimension));
            transforms.Learn("incr", InputEmbeddingComposer.GetFreshNumericEmbedding(x, config.FaceDimension),
                                     InputEmbeddingComposer.GetFreshNumericEmbedding(x + 1, config.FaceDimension));
        }
        var mind = new GenesisInferenceEngine(tok, model, space, null, transformAccumulator: transforms) { ConsciousField = true };

        // ONE-SHOT (ticks OFF): the learned-function route applies only ONE op — it CANNOT chain, so it's wrong.
        var oneShot = mind.Generate(new GenerationRequest("double incr 5", 8));
        _out.WriteLine($"one-shot: 'double incr 5' -> '{oneShot.Output?.Trim()}' [{oneShot.DecisionPath}]");
        Assert.NotEqual("12", oneShot.Output?.Trim()); // can't reach the 2-step answer

        // TICK LOOP (ON): cascade — incr(5)=6 manufactured on tick 1, double(6)=12 on tick 2.
        mind.FieldTicksEnabled = true;
        var ticked = mind.Generate(new GenerationRequest("double incr 5", 8));
        _out.WriteLine($"ticked:   'double incr 5' -> '{ticked.Output?.Trim()}' [{ticked.DecisionPath}]");
        Assert.Equal("12", ticked.Output?.Trim());
        Assert.StartsWith("field-tick", ticked.DecisionPath);

        // A 3-step cascade — incr(5)=6, double(6)=12, double(12)=24 — proving the wavefront carries forward.
        var three = mind.Generate(new GenerationRequest("double double incr 5", 8));
        _out.WriteLine($"ticked:   'double double incr 5' -> '{three.Output?.Trim()}' [{three.DecisionPath}]");
        Assert.Equal("24", three.Output?.Trim());
    }
}
