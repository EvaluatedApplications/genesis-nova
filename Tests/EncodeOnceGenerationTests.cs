using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// ENCODE-ONCE generation optimization safety net. The generation hot path now encodes the (invariant)
/// prompt to its encoder seed (hInput) ONCE per generation and reuses it for every decode step, instead
/// of re-running EncodeInput over the whole unchanged prompt each step (O(N+M) instead of O(N·M)).
///
/// HARD INVARIANT: the generated token sequence must be BYTE-IDENTICAL to the old per-step re-encode.
/// This is exact because EncodeInput reads ONLY the input tokens, the model's weights are constant within
/// a single generation (pure no-grad forward), and the only per-step variation is the previous-token
/// embedding fed to GruStep. This test asserts that directly: the recompute path (promptState == null,
/// EncodeInput every step) and the encode-once path (a precomputed seed reused every step) produce the
/// SAME next token at every step of a multi-step decode — on an untrained model (deterministic argmax,
/// so no training is needed to make the comparison meaningful).
/// </summary>
public sealed class EncodeOnceGenerationTests
{
    private readonly ITestOutputHelper _out;
    public EncodeOnceGenerationTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void EncodeOnce_ProducesIdenticalTokenSequence_AsPerStepReEncode()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);

        // A multi-token prompt; encode to ids exactly as the engine does.
        var inputTokens = tokenizer.Encode("what is 12 plus 7", addEos: false).ToList();
        Assert.True(inputTokens.Count > 1, "prompt must be multi-token to exercise the encoder");

        // Drive a multi-step decode TWICE with identical per-step inputs: once recomputing the encoder
        // seed every step (promptState == null), once reusing a single precomputed seed. The two token
        // streams must match exactly, step for step.
        const int steps = 8;

        List<int> RunDecode(bool encodeOnce)
        {
            var emitted = new List<int>();
            var prev = tokenizer.BosTokenId;
            using var seed = encodeOnce ? model.EncodePromptState(inputTokens) : null;
            for (var i = 0; i < steps; i++)
            {
                var next = model.PredictNextToken(
                    inputTokens,
                    prev,
                    stepIndex: i,
                    disallowToken: i == 0 ? tokenizer.EosTokenId : null,
                    penalizedTokens: emitted,
                    repetitionPenalty: 0.35,
                    tokenBiases: null,
                    stopToken: tokenizer.EosTokenId,
                    promptState: encodeOnce ? seed : null);
                emitted.Add(next);
                if (next == tokenizer.EosTokenId)
                    break;
                prev = next;
            }
            return emitted;
        }

        var reEncode = RunDecode(encodeOnce: false);
        var encodeOnce = RunDecode(encodeOnce: true);

        _out.WriteLine($"re-encode : [{string.Join(",", reEncode)}]");
        _out.WriteLine($"encode-once: [{string.Join(",", encodeOnce)}]");

        Assert.NotEmpty(reEncode);
        Assert.Equal(reEncode, encodeOnce); // byte-identical generated token sequence
    }

    [Fact]
    public void EncodeOnce_FullEngineGeneration_ProducesSensibleNonEmptySequence()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory, null);

        // Exercise the real engine generation loop (which now threads the encode-once seed through every
        // step and disposes it after). The output is a sensible, bounded, non-empty token sequence.
        var result = inference.Generate(new GenerationRequest("what is 12 plus 7", 6));

        Assert.NotNull(result);
        Assert.NotEmpty(result.GeneratedTokens);
        Assert.True(result.GeneratedTokens.Length <= 6, "generation must respect the token budget");
    }
}
