using GenesisNova.Core;
using GenesisNova.Cognition;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class GenesisInferenceEngineTests
{
    [Fact]
    public void WhenTrainedOnSimplePattern_ThenInferenceProducesLearnedOutput()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.08));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        for (var i = 0; i < 30; i++)
            trainer.TrainStep(new GenesisExample("say hello", "hello"));

        var result = inference.Generate(new GenerationRequest("say hello", MaxNewTokens: 4));

        Assert.Contains("hello", result.Output);
    }

    [Fact]
    public void WhenExpressionsHaveDifferentOrder_ThenModelLearnsDifferentOutputs()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        for (var i = 0; i < 120; i++)
        {
            trainer.TrainStep(new GenesisExample("2-1", "1"));
            trainer.TrainStep(new GenesisExample("1-2", "-1"));
        }

        var a = inference.Generate(new GenerationRequest("2-1", MaxNewTokens: 3));
        var b = inference.Generate(new GenerationRequest("1-2", MaxNewTokens: 3));

        Assert.NotEqual(a.Output, b.Output);
    }

    [Fact]
    // Natural language queries like "what is 4 plus 5" are deliberately NOT handled by
    // keyword heuristics. They route through the ML path. This test validates that:
    // 1. Compact arithmetic "4+5" correctly routes to platonic slot-decode after training
    // 2. Natural language for the same expression still produces the correct answer (via ML)
    public void WhenPlatonicArithmeticFaceIsLearned_ThenCompactExpressionUsesSlotDecode()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        for (var i = 0; i < 120; i++)
            trainer.TrainStep(new GenesisExample("4+5", "9"));

        // Compact expression always routes through platonic slot-decode (structural, not keyword)
        var compact = inference.Generate(new GenerationRequest(
            Input: "4+5",
            MaxNewTokens: 4));
        Assert.Contains("9", compact.Output);
        Assert.True(compact.UsedPlatonicQuery);
        Assert.Contains("platonic-query-slot-decode", compact.DecisionPath, StringComparison.OrdinalIgnoreCase);
        Assert.True(compact.PlatonicConfidence > 0.0);

        // Natural language routes through ML — no assertion on decision path, only correctness
        var natural = inference.Generate(new GenerationRequest(
            Input: "what is 4 plus 5",
            MaxNewTokens: 4));
        Assert.Contains("9", natural.Output);
    }

    [Fact]
    public void WhenInputIsCompactArithmeticExpression_ThenSlotDecoderReturnsExactResult()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        // Route head needs to learn platonic path for arithmetic expressions.
        for (var i = 0; i < 30; i++)
            trainer.TrainStep(new GenesisExample("1+1", "2"));

        var result = inference.Generate(new GenerationRequest("1+1", MaxNewTokens: 4));

        Assert.Equal("2", result.Output);
        Assert.Equal(1, result.ChunksGenerated);
    }

    [Fact]
    public void WhenChunkBudgetIsSmall_ThenGenerateUsesMultipleInferencePasses()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        memory.ObserveContradiction("alpha", "beta", 0.10);
        memory.ObserveContradiction("beta", "gamma", 0.12);

        var inference = new GenesisInferenceEngine(tokenizer, model, memory);
        var result = inference.Generate(new GenerationRequest(
            Input: "alpha",
            MaxNewTokens: 2,
            ChunkTokenBudget: 1));

        Assert.Equal(2, result.ChunksGenerated);
        Assert.Equal(2, result.GeneratedTokens.Length);
        Assert.Contains("chunked", result.DecisionPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WhenArithmeticIsWrappedInConversationContext_ThenSlotDecoderIsNotUsed()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        var wrapped =
            "Context:\nUser asked arithmetic before\n\nUser: 1+1\nAssistant:";
        var result = inference.Generate(new GenerationRequest(wrapped, MaxNewTokens: 4));

        Assert.NotEqual("platonic-query-slot-decode", result.DecisionPath);
    }

    [Fact]
    public void WhenChainLearnsDoneToken_ThenGenerateStopsBeforeDoneMarker()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 64, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        const string input = "say chain";
        const string first = "alpha";
        const string stop = "[DONE]";
        var continuation = $"{input} | {first} | [continue]";

        for (var i = 0; i < 140; i++)
        {
            trainer.TrainStep(new GenesisExample(input, first));
            trainer.TrainStep(new GenesisExample(continuation, stop));
        }

        var result = inference.Generate(new GenerationRequest(
            Input: input,
            MaxNewTokens: 4));

        Assert.False(string.IsNullOrWhiteSpace(result.Output));
        Assert.DoesNotContain(stop, result.Output, StringComparison.OrdinalIgnoreCase);
        Assert.True(result.ChunksGenerated == 1);
        Assert.False(string.IsNullOrWhiteSpace(result.DecisionPath));
    }

    [Fact]
    public void WhenKnownConceptsExist_ThenAutoRoutingCanUsePlatonicConceptChain()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        memory.ObserveContradiction("cat", "pet", 0.08);
        memory.ObserveContradiction("pet", "animal", 0.12);

        // Route head needs to learn platonic path for concept queries.
        for (var i = 0; i < 50; i++)
            trainer.TrainStep(new GenesisExample("tell me about cat", "pet"));

        var result = inference.Generate(new GenerationRequest(
            Input: "tell me about cat",
            MaxNewTokens: 8));

        Assert.True(result.UsedPlatonicQuery);
        Assert.Equal("platonic-query-concept-chain", result.DecisionPath);
        Assert.True(result.PlatonicHopCount >= 1);
        Assert.Contains("pet", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WhenNoDirectQueryPlanButPlatonicRelationsExist_ThenInferenceStaysOnNeuralPath()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        memory.ObserveContradiction("alpha", "beta", 0.10);
        memory.ObserveContradiction("beta", "gamma", 0.12);
        memory.ObserveContradiction("gamma", "delta", 0.14);

        var result = inference.Generate(new GenerationRequest(
            Input: "unknown prompt with no anchors",
            MaxNewTokens: 6));

        Assert.False(result.UsedPlatonicQuery);
        Assert.Contains("neural-token+platonic-bias", result.DecisionPath, StringComparison.OrdinalIgnoreCase);
        Assert.False(string.IsNullOrWhiteSpace(result.Output));
    }

    [Fact]
    public void WhenInputIsGreetingLike_ThenNeuralPathIsUsed()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(tokenizer, model, memory);

        memory.ObserveContradiction("alpha", "beta", 0.10);
        memory.ObserveContradiction("beta", "gamma", 0.12);
        memory.ObserveContradiction("gamma", "delta", 0.14);

        for (var i = 0; i < 120; i++)
            trainer.TrainStep(new GenesisExample("hello", "hello!"));

        var result = inference.Generate(new GenerationRequest(
            Input: "hello",
            MaxNewTokens: 4));

        Assert.Equal("neural-token+platonic-bias", result.DecisionPath);
        Assert.False(result.UsedPlatonicQuery);
        Assert.Contains("hello", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WhenCheckpointProvidesPlatonicConcepts_ThenNeuralBiasUsesFileContext()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        _ = tokenizer.Encode("alpha beta unrelated");

        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        memory.ObserveContradiction("alpha", "beta", 0.08);

        var checkpointPath = Path.Combine(Path.GetTempPath(), $"genesis-nova-platonic-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(checkpointPath,
                """
                {
                  "PlatonicSpace": {
                    "FaceDimension": 8,
                    "Nodes": [
                      {
                        "Name": "alpha",
                        "PositiveFace": [0,0,0,0,0,0,0,0],
                        "NegativeFace": [0,0,0,0,0,0,0,0],
                        "ObservationCount": 1
                      }
                    ],
                    "Relations": []
                  }
                }
                """);

            var withFile = new GenesisInferenceEngine(tokenizer, model, memory, () => checkpointPath);

            var method = typeof(GenesisInferenceEngine).GetMethod(
                "LoadCheckpointContextConcepts",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            Assert.NotNull(method);

            var concepts = method!.Invoke(withFile, null) as IReadOnlyCollection<string>;
            Assert.NotNull(concepts);
            Assert.NotEmpty(concepts!);
            Assert.Contains("alpha", concepts!, StringComparer.OrdinalIgnoreCase);
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
        }
    }
}
