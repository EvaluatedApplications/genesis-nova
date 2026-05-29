using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class GenesisTransformDiscoveryTests
{
    [Fact]
    public void WhenTrainingArithmeticOperations_ThenFoldAndLogLinearDiscoveryAreAvailable()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        var training = new[]
        {
            new GenesisExample("1+2", "3"),
            new GenesisExample("2+3", "5"),
            new GenesisExample("3+4", "7"),
            new GenesisExample("4+5", "9"),
            new GenesisExample("2*3", "6"),
            new GenesisExample("3*4", "12"),
            new GenesisExample("4*5", "20"),
            new GenesisExample("5*6", "30"),
            new GenesisExample("8/2", "4"),
            new GenesisExample("9/3", "3"),
            new GenesisExample("12/3", "4"),
            new GenesisExample("15/5", "3")
        };

        for (var i = 0; i < 24; i++)
        {
            foreach (var example in training)
                trainer.TrainStep(example);
        }

        Assert.True(trainer.FoldPathDiscovery.TryPredict("mul", 3, 4, out var foldPrediction, out var route));
        Assert.InRange(foldPrediction, 11.5, 12.5);
        Assert.False(string.IsNullOrWhiteSpace(route));
    }

    [Fact]
    public void WhenInferenceRunsWithSharedDiscovery_ThenInterceptAndMetricsAreUpdated()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var memory = new PlatonicSpaceMemory(faceDimension: 24);
        var trainer = new GenesisTrainer(tokenizer, model, memory);
        var inference = new GenesisInferenceEngine(
            tokenizer,
            model,
            memory,
            foldPathDiscovery: trainer.FoldPathDiscovery,
            transformLibrary: trainer.TransformLibrary,
            transformAccumulator: trainer.TransformAccumulator);

        for (var i = 0; i < 32; i++)
        {
            trainer.TrainStep(new GenesisExample("1+1", "2"));
            trainer.TrainStep(new GenesisExample("2+2", "4"));
            trainer.TrainStep(new GenesisExample("2*3", "6"));
            trainer.TrainStep(new GenesisExample("3*4", "12"));
        }

        var before = trainer.TransformAccumulator.TryGetTransform("mul", out var beforeTransform)
            ? beforeTransform.ObservationCount
            : 0;

        var result = inference.Generate(new GenerationRequest("3*4", MaxNewTokens: 4));

        Assert.True(result.UsedPlatonicQuery);
        var match = System.Text.RegularExpressions.Regex.Match(result.Output, @"-?\d+(\.\d+)?");
        Assert.True(match.Success, $"Expected numeric output, got '{result.Output}'.");
        var numeric = double.Parse(match.Value, System.Globalization.CultureInfo.InvariantCulture);
        Assert.True(numeric > 0);
        Assert.Contains(result.DecisionPath, new[] { "platonic-discovered-transform", "platonic-query-slot-decode" });
        if (result.DecisionPath == "platonic-discovered-transform")
        {
            Assert.Equal("mul", result.RoutedTransform);
            Assert.False(string.IsNullOrWhiteSpace(result.TransformIntercept));
        }

        Assert.True(trainer.TransformAccumulator.TryGetTransform("mul", out var afterTransform));
        Assert.True(afterTransform.ObservationCount >= before);

        var tracked = trainer.TransformLibrary?.GetTransform("mul");
        Assert.NotNull(tracked);
        Assert.True(tracked!.SuccessfulApplications > 0);
    }

    [Fact]
    public void WhenTrainingProgresses_ThenTickPatternPromotionLoopActivates()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        for (var i = 0; i < 12; i++)
        {
            trainer.TrainStep(new GenesisExample("4+5", "9"));
            trainer.TrainStep(new GenesisExample("3+7", "10"));
        }

        Assert.True(trainer.TickPatternPromotions > 0);
    }
}
