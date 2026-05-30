using GenesisNova.Core;
using GenesisNova.Cognition;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class GenesisCheckpointStoreTests
{
    [Fact]
    public void WhenSavingAndLoadingCheckpoint_ThenInferenceRemainsUsable()
    {
        var config = new GenesisNovaConfig(HiddenSize: 32, LearningRate: 0.08);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var platonicSpace = new PlatonicSpaceMemory(faceDimension: 16, seed: 42);
        var trainer = new GenesisTrainer(tokenizer, model, platonicSpace);
        var infer = new GenesisInferenceEngine(tokenizer, model, platonicSpace);

        for (var i = 0; i < 20; i++)
            trainer.TrainStep(new GenesisExample("1 + 1", "2"));

        var before = infer.Generate(new GenerationRequest("1 + 1", 4));
        var checkpointPath = Path.Combine(Path.GetTempPath(), $"genesis-nova-{Guid.NewGuid():N}.json");
        try
        {
            GenesisCheckpointStore.Save(checkpointPath, config, tokenizer, model, platonicSpace.ExportSnapshot());
            var loaded = GenesisCheckpointStore.Load(checkpointPath);
            var loadedMemory = loaded.PlatonicSpace is null
                ? new PlatonicSpaceMemory(faceDimension: Math.Max(4, loaded.Config.HiddenSize / 2), seed: loaded.Config.Seed)
                : new PlatonicSpaceMemory(faceDimension: loaded.PlatonicSpace.FaceDimension, seed: loaded.Config.Seed);
            if (loaded.PlatonicSpace is not null)
                loadedMemory.ImportSnapshot(loaded.PlatonicSpace);
            var loadedInfer = new GenesisInferenceEngine(loaded.Tokenizer, loaded.Model, loadedMemory);
            var after = loadedInfer.Generate(new GenerationRequest("1 + 1", 4));

            Assert.False(string.IsNullOrWhiteSpace(before.Output));
            Assert.False(string.IsNullOrWhiteSpace(after.Output));
            Assert.NotNull(loaded.PlatonicSpace);
            Assert.NotEmpty(loaded.PlatonicSpace!.Nodes);
            Assert.Equal(platonicSpace.NodeCount, loaded.PlatonicSpace!.Nodes.Length);
            Assert.Null(loaded.Conversation);
            Assert.Null(loaded.AutonomousTraining);
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
        }
    }

    [Fact]
    public void WhenSavingAndLoadingCheckpoint_ThenAutonomousHistoryIsPreserved()
    {
        var config = new GenesisNovaConfig(HiddenSize: 16);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var checkpointPath = Path.Combine(Path.GetTempPath(), $"genesis-nova-{Guid.NewGuid():N}.json");
        var history = new[]
        {
            new GenesisAutonomousTrainingRound(
                Round: 1,
                CreatorName: "language:greet",
                SampleCount: 24,
                Difficulty: 0,
                Epochs: 1,
                Report: new GenesisTrainingReport(
                    Epochs: 1,
                    ExampleCount: 1,
                    AverageLoss: new GenesisStepLoss(0.09, 0, 0, 0, 0, 0.09),
                    ContradictionRate: 0,
                    ConservationDrift: 0,
                    MemoryOverwriteRate: 0,
                    IntrospectionCycles: 0,
                    PendingQueueDepth: 0))
        };

        try
        {
            GenesisCheckpointStore.Save(
                checkpointPath,
                config,
                tokenizer,
                model,
                autonomousTraining: new GenesisAutonomousTrainingSnapshot(history));

            var loaded = GenesisCheckpointStore.Load(checkpointPath);
            Assert.NotNull(loaded.AutonomousTraining);
            Assert.Single(loaded.AutonomousTraining!.History);
            Assert.Equal("language:greet", loaded.AutonomousTraining.History[0].CreatorName);
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
        }
    }
}
