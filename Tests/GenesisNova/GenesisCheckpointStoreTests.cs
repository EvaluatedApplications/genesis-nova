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
        var memory = new PlatonicSpaceMemory(faceDimension: 16, seed: 42);
        var cognition = new PlatonicIntrospectionEngine(memory);
        var trainer = new GenesisTrainer(tokenizer, model, cognition);
        var infer = new GenesisInferenceEngine(tokenizer, model, cognition);

        for (var i = 0; i < 20; i++)
            trainer.TrainStep(new GenesisExample("say green", "green", RouteLabel: 2));

        var before = infer.Generate(new GenerationRequest("say green", 4));
        var checkpointPath = Path.Combine(Path.GetTempPath(), $"genesis-nova-{Guid.NewGuid():N}.json");
        try
        {
            GenesisCheckpointStore.Save(checkpointPath, config, tokenizer, model, trainer.ExportCognitionSnapshot());
            var loaded = GenesisCheckpointStore.Load(checkpointPath);
            var loadedInfer = new GenesisInferenceEngine(loaded.Tokenizer, loaded.Model);
            var after = loadedInfer.Generate(new GenerationRequest("say green", 4));

            Assert.False(string.IsNullOrWhiteSpace(before.Output));
            Assert.False(string.IsNullOrWhiteSpace(after.Output));
            Assert.NotNull(loaded.Cognition);
            Assert.Null(loaded.Conversation);
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
        }
    }
}
