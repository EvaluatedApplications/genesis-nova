using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class GenesisTrainerParityTests
{
    [Fact]
    public void WhenObservingInferenceResult_ThenPlatonicSpaceIsUpdated()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 24, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        trainer.ObserveInferenceResult("1+1", "2");

        Assert.True(memory.NodeCount > 0);
        Assert.True(memory.RelationCount > 0);
    }
}
