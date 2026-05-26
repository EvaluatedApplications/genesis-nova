using GenesisNova.Core;
using GenesisNova.Model;

namespace GenesisNova.Tests;

public sealed class GenesisNeuralModelTests
{
    [Fact]
    public void WhenExpandingHiddenSize_ThenExistingWeightsArePreserved()
    {
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 8, LearningRate: 0.05));
        model.EnsureVocabularySize(4);

        var before = model.Export();
        model.EnsureHiddenSize(16);
        var after = model.Export();

        Assert.Equal(16, model.HiddenSize);
        Assert.Equal(before.Embeddings[0, 0], after.Embeddings[0, 0]);
        Assert.Equal(before.OutputWeights[0, 0], after.OutputWeights[0, 0]);
        Assert.Equal(before.OutputBias[0], after.OutputBias[0]);
    }
}
