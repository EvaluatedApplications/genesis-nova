using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

/// <summary>
/// Clean-slate smoke test. Verifies the core training interface wires together and
/// runs end-to-end without throwing: tokenizer -> neural model + platonic space -> trainer.
/// Deliberately the ONLY test in the suite; richer coverage is added back deliberately.
/// </summary>
public sealed class GenesisSmokeTest
{
    [Fact]
    public void Tokenizer_RoundTrips()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();

        var tokens = tokenizer.Encode("Hello   WORLD", addBos: true, addEos: true);
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("hello world", decoded);
        Assert.True(tokenizer.VocabularySize >= 5);
    }

    [Fact]
    public void TrainingInterface_RunsSingleStep_WithoutThrowing()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 24, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 16, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        var loss = trainer.TrainStep(new GenesisExample("say one", "one"));

        Assert.True(loss.TokenLoss >= 0.0);
        Assert.True(loss.TotalLoss >= 0.0);
        Assert.False(double.IsNaN(loss.TotalLoss));
    }
}
