using GenesisNova.Core;
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
        var trainer = new GenesisTrainer(tokenizer, model);
        var inference = new GenesisInferenceEngine(tokenizer, model);

        for (var i = 0; i < 30; i++)
            trainer.TrainStep(new GenesisExample("say hello", "hello", RouteLabel: 1));

        var result = inference.Generate(new GenerationRequest("say hello", MaxNewTokens: 4));

        Assert.Contains("hello", result.Output);
    }

    [Fact]
    public void WhenExpressionsHaveDifferentOrder_ThenModelLearnsDifferentOutputs()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 48, LearningRate: 0.06));
        var trainer = new GenesisTrainer(tokenizer, model);
        var inference = new GenesisInferenceEngine(tokenizer, model);

        for (var i = 0; i < 120; i++)
        {
            trainer.TrainStep(new GenesisExample("2-1", "1", RouteLabel: 1));
            trainer.TrainStep(new GenesisExample("1-2", "-1", RouteLabel: 1));
        }

        var a = inference.Generate(new GenerationRequest("2-1", MaxNewTokens: 3));
        var b = inference.Generate(new GenerationRequest("1-2", MaxNewTokens: 3));

        Assert.NotEqual(a.Output, b.Output);
    }
}
