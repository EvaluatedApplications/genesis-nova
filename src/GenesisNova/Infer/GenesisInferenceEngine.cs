using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Cognition;

namespace GenesisNova.Infer;

public sealed class GenesisInferenceEngine
{
    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly PlatonicIntrospectionEngine? _cognition;

    public GenesisInferenceEngine(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicIntrospectionEngine? cognition = null)
    {
        _tokenizer = tokenizer;
        _model = model;
        _cognition = cognition;
    }

    public GenerationResult Generate(GenerationRequest request)
    {
        var inputTokens = _tokenizer.Encode(request.Input);

        var generated = new List<int>(Math.Max(1, request.MaxNewTokens));
        var prev = _tokenizer.BosTokenId;
        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            var next = _model.PredictNextToken(
                inputTokens,
                prev,
                stepIndex: i,
                disallowToken: i == 0 ? _tokenizer.EosTokenId : null,
                penalizedTokens: generated,
                repetitionPenalty: 0.35);
            generated.Add(next);
            if (next == _tokenizer.EosTokenId)
                break;
            prev = next;
        }

        var result = new GenerationResult(
            Output: _tokenizer.Decode(generated),
            GeneratedTokens: generated.ToArray());

        _cognition?.QueueInference(request.Input, result.Output, 0, 0.0);
        return result;
    }
}
