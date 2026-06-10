using GenesisNova.Infer;

namespace GenesisNova.Runtime;

internal sealed class PredictGpuStep
{
    private readonly GenesisRuntimeState _state;

    public PredictGpuStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public GenesisPredictTaskData Execute(GenesisPredictTaskData data)
    {
        var result = _state.Inference.Generate(new GenerationRequest(
            Input: data.Input,
            MaxNewTokens: data.MaxNewTokens));
        _state.Trainer.ObserveInferenceResult(data.Input, result.Output);
        return data with { Result = result };
    }
}
