using EvalApp.Consumer;
using GenesisNova.Infer;

namespace GenesisNova.Runtime;

internal sealed class PredictGpuStep : IStep<GenesisPredictTaskData>
{
    private readonly GenesisRuntimeState _state;

    public PredictGpuStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public ValueTask<GenesisPredictTaskData> ExecuteAsync(GenesisPredictTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var result = _state.Inference.Generate(new GenerationRequest(
            Input: data.Input,
            MaxNewTokens: data.MaxNewTokens));
        _state.Trainer.ObserveInferenceResult(data.Input, result.Output, result);
        return ValueTask.FromResult(data with { Result = result });
    }
}
