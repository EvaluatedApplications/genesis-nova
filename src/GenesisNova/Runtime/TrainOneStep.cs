using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class TrainOneStep : IStep<GenesisTrainOneTaskData>
{
    private readonly GenesisRuntimeState _state;

    public TrainOneStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public ValueTask<GenesisTrainOneTaskData> ExecuteAsync(GenesisTrainOneTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var loss = _state.Trainer.TrainStep(data.Example);
        return ValueTask.FromResult(data with { Loss = loss });
    }
}
