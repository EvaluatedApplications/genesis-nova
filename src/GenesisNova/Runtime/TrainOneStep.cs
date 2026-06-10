namespace GenesisNova.Runtime;

internal sealed class TrainOneStep
{
    private readonly GenesisRuntimeState _state;

    public TrainOneStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public GenesisTrainOneTaskData Execute(GenesisTrainOneTaskData data)
    {
        var loss = _state.Trainer.TrainStep(data.Example);
        return data with { Loss = loss };
    }
}
