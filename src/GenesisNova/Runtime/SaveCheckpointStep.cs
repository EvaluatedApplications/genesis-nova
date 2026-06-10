namespace GenesisNova.Runtime;

internal sealed class SaveCheckpointStep
{
    private readonly BestLossTracker _lossTracker;
    private readonly GenesisCheckpointPersister _persister;

    public SaveCheckpointStep(BestLossTracker lossTracker, GenesisCheckpointPersister persister)
    {
        _lossTracker = lossTracker;
        _persister = persister;
    }

    public GenesisTrainTaskData Execute(GenesisTrainTaskData data)
    {
        var currentLoss = data.Report?.AverageLoss.TotalLoss ?? double.MaxValue;
        if (currentLoss < _lossTracker.BestLoss)
        {
            _lossTracker.BestLoss = currentLoss;
            _persister.Persist(
                reason: "train-improved",
                explicitPath: data.SavePath,
                detail: $"epochs={data.Epochs} loss={currentLoss:F4}",
                exampleCount: data.Examples?.Count ?? 0,
                loss: currentLoss);
        }
        return data;
    }
}
