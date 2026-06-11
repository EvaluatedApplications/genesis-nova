using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class SaveCheckpointStep : IStep<GenesisTrainTaskData>
{
    private readonly BestLossTracker _lossTracker;
    private readonly GenesisCheckpointPersister _persister;

    public SaveCheckpointStep(BestLossTracker lossTracker, GenesisCheckpointPersister persister)
    {
        _lossTracker = lossTracker;
        _persister = persister;
    }

    public ValueTask<GenesisTrainTaskData> ExecuteAsync(GenesisTrainTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var currentLoss = data.Report?.AverageLoss.TotalLoss ?? double.MaxValue;
        var improved = currentLoss < _lossTracker.BestLoss;

        if (improved)
            _lossTracker.BestLoss = currentLoss;

        // CRITICAL: Always persist after training, not just on improvement
        // This ensures model state is saved regardless of loss trajectory
        // allowing recovery from any training state, not just best checkpoints
        var reason = improved ? "train-improved" : "train-completed";
        _persister.Persist(
            reason: reason,
            explicitPath: data.SavePath,
            detail: $"epochs={data.Epochs} loss={currentLoss:F4} improved={improved}",
            exampleCount: data.Examples?.Count ?? 0,
            loss: currentLoss);

        return ValueTask.FromResult(data);
    }
}
