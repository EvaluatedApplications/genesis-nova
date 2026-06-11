using EvalApp.Consumer;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class LoadExamplesStep : IStep<GenesisTrainTaskData>
{
    public ValueTask<GenesisTrainTaskData> ExecuteAsync(GenesisTrainTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        return ValueTask.FromResult(data with { Examples = GenesisTrainingDataLoader.LoadFromFile(data.FilePath) });
    }
}
