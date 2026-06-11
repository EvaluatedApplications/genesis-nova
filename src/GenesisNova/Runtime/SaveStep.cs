using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class SaveStep : IStep<GenesisSaveTaskData>
{
    private readonly GenesisCheckpointPersister _persister;

    public SaveStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public ValueTask<GenesisSaveTaskData> ExecuteAsync(GenesisSaveTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _persister.Persist(reason: "save", explicitPath: data.Path, detail: "manual");
        return ValueTask.FromResult(data with { Saved = true });
    }
}
