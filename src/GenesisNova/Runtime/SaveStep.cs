namespace GenesisNova.Runtime;

internal sealed class SaveStep
{
    private readonly GenesisCheckpointPersister _persister;

    public SaveStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public GenesisSaveTaskData Execute(GenesisSaveTaskData data)
    {
        _persister.Persist(reason: "save", explicitPath: data.Path, detail: "manual");
        return data with { Saved = true };
    }
}
