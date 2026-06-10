namespace GenesisNova.Runtime;

internal sealed class PersistConversationStep
{
    private readonly GenesisCheckpointPersister _persister;

    public PersistConversationStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public GenesisConversationTaskData Execute(GenesisConversationTaskData data)
    {
        _persister.Persist(
            reason: data.ResetSignal ? "conversation-reset" : "conversation",
            detail: data.Note ?? data.UserInput);
        return data;
    }
}
