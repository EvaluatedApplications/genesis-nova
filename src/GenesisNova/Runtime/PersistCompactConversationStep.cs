namespace GenesisNova.Runtime;

internal sealed class PersistCompactConversationStep
{
    private readonly GenesisCheckpointPersister _persister;

    public PersistCompactConversationStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public GenesisCompactConversationTaskData Execute(GenesisCompactConversationTaskData data)
    {
        _persister.Persist(
            reason: "conversation-compact",
            detail: data.Note ?? "manual-compact");
        return data;
    }
}
