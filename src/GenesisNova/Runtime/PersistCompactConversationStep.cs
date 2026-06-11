using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class PersistCompactConversationStep : IStep<GenesisCompactConversationTaskData>
{
    private readonly GenesisCheckpointPersister _persister;

    public PersistCompactConversationStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public ValueTask<GenesisCompactConversationTaskData> ExecuteAsync(GenesisCompactConversationTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _persister.Persist(
            reason: "conversation-compact",
            detail: data.Note ?? "manual-compact");
        return ValueTask.FromResult(data);
    }
}
