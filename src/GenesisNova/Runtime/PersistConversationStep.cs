using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class PersistConversationStep : IStep<GenesisConversationTaskData>
{
    private readonly GenesisCheckpointPersister _persister;

    public PersistConversationStep(GenesisCheckpointPersister persister)
    {
        _persister = persister;
    }

    public ValueTask<GenesisConversationTaskData> ExecuteAsync(GenesisConversationTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _persister.Persist(
            reason: data.ResetSignal ? "conversation-reset" : "conversation",
            detail: data.Note ?? data.UserInput);
        return ValueTask.FromResult(data);
    }
}
