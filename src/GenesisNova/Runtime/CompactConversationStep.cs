using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class CompactConversationStep : IStep<GenesisCompactConversationTaskData>
{
    private readonly GenesisRuntimeState _state;

    public CompactConversationStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public ValueTask<GenesisCompactConversationTaskData> ExecuteAsync(GenesisCompactConversationTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var compacted = _state.Conversation.Compact();
        return ValueTask.FromResult(data with
        {
            Compacted = compacted,
            ContextBrief = _state.Conversation.BuildContextBrief(),
            RecentTurnCount = _state.Conversation.RecentTurns.Count
        });
    }
}
