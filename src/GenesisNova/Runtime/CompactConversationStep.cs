namespace GenesisNova.Runtime;

internal sealed class CompactConversationStep
{
    private readonly GenesisRuntimeState _state;

    public CompactConversationStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public GenesisCompactConversationTaskData Execute(GenesisCompactConversationTaskData data)
    {
        var compacted = _state.Conversation.Compact();
        return data with
        {
            Compacted = compacted,
            ContextBrief = _state.Conversation.BuildContextBrief(),
            RecentTurnCount = _state.Conversation.RecentTurns.Count
        };
    }
}
