namespace GenesisNova.Runtime;

internal sealed class ObserveConversationStep
{
    private readonly GenesisRuntimeState _state;

    public ObserveConversationStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public GenesisConversationTaskData Execute(GenesisConversationTaskData data)
    {
        _state.Conversation.ObserveTurn("user", data.UserInput, resetSignal: data.ResetSignal, note: data.Note);
        _state.Conversation.ObserveTurn("assistant", data.AssistantOutput, note: data.Note);
        return data with
        {
            ContextBrief = _state.Conversation.BuildContextBrief(),
            RecentTurnCount = _state.Conversation.RecentTurns.Count
        };
    }
}
