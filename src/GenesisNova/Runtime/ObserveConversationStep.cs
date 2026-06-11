using EvalApp.Consumer;

namespace GenesisNova.Runtime;

internal sealed class ObserveConversationStep : IStep<GenesisConversationTaskData>
{
    private readonly GenesisRuntimeState _state;

    public ObserveConversationStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public ValueTask<GenesisConversationTaskData> ExecuteAsync(GenesisConversationTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        _state.Conversation.ObserveTurn("user", data.UserInput, resetSignal: data.ResetSignal, note: data.Note);
        _state.Conversation.ObserveTurn("assistant", data.AssistantOutput, note: data.Note);
        return ValueTask.FromResult(data with
        {
            ContextBrief = _state.Conversation.BuildContextBrief(),
            RecentTurnCount = _state.Conversation.RecentTurns.Count
        });
    }
}
