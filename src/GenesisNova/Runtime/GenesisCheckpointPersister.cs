using GenesisNova.Core;
using GenesisNova.Persistence;

namespace GenesisNova.Runtime;

internal sealed class GenesisCheckpointPersister
{
    private readonly GenesisRuntimeState _state;
    private readonly GenesisNovaConfig _runtimeConfig;
    private readonly AutonomousHistoryStore _historyStore;

    public GenesisCheckpointPersister(
        GenesisRuntimeState state,
        GenesisNovaConfig runtimeConfig,
        AutonomousHistoryStore historyStore)
    {
        _state = state;
        _runtimeConfig = runtimeConfig;
        _historyStore = historyStore;
    }

    public void Persist(
        string reason,
        string? explicitPath = null,
        string? detail = null,
        int? exampleCount = null,
        double? loss = null)
    {
        var snapshotConfig = CreateCheckpointConfig();
        var autoPath = GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);
        var wrote = false;

        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            GenesisCheckpointStore.Save(
                explicitPath,
                snapshotConfig,
                _state.Tokenizer,
                _state.Model,
                platonicSpace: _state.Memory.ExportSnapshot(),
                conversation: _state.Conversation.ExportSnapshot(),
                autonomousTraining: _historyStore.Export());
            wrote = true;
        }

        if (_runtimeConfig.AutoPersist)
        {
            if (string.IsNullOrWhiteSpace(explicitPath) ||
                !string.Equals(explicitPath, autoPath, StringComparison.OrdinalIgnoreCase))
            {
                GenesisCheckpointStore.Save(
                    autoPath,
                    snapshotConfig,
                    _state.Tokenizer,
                    _state.Model,
                    platonicSpace: _state.Memory.ExportSnapshot(),
                    conversation: _state.Conversation.ExportSnapshot(),
                    autonomousTraining: _historyStore.Export());
                wrote = true;
            }
        }

        if (wrote)
        {
            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                reason,
                detail,
                exampleCount,
                loss);
        }
    }

    private GenesisNovaConfig CreateCheckpointConfig()
        => _state.Config with
        {
            HiddenSize = _state.Model.HiddenSize,
            AutoPersist = _runtimeConfig.AutoPersist,
            AutoResume = _runtimeConfig.AutoResume,
            AutoScaleVram = _runtimeConfig.AutoScaleVram,
            TargetVramUtilization = _runtimeConfig.TargetVramUtilization,
            ReserveVramMb = _runtimeConfig.ReserveVramMb,
            LocalStateDirectory = _runtimeConfig.LocalStateDirectory
        };
}
