using System.Linq;
using System.Text.Json;
using GenesisNova.Core;
using GenesisNova.Persistence;

namespace GenesisNova.Runtime;

internal sealed class GenesisCheckpointPersister
{
    private readonly GenesisRuntimeState _state;
    private readonly GenesisNovaConfig _runtimeConfig;
    private readonly AutonomousHistoryStore _historyStore;
    private readonly GenesisProbeSet _probeSet = new();
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };
    private const double LastGoodTolerance = 0.10;

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
        var trainerLearningStateJson = TryExportTrainerLearningState();
        var grammarRoles = _state.Inference.ExportGrammarRoles()
            .Select(r => new GrammarRoleSnapshot(r.Token, r.Present, r.Absent, r.AsAnswer, r.AsCopula))
            .ToArray();
        var wrote = false;
        var wroteLatest = false;

        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            GenesisCheckpointStore.Save(
                explicitPath,
                snapshotConfig,
                _state.Tokenizer,
                _state.Model,
                platonicSpace: _state.Memory.ExportSnapshot(),
                conversation: _state.Conversation.ExportSnapshot(),
                autonomousTraining: _historyStore.Export(),
                trainerLearningStateJson: trainerLearningStateJson,
                grammarRoles: grammarRoles);
            wrote = true;
            wroteLatest = string.Equals(explicitPath, autoPath, StringComparison.OrdinalIgnoreCase);
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
                    autonomousTraining: _historyStore.Export(),
                    trainerLearningStateJson: trainerLearningStateJson);
                wrote = true;
                wroteLatest = true;
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

        if (wroteLatest && IsTrainingCompletionReason(reason))
            GateLastGoodPromotion(reason, detail, exampleCount, loss, autoPath);
    }

    private string? TryExportTrainerLearningState()
    {
        try
        {
            var state = _state.Trainer.ExportLearningState();
            return state is null ? null : JsonSerializer.Serialize(state, JsonOptions);
        }
        catch (Exception ex)
        {
            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                "trainer-learning-state-export-failed",
                detail: ex.Message);
            return null;
        }
    }

    private void GateLastGoodPromotion(
        string reason,
        string? detail,
        int? exampleCount,
        double? loss,
        string autoPath)
    {
        try
        {
            var report = _probeSet.Evaluate(_state);
            var previous = ReadLastGoodProbeScore();
            if (previous is null || report.Score >= previous.Score - LastGoodTolerance)
            {
                var lastGoodPath = GenesisLocalStateStore.ResolveLastGoodCheckpointPath(_runtimeConfig);
                GenesisCheckpointStore.CopyCheckpointFile(autoPath, lastGoodPath);
                WriteLastGoodProbeScore(new LastGoodProbeScore(report.TimestampUtc, report.Score));
                GenesisLocalStateStore.AppendJournalEntry(
                    _runtimeConfig,
                    "last-good-promoted",
                    detail: $"{detail}; reason={reason} probe={report.Score:F2} previous={(previous?.Score.ToString("F2") ?? "none")}",
                    exampleCount: exampleCount,
                    loss: loss);
                return;
            }

            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                "last-good-regression",
                detail: $"{detail}; reason={reason} probe={report.Score:F2} previous={previous.Score:F2} tolerance={LastGoodTolerance:F2}; latest not promoted",
                exampleCount: exampleCount,
                loss: loss);
        }
        catch (Exception ex)
        {
            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                "last-good-probe-failed",
                detail: $"{detail}; reason={reason}; {ex.Message}",
                exampleCount: exampleCount,
                loss: loss);
        }
    }

    private LastGoodProbeScore? ReadLastGoodProbeScore()
    {
        var path = GenesisLocalStateStore.ResolveLastGoodProbeScorePath(_runtimeConfig);
        if (!File.Exists(path))
            return null;

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<LastGoodProbeScore>(json, JsonOptions);
        }
        catch
        {
            return null;
        }
    }

    private void WriteLastGoodProbeScore(LastGoodProbeScore score)
    {
        var path = GenesisLocalStateStore.ResolveLastGoodProbeScorePath(_runtimeConfig);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        var tempPath = $"{path}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(tempPath, JsonSerializer.Serialize(score, JsonOptions));
            File.Move(tempPath, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private static bool IsTrainingCompletionReason(string reason)
        => reason is "train-improved" or "train-completed" or "auto-train-improved" or "auto-train-completed";

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

    private sealed record LastGoodProbeScore(DateTimeOffset TimestampUtc, double Score);
}
