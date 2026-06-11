using System.Text.Json;
using GenesisNova.Core;

namespace GenesisNova.Persistence;

public static class GenesisLocalStateStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false
    };

    public static string ResolveStateDirectory(GenesisNovaConfig config)
    {
        if (!string.IsNullOrWhiteSpace(config.LocalStateDirectory))
            return config.LocalStateDirectory;

        // Use a stable user-local folder on the system drive by default.
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova",
            "models");
    }

    public static string ResolveCheckpointPath(GenesisNovaConfig config)
        => Path.Combine(ResolveStateDirectory(config), "genesis-nova.autosave.checkpoint.json");

    public static string ResolveLastGoodCheckpointPath(GenesisNovaConfig config)
        => Path.Combine(ResolveStateDirectory(config), "genesis-nova.autosave.lastgood.json");

    public static string ResolveLastGoodProbeScorePath(GenesisNovaConfig config)
        => Path.Combine(ResolveStateDirectory(config), "genesis-nova.autosave.lastgood.probe.json");

    public static string ResolveJournalPath(GenesisNovaConfig config)
        => Path.Combine(ResolveStateDirectory(config), "genesis-nova.journal.jsonl");

    public static void AppendJournalEntry(
        GenesisNovaConfig config,
        string @event,
        string? detail = null,
        int? exampleCount = null,
        double? loss = null,
        int? queueDepth = null)
    {
        var path = ResolveJournalPath(config);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        var entry = new GenesisStateJournalEntry(
            Timestamp: DateTimeOffset.UtcNow,
            Event: @event,
            Detail: detail,
            ExampleCount: exampleCount,
            Loss: loss,
            QueueDepth: queueDepth);

        File.AppendAllText(path, JsonSerializer.Serialize(entry, JsonOptions) + Environment.NewLine);
    }

    public static bool TryResolveBootstrapCheckpoint(GenesisNovaConfig config, out string path)
    {
        var autoPath = ResolveCheckpointPath(config);
        if (File.Exists(autoPath))
        {
            path = autoPath;
            return true;
        }

        path = autoPath;
        return false;
    }
}

public sealed record GenesisStateJournalEntry(
    DateTimeOffset Timestamp,
    string Event,
    string? Detail = null,
    int? ExampleCount = null,
    double? Loss = null,
    int? QueueDepth = null);
