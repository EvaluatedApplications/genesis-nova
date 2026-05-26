using GenesisNova.Core;
using GenesisNova.Persistence;

namespace GenesisNova.Tests;

public sealed class GenesisLocalStateStoreTests
{
    [Fact]
    public void WhenAppendingJournalEntry_ThenWritesLocalJournalFile()
    {
        var root = Path.Combine(Path.GetTempPath(), $"genesis-state-{Guid.NewGuid():N}");
        var config = new GenesisNovaConfig(LocalStateDirectory: root);

        try
        {
            GenesisLocalStateStore.AppendJournalEntry(
                config,
                "introspect",
                detail: "cycles=4",
                exampleCount: 12,
                loss: 0.42,
                queueDepth: 7);

            var journalPath = GenesisLocalStateStore.ResolveJournalPath(config);
            Assert.True(File.Exists(journalPath));

            var content = File.ReadAllText(journalPath);
            Assert.Contains("introspect", content);
            Assert.Contains("cycles=4", content);
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}
