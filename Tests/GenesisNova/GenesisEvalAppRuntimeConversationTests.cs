using GenesisNova.Core;
using GenesisNova.Persistence;
using GenesisNova.Runtime;

namespace GenesisNova.Tests;

public sealed class GenesisEvalAppRuntimeConversationTests
{
    [Fact]
    public async Task WhenObservingConversation_ThenCheckpointStoresConversationSnapshot()
    {
        var root = Path.Combine(Path.GetTempPath(), $"genesis-runtime-{Guid.NewGuid():N}");
        var config = new GenesisNovaConfig(
            LocalStateDirectory: root,
            AutoPersist: true,
            AutoResume: false);

        try
        {
            var runtime = new GenesisEvalAppRuntime(config);
            var state = await runtime.ObserveConversationAsync(
                userInput: "reset",
                assistantOutput: "reset signal recorded; memory retained",
                resetSignal: true,
                note: "test-reset");

            var checkpointPath = GenesisLocalStateStore.ResolveCheckpointPath(config);
            Assert.True(File.Exists(checkpointPath));
            Assert.True(state.RecentTurnCount >= 2);

            var loaded = GenesisCheckpointStore.LoadForRuntime(checkpointPath, config);
            Assert.NotNull(loaded.Conversation);
            Assert.Contains("reset signal", loaded.Conversation!.Summary);
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}
