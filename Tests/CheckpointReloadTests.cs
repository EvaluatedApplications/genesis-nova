using System;
using System.IO;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Save/reload lifecycle regression. The gym AUTOSAVES every interval AND probes (predicts) every cycle; the predict
/// path runs a reload-on-change guard. A bug had SaveAsync NOT advancing the watermark that guard compares against, so
/// the gym's own autosave looked like an external change and the next predict RELOADED the just-written checkpoint —
/// tearing down + rebuilding model/space/trainer mid-run (lossy, wiped conversation, desynced levels). These tests pin
/// the fix: our own save never triggers a self-reload, while a genuine external change still does.
/// </summary>
public sealed class CheckpointReloadTests
{
    private static GenesisNovaConfig Config(string dir) => new GenesisNovaConfig(
        HiddenSize: 64, AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();

    [Fact]
    public async Task OwnAutosave_DoesNotTriggerSelfReload_OnNextPredict()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir));
            await rt.SaveAsync(rt.AutoCheckpointPath);            // the gym's autosave
            var before = rt.ReloadCount;
            await rt.PredictAsync("1 + 1", 4);                    // would self-reload without the watermark fix
            await rt.PredictAsync("2 + 2", 4);
            Assert.Equal(before, rt.ReloadCount);                 // our own save is NOT an external change
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public async Task ExternalCheckpointChange_StillTriggersReload()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir));
            await rt.SaveAsync(rt.AutoCheckpointPath);
            var before = rt.ReloadCount;
            // Simulate a DIFFERENT process writing a newer checkpoint: bump the watched pointer file's write time.
            File.SetLastWriteTimeUtc(rt.AutoCheckpointPath, DateTime.UtcNow.AddMinutes(5));
            await rt.PredictAsync("1 + 1", 4);                    // a real external change SHOULD be picked up
            Assert.True(rt.ReloadCount > before, "an external checkpoint change still reloads");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
