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
    private static GenesisNovaConfig Config(string dir, bool watch = false) => new GenesisNovaConfig(
        HiddenSize: 64, AutoPersist: true, AutoResume: true, WatchExternalCheckpoint: watch, LocalStateDirectory: dir).WithProductionMechanisms();

    [Fact]
    public async Task PredictReload_OffByDefault_NoSelfReload()
    {
        // DEFAULT (watch off): predicts NEVER reload — even after our own save AND even if the file's timestamp moves.
        // This is the real-world default that kills the self-reload that degraded the model on resume.
        var dir = Path.Combine(Path.GetTempPath(), "gn-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir));
            await rt.SaveAsync(rt.AutoCheckpointPath);
            File.SetLastWriteTimeUtc(rt.AutoCheckpointPath, DateTime.UtcNow.AddMinutes(5)); // looks like a change
            var before = rt.ReloadCount;
            await rt.PredictAsync("1 + 1", 4);
            await rt.PredictAsync("2 + 2", 4);
            Assert.Equal(before, rt.ReloadCount); // no predict-time reload when not watching
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public async Task OwnSave_DoesNotSelfReload_EvenWhenWatching()
    {
        // Even with watching ON, our OWN save advances the watermark (fix A), so it is not mistaken for an external change.
        var dir = Path.Combine(Path.GetTempPath(), "gn-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir, watch: true));
            await rt.SaveAsync(rt.AutoCheckpointPath);
            var before = rt.ReloadCount;
            await rt.PredictAsync("1 + 1", 4);
            Assert.Equal(before, rt.ReloadCount);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public async Task ExternalChange_ReloadsWhenWatching()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir, watch: true));
            await rt.SaveAsync(rt.AutoCheckpointPath);
            var before = rt.ReloadCount;
            File.SetLastWriteTimeUtc(rt.AutoCheckpointPath, DateTime.UtcNow.AddMinutes(5)); // external writer
            await rt.PredictAsync("1 + 1", 4);
            Assert.True(rt.ReloadCount > before, "an external checkpoint change still reloads when watching is enabled");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
