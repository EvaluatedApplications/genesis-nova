using System;
using System.IO;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Persistence;
using GenesisNova.Runtime;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// CRASH ATOMICITY across the model dir + substrate dir. A save writes them (and the pointer) separately, so a crash
/// between writes can pair model-gen N with substrate-gen N-1 — an inconsistent ("torn") checkpoint. Each save now
/// stamps a single GENERATION on both manifests and the pointer; a load detects a mismatch and falls back to the
/// last-good (consistent) checkpoint instead of resuming a torn brain. These tests pin that.
/// </summary>
public sealed class CheckpointAtomicityTests
{
    private static GenesisNovaConfig Config(string dir) => new GenesisNovaConfig(
        Backend: ComputeBackend.Cpu, HiddenSize: 64, AutoPersist: true, AutoResume: true,
        LocalStateDirectory: dir).WithProductionMechanisms();

    [Fact]
    public async Task CleanSave_IsConsistent_TornSave_IsDetected()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-atom-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var rt = new GenesisEvalAppRuntime(Config(dir));
            await rt.SaveAsync(rt.AutoCheckpointPath);
            Assert.True(GenesisShardedCheckpointStore.IsConsistent(rt.AutoCheckpointPath), "a clean save is consistent");

            // Simulate a torn save: rewrite the pointer (the commit) with a generation that no longer matches the dirs.
            File.WriteAllText(rt.AutoCheckpointPath, GenesisShardedCheckpointStore.PointerJson("torn-deadbeef"));
            Assert.False(GenesisShardedCheckpointStore.IsConsistent(rt.AutoCheckpointPath), "a generation mismatch is a torn save");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public async Task TornCheckpoint_OnResume_FallsBackToLastGood()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-atom-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var cfg = Config(dir);
            var rt1 = new GenesisEvalAppRuntime(cfg);
            await rt1.SaveAsync(rt1.AutoCheckpointPath);                       // the live autosave
            var lastGood = GenesisLocalStateStore.ResolveLastGoodCheckpointPath(cfg);
            await rt1.SaveAsync(lastGood);                                     // a consistent last-good snapshot
            Assert.True(GenesisShardedCheckpointStore.IsConsistent(lastGood));

            // TEAR the autosave (as a mid-save crash would): pointer generation no longer matches the dirs.
            File.WriteAllText(rt1.AutoCheckpointPath, GenesisShardedCheckpointStore.PointerJson("torn-deadbeef"));
            Assert.False(GenesisShardedCheckpointStore.IsConsistent(rt1.AutoCheckpointPath));

            // RESUME: a fresh runtime bootstraps from the torn autosave — it must fall back to last-good, not crash.
            var rt2 = new GenesisEvalAppRuntime(cfg);
            var r = (await rt2.PredictAsync("1 + 1", 4)).Result;
            Assert.NotNull(r);
            Assert.Equal("2", r!.Output?.Trim());                             // loaded a working model
            var journal = GenesisLocalStateStore.ResolveJournalPath(cfg);
            Assert.Contains("torn-checkpoint-fallback", File.ReadAllText(journal)); // the fallback actually fired
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
