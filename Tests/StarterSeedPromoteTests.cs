using System.IO;
using GenesisNova.Persistence;
using Xunit;

namespace GenesisNova.Tests;

// Promote-a-warmed-checkpoint-to-the-repo-starter + seed-a-fresh-start-from-it (MODEL_STORAGE.md). Pure file ops — no
// model load — so it's a FAST test: a local checkpoint promotes to a starter location, a DIFFERENT empty location seeds
// from that starter verbatim, an EXISTING local fork is never overwritten, and a torn/absent starter never seeds.
public sealed class StarterSeedPromoteTests
{
    // A minimal-but-valid sharded checkpoint (pointer + model dir + substrate dir) all stamped the SAME generation,
    // so IsConsistent passes. Returns the pointer path.
    private static string MakeCheckpoint(string dir, string baseName, string generation)
    {
        Directory.CreateDirectory(dir);
        var pointer = Path.Combine(dir, baseName + ".json");
        File.WriteAllText(pointer, GenesisShardedCheckpointStore.PointerJson(generation));
        foreach (var d in new[] { baseName, baseName + ".platonic" })
        {
            var md = Path.Combine(dir, d);
            Directory.CreateDirectory(Path.Combine(md, "shards"));
            File.WriteAllText(Path.Combine(md, "manifest.json"),
                $"{{\"FormatVersion\":1,\"ModelVersion\":1,\"ShardBytes\":33554432,\"CreatedUtc\":\"2026-01-01T00:00:00Z\",\"Sections\":{{}},\"Generation\":\"{generation}\"}}");
            File.WriteAllBytes(Path.Combine(md, "shards", d + "-shard.gnv"), new byte[] { 1, 2, 3, 4 });
        }
        return pointer;
    }

    [Fact]
    public void Promote_Then_SeedFreshStart_RoundTrips_And_NeverOverwritesAnExistingFork()
    {
        var root = Path.Combine(Path.GetTempPath(), "gn-seed-" + System.Guid.NewGuid().ToString("N"));
        try
        {
            // A warmed LOCAL gym checkpoint (mirrors the real gym layout) + an empty STARTER + a fresh-machine LOCAL.
            var localPointer = MakeCheckpoint(Path.Combine(root, "gym"), "genesis-nova.autosave.checkpoint", "GEN-A");
            var starterPointer = Path.Combine(root, "models", "genesis-nova.json");
            var freshPointer = Path.Combine(root, "fresh-machine", "genesis-nova.autosave.checkpoint.json");

            // PROMOTE: the warmed local → the committed starter.
            GenesisShardedCheckpointStore.PromoteToStarter(localPointer, starterPointer);
            Assert.True(File.Exists(starterPointer));
            Assert.True(File.Exists(Path.Combine(root, "models", "genesis-nova", "manifest.json")));
            Assert.True(File.Exists(Path.Combine(root, "models", "genesis-nova.platonic", "manifest.json")));
            Assert.True(GenesisShardedCheckpointStore.IsConsistent(starterPointer));

            // SEED a fresh machine FROM the starter — pointer + model + substrate land verbatim, consistent.
            Assert.True(GenesisShardedCheckpointStore.SeedFromStarter(starterPointer, freshPointer));
            Assert.True(File.Exists(freshPointer));
            Assert.True(File.Exists(Path.Combine(root, "fresh-machine", "genesis-nova.autosave.checkpoint", "manifest.json")));
            Assert.True(File.Exists(Path.Combine(root, "fresh-machine", "genesis-nova.autosave.checkpoint.platonic", "shards", "genesis-nova.autosave.checkpoint.platonic-shard.gnv")));
            Assert.True(GenesisShardedCheckpointStore.IsConsistent(freshPointer));

            // NO-OP once a local fork exists: a second seed (now the pointer is present) must NOT overwrite it.
            File.WriteAllText(freshPointer, GenesisShardedCheckpointStore.PointerJson("FORK-EVOLVED"));
            Assert.False(GenesisShardedCheckpointStore.SeedFromStarter(starterPointer, freshPointer));
            Assert.Equal("FORK-EVOLVED", GenesisShardedCheckpointStore.ReadPointerGeneration(freshPointer));
        }
        finally { try { Directory.Delete(root, true); } catch { } }
    }

    [Fact]
    public void SeedFromStarter_Refuses_Absent_And_Torn_Starters()
    {
        var root = Path.Combine(Path.GetTempPath(), "gn-seed-" + System.Guid.NewGuid().ToString("N"));
        try
        {
            var freshPointer = Path.Combine(root, "fresh", "genesis-nova.autosave.checkpoint.json");

            // ABSENT starter → no seed (fresh empty brain).
            Assert.False(GenesisShardedCheckpointStore.SeedFromStarter(Path.Combine(root, "nope", "genesis-nova.json"), freshPointer));
            Assert.False(File.Exists(freshPointer));

            // TORN starter: pointer says GEN-A but the model manifest says GEN-B → inconsistent → refuse to seed.
            var starterPointer = MakeCheckpoint(Path.Combine(root, "models"), "genesis-nova", "GEN-A");
            File.WriteAllText(Path.Combine(root, "models", "genesis-nova", "manifest.json"),
                "{\"FormatVersion\":1,\"ModelVersion\":1,\"ShardBytes\":33554432,\"CreatedUtc\":\"2026-01-01T00:00:00Z\",\"Sections\":{},\"Generation\":\"GEN-B\"}");
            Assert.False(GenesisShardedCheckpointStore.IsConsistent(starterPointer));
            Assert.False(GenesisShardedCheckpointStore.SeedFromStarter(starterPointer, freshPointer));
            Assert.False(File.Exists(freshPointer));
        }
        finally { try { Directory.Delete(root, true); } catch { } }
    }
}
