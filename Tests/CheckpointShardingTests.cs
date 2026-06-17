using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Binary SHARDED checkpoint storage (MODEL_STORAGE.md): exact all-double round-trip, shards bounded to the
/// 32&#160;MiB target, and one-time migration from the legacy single-JSON checkpoint with cleanup of the old
/// files. Fast (tiny model).
/// </summary>
public sealed class CheckpointShardingTests
{
    private static (GenesisNovaConfig, WhitespaceGenesisTokenizer, GenesisNeuralModel, PlatonicSpaceMemory) TrainTiny(string dir)
    {
        var config = new GenesisNovaConfig(HiddenSize: 32, Backend: ComputeBackend.Cpu, LocalStateDirectory: dir);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 5);
        var trainer = new GenesisTrainer(tok, model, memory, config);
        var inference = new GenesisInferenceEngine(tok, model, memory, null);
        trainer.SetInferencePolicy(inference);
        foreach (var (a, b) in new[] { ("apple", "fruit"), ("sparrow", "bird"), ("copper", "metal") })
            for (var i = 0; i < 5; i++) trainer.TrainStep(new GenesisExample(a, b));
        for (var i = 0; i < 5; i++) trainer.TrainStep(new GenesisExample("2 + 3", "5")); // exercises the query + plan heads
        return (config, tok, model, memory);
    }

    [Fact]
    public void ShardedCheckpoint_RoundTrips_AllDouble_AndShardsAreBounded()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gnv-shard-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var (config, tok, model, memory) = TrainTiny(dir);
            var path = Path.Combine(dir, "genesis-nova.autosave.checkpoint.json");

            GenesisCheckpointStore.Save(path, config, tok, model, platonicSpace: memory.ExportSnapshot());

            // Pointer marker at the legacy path; real data in the sharded model dir; every shard within budget.
            Assert.True(File.Exists(path));
            Assert.Contains("GnvSharded", File.ReadAllText(path));
            var modelDir = GenesisShardedCheckpointStore.ModelDir(path);
            Assert.True(GenesisShardedCheckpointStore.ModelExists(modelDir));
            foreach (var f in Directory.EnumerateFiles(Path.Combine(modelDir, "shards"), "*.gnv"))
                Assert.True(new FileInfo(f).Length <= GenesisShardedCheckpointStore.TargetShardBytes);

            // Exact round-trip of the PERSISTED weights — bit-faithful (all-double), shapes preserved, vocab and
            // substrate survive. (Query/plan heads aren't part of the checkpoint format and lazily re-init on use,
            // so total ParameterCount intentionally isn't asserted here.)
            var orig = model.Export();
            var loaded = GenesisCheckpointStore.Load(path);
            var back = loaded.Model.Export();
            Assert.Equal(orig.Embeddings.GetLength(0), back.Embeddings.GetLength(0));
            Assert.Equal(orig.Embeddings.GetLength(1), back.Embeddings.GetLength(1));
            Assert.Equal(orig.Embeddings[0, 0], back.Embeddings[0, 0], 12);
            Assert.Equal(orig.Embeddings[orig.Embeddings.GetLength(0) - 1, orig.Embeddings.GetLength(1) - 1],
                         back.Embeddings[back.Embeddings.GetLength(0) - 1, back.Embeddings.GetLength(1) - 1], 12);
            Assert.Equal(orig.OutputWeights[0, 0], back.OutputWeights[0, 0], 12);
            Assert.NotNull(orig.GruWih);
            Assert.Equal(orig.GruWih!.GetLength(0), back.GruWih!.GetLength(0));
            Assert.Equal(orig.GruWih[0, 0], back.GruWih[0, 0], 12);
            Assert.Equal(tok.Vocabulary.Count, loaded.Tokenizer.Vocabulary.Count);
            Assert.NotNull(loaded.PlatonicSpace);

            // Query + plan heads (created by the arithmetic training) now SURVIVE the round-trip — they're no
            // longer dropped/re-init untrained on load.
            Assert.NotNull(orig.QueryOpWeights);
            Assert.NotNull(back.QueryOpWeights);
            Assert.Equal(orig.QueryOpWeights![0, 0], back.QueryOpWeights![0, 0], 12);
            Assert.NotNull(orig.PlanWeights);
            Assert.NotNull(back.PlanWeights);
            Assert.Equal(orig.PlanWeights![0, 0], back.PlanWeights![0, 0], 12);
            Assert.Equal(orig.QueryOperandWeights![0, 0], back.QueryOperandWeights![0, 0], 12);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void LegacyJsonCheckpoint_MigratesToSharded_AndCleansUpOldFiles()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gnv-mig-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var (config, tok, model, _) = TrainTiny(dir);
            var path = Path.Combine(dir, "genesis-nova.autosave.checkpoint.json");

            // Hand-write a LEGACY single-JSON checkpoint (the pre-sharding format).
            var s = model.Export();
            var legacy = new GenesisCheckpoint(
                Config: config,
                Vocabulary: tok.Vocabulary.ToArray(),
                Embeddings: MatrixSnapshot.From(s.Embeddings),
                OutputWeights: MatrixSnapshot.From(s.OutputWeights),
                OutputBias: s.OutputBias,
                RouteWeights: s.RouteWeights is not null ? MatrixSnapshot.From(s.RouteWeights) : null,
                RouteBias: s.RouteBias,
                EditWeights: s.EditWeights is not null ? MatrixSnapshot.From(s.EditWeights) : null,
                EditBias: s.EditBias,
                GruWih: s.GruWih is not null ? MatrixSnapshot.From(s.GruWih) : null,
                GruWhh: s.GruWhh is not null ? MatrixSnapshot.From(s.GruWhh) : null,
                GruBih: s.GruBih,
                GruBhh: s.GruBhh,
                SpacingModel: tok.SpacingModel.Export(),
                CasingModel: tok.CasingModel.Export(),
                Version: GenesisCheckpoint.CurrentVersion);
            File.WriteAllText(path, JsonSerializer.Serialize(legacy));

            // Load triggers migration.
            var loaded = GenesisCheckpointStore.Load(path);

            Assert.True(GenesisShardedCheckpointStore.ModelExists(GenesisShardedCheckpointStore.ModelDir(path)));            // migrated
            Assert.Contains("GnvSharded", File.ReadAllText(path));                                                          // pointer left behind
            Assert.True(File.Exists(Path.Combine(dir, ".legacy-backup", "genesis-nova.autosave.checkpoint.json")));        // old file moved out
            Assert.Equal(s.Embeddings[0, 0], loaded.Model.Export().Embeddings[0, 0], 12);                                   // data intact
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
