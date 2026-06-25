using System.Text.Json;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;

namespace GenesisNova.Persistence;

public static class GenesisCheckpointStore
{
    private static readonly Mutex CheckpointFileMutex = new(initiallyOwned: false, name: @"Local\GenesisNova.CheckpointFile");
    private static readonly TimeSpan CheckpointLockTimeout = TimeSpan.FromSeconds(15);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        // Compact (not indented): the checkpoint is large (100s of MB at scale); indentation is pure bloat that
        // slows serialize/write/parse. JSON parses fine either way, so old indented checkpoints still load.
        WriteIndented = false
    };

    public static void Save(
        string path,
        GenesisNovaConfig config,
        WhitespaceGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicMemorySnapshot? platonicSpace = null,
        GenesisConversationSnapshot? conversation = null,
        GenesisAutonomousTrainingSnapshot? autonomousTraining = null,
        string? trainerLearningStateJson = null)
    {
        var snapshot = model.Export();
        var payload = new GenesisCheckpoint(
            Config: config,
            Vocabulary: tokenizer.Vocabulary.ToArray(),
            Embeddings: MatrixSnapshot.From(snapshot.Embeddings),
            OutputWeights: MatrixSnapshot.From(snapshot.OutputWeights),
            OutputBias: snapshot.OutputBias,
            // The platonic space is persisted to a SEPARATE companion file (see below), not embedded in the NN
            // checkpoint — so the substrate can be wiped (delete the companion) while the long-lived NN persists.
            PlatonicSpace: null,
            Conversation: conversation,
            AutonomousTraining: autonomousTraining,
            RouteWeights: snapshot.RouteWeights is not null ? MatrixSnapshot.From(snapshot.RouteWeights) : null,
            RouteBias: snapshot.RouteBias,
            TrainerLearningStateJson: trainerLearningStateJson,
            // Persist the edit head + shared GRU so a loaded model is the SAME trained model. Without
            // these the GRU (read by every head) reinitialised untrained on load.
            EditWeights: snapshot.EditWeights is not null ? MatrixSnapshot.From(snapshot.EditWeights) : null,
            EditBias: snapshot.EditBias,
            GruWih: snapshot.GruWih is not null ? MatrixSnapshot.From(snapshot.GruWih) : null,
            GruWhh: snapshot.GruWhh is not null ? MatrixSnapshot.From(snapshot.GruWhh) : null,
            GruBih: snapshot.GruBih,
            GruBhh: snapshot.GruBhh,
            SpacingModel: tokenizer.SpacingModel.Export(),
            CasingModel: tokenizer.CasingModel.Export(),
            // Query-construction + plan heads — persist so they aren't reset to untrained on every load.
            QueryOpWeights: snapshot.QueryOpWeights is not null ? MatrixSnapshot.From(snapshot.QueryOpWeights) : null,
            QueryOpBias: snapshot.QueryOpBias,
            QueryOperandWeights: snapshot.QueryOperandWeights is not null ? MatrixSnapshot.From(snapshot.QueryOperandWeights) : null,
            QueryOperandBias: snapshot.QueryOperandBias,
            PlanWeights: snapshot.PlanWeights is not null ? MatrixSnapshot.From(snapshot.PlanWeights) : null,
            PlanBias: snapshot.PlanBias,
            // Persist the shared reasoning trunk so a loaded model routes identically (not random trunk × head).
            TrunkWeights: snapshot.TrunkWeights is not null ? MatrixSnapshot.From(snapshot.TrunkWeights) : null,
            TrunkBias: snapshot.TrunkBias,
            Version: GenesisCheckpoint.CurrentVersion);

        // BINARY SHARDED storage (see MODEL_STORAGE.md): the NN goes to a sharded model directory, the substrate
        // to a separate sharded directory (still resettable), and the legacy ".json" path holds a tiny POINTER
        // marker so existence checks + the daemon's resume resolve keep working unchanged.
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);
        ExecuteWithCheckpointLock(path, () =>
        {
            // One GENERATION per save, stamped on the model + substrate manifests and the pointer (written LAST as
            // the commit). A crash between the writes leaves them at different generations → IsConsistent() detects
            // the tear on load and falls back to last-good, so we never resume a model-vs-substrate-mismatched brain.
            var generation = Guid.NewGuid().ToString("N");
            GenesisShardedCheckpointStore.WriteModel(GenesisShardedCheckpointStore.ModelDir(path), payload, generation);
            if (platonicSpace is not null)
                GenesisShardedCheckpointStore.WriteSubstrate(GenesisShardedCheckpointStore.SubstrateDir(path), platonicSpace, generation);
            WriteAtomic(path, GenesisShardedCheckpointStore.PointerJson(generation)); // commit
            // A stale legacy substrate companion (data now lives in the .platonic dir) would only confuse — drop it.
            var companion = PlatonicCompanionPath(path);
            if (File.Exists(companion)) try { File.Delete(companion); } catch { }
        });
    }

    /// <summary>The companion file holding the platonic space for a given NN checkpoint path. Deleting it
    /// resets the substrate while keeping the long-lived NN.</summary>
    public static string PlatonicCompanionPath(string checkpointPath) => checkpointPath + ".platonic.json";

    private static void WriteAtomic(string path, string content)
    {
        var tempPath = $"{path}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(tempPath, content);
            File.Move(tempPath, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicMemorySnapshot? PlatonicSpace, GenesisConversationSnapshot? Conversation, GenesisAutonomousTrainingSnapshot? AutonomousTraining) Load(string path)
    {
        var (payload, platonic) = ResolveStored(path);
        var loaded = CreateRuntimePayload(payload, payload.Config);
        return (loaded.Config, loaded.Tokenizer, loaded.Model, platonic, loaded.Conversation, loaded.AutonomousTraining);
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicMemorySnapshot? PlatonicSpace, GenesisConversationSnapshot? Conversation, GenesisAutonomousTrainingSnapshot? AutonomousTraining, string? TrainerLearningStateJson) LoadForRuntime(
        string path,
        GenesisNovaConfig runtimeConfig)
    {
        var loadPath = ResolveConsistentPath(path, runtimeConfig);
        var (payload, platonic) = ResolveStored(loadPath);
        var loaded = CreateRuntimePayload(payload, runtimeConfig);
        return (loaded.Config, loaded.Tokenizer, loaded.Model, platonic, loaded.Conversation, loaded.AutonomousTraining, loaded.TrainerLearningStateJson);
    }

    /// <summary>Guard against a TORN save (a crash between writing the model dir, the substrate dir, and the pointer):
    /// if the checkpoint's generations don't all match, resume from the last-good (probe-passing, consistent)
    /// checkpoint instead of a model-vs-substrate-mismatched brain. Falls through to the original path when it is
    /// consistent, legacy (unverifiable), or there is no consistent last-good (graceful: a one-generation-stale
    /// substrate beats refusing to resume a long run).</summary>
    private static string ResolveConsistentPath(string path, GenesisNovaConfig runtimeConfig)
    {
        if (!GenesisShardedCheckpointStore.ModelExists(GenesisShardedCheckpointStore.ModelDir(path))
            || GenesisShardedCheckpointStore.IsConsistent(path))
            return path;

        var lastGood = GenesisLocalStateStore.ResolveLastGoodCheckpointPath(runtimeConfig);
        if (!string.Equals(lastGood, path, StringComparison.OrdinalIgnoreCase)
            && GenesisShardedCheckpointStore.ModelExists(GenesisShardedCheckpointStore.ModelDir(lastGood))
            && GenesisShardedCheckpointStore.IsConsistent(lastGood))
        {
            try { GenesisLocalStateStore.AppendJournalEntry(runtimeConfig, "torn-checkpoint-fallback", detail: $"{path} -> {lastGood}"); } catch { }
            return lastGood;
        }
        try { GenesisLocalStateStore.AppendJournalEntry(runtimeConfig, "torn-checkpoint-no-fallback", detail: path); } catch { }
        return path;
    }

    /// <summary>Resolve a checkpoint at <paramref name="path"/>: read the sharded model if present, otherwise
    /// load a LEGACY single-JSON checkpoint and migrate it to the sharded layout (cleaning up the old files).</summary>
    private static (GenesisCheckpoint Payload, PlatonicMemorySnapshot? Platonic) ResolveStored(string path)
    {
        var modelDir = GenesisShardedCheckpointStore.ModelDir(path);
        if (GenesisShardedCheckpointStore.ModelExists(modelDir))
        {
            var sharded = GenesisShardedCheckpointStore.ReadModel(modelDir);
            var substrate = GenesisShardedCheckpointStore.ReadSubstrate(GenesisShardedCheckpointStore.SubstrateDir(path));
            return (sharded, substrate);
        }

        // LEGACY: a pre-sharding single-JSON checkpoint (+ optional .platonic.json companion). Load it, migrate to
        // the sharded layout, then move the originals out of the active directory. DELETE this branch once
        // migration is proven in the wild (it is the only reason the JSON reader below still exists).
        var legacy = ReadPayload(path);
        var legacyPlatonic = ReadPlatonicCompanion(path) ?? legacy.PlatonicSpace;
        MigrateLegacyToSharded(path, legacy, legacyPlatonic);
        return (legacy, legacyPlatonic);
    }

    private static void MigrateLegacyToSharded(string path, GenesisCheckpoint payload, PlatonicMemorySnapshot? platonic)
    {
        ExecuteWithCheckpointLock(path, () =>
        {
            var generation = Guid.NewGuid().ToString("N");
            GenesisShardedCheckpointStore.WriteModel(GenesisShardedCheckpointStore.ModelDir(path), payload with { PlatonicSpace = null }, generation);
            if (platonic is not null)
                GenesisShardedCheckpointStore.WriteSubstrate(GenesisShardedCheckpointStore.SubstrateDir(path), platonic, generation);

            // Clean up: move the legacy files into .legacy-backup so the active dir only holds what's relevant
            // (no confusion), then leave the pointer marker at the original path.
            var stateDir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(stateDir))
            {
                var backup = Path.Combine(stateDir, ".legacy-backup");
                Directory.CreateDirectory(backup);
                MoveToBackup(path, backup);
                MoveToBackup(PlatonicCompanionPath(path), backup);
            }
            WriteAtomic(path, GenesisShardedCheckpointStore.PointerJson(generation));
        });
    }

    private static void MoveToBackup(string file, string backupDir)
    {
        if (!File.Exists(file)) return;
        try { File.Move(file, Path.Combine(backupDir, Path.GetFileName(file)), overwrite: true); } catch { /* best-effort */ }
    }

    private static GenesisCheckpoint ReadPayload(string path)
    {
        var json = ExecuteWithCheckpointLock(path, () => File.ReadAllText(path));
        return JsonSerializer.Deserialize<GenesisCheckpoint>(json, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint.");
    }

    // Read the companion platonic-space file if present (null when it's been deleted → substrate resets fresh
    // while the NN checkpoint is unchanged). General mechanism; the consumer chooses when to delete it.
    private static PlatonicMemorySnapshot? ReadPlatonicCompanion(string path)
    {
        var companion = PlatonicCompanionPath(path);
        if (!File.Exists(companion)) return null;
        try
        {
            var json = ExecuteWithCheckpointLock(companion, () => File.ReadAllText(companion));
            return JsonSerializer.Deserialize<PlatonicMemorySnapshot>(json, JsonOptions);
        }
        catch
        {
            return null;
        }
    }

    public static void CopyCheckpointFile(string sourcePath, string destinationPath)
    {
        ExecuteWithCheckpointLock(sourcePath, () =>
        {
            var dir = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);

            // Sharded checkpoint: copy the model dir (+ substrate dir) and the pointer marker.
            if (GenesisShardedCheckpointStore.ModelExists(GenesisShardedCheckpointStore.ModelDir(sourcePath)))
            {
                GenesisShardedCheckpointStore.CopyModel(
                    GenesisShardedCheckpointStore.ModelDir(sourcePath),
                    GenesisShardedCheckpointStore.ModelDir(destinationPath));
                if (File.Exists(sourcePath)) File.Copy(sourcePath, destinationPath, overwrite: true);
                return;
            }

            // Legacy single-file copy.
            var tempPath = $"{destinationPath}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
            try
            {
                File.Copy(sourcePath, tempPath, overwrite: true);
                File.Move(tempPath, destinationPath, overwrite: true);
            }
            finally
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        });
    }

    private static void ExecuteWithCheckpointLock(string path, Action action)
    {
        _ = ExecuteWithCheckpointLock(path, () =>
        {
            action();
            return true;
        });
    }

    private static T ExecuteWithCheckpointLock<T>(string path, Func<T> action)
    {
        if (!CheckpointFileMutex.WaitOne(CheckpointLockTimeout))
            throw new IOException($"Timed out waiting for checkpoint file lock: {path}");

        try
        {
            return action();
        }
        finally
        {
            CheckpointFileMutex.ReleaseMutex();
        }
    }

    private static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicMemorySnapshot? PlatonicSpace, GenesisConversationSnapshot? Conversation, GenesisAutonomousTrainingSnapshot? AutonomousTraining, string? TrainerLearningStateJson) CreateRuntimePayload(
        GenesisCheckpoint payload,
        GenesisNovaConfig runtimeConfig)
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        tokenizer.ReplaceVocabulary(payload.Vocabulary);
        tokenizer.SpacingModel.Import(payload.SpacingModel);
        tokenizer.CasingModel.Import(payload.CasingModel);

        var effectiveConfig = payload.Config with
        {
            Backend = runtimeConfig.Backend,
            EnableParallelMath = runtimeConfig.EnableParallelMath,
            MaxDegreeOfParallelism = runtimeConfig.MaxDegreeOfParallelism,
            Deterministic = runtimeConfig.Deterministic,
            AutoPersist = runtimeConfig.AutoPersist,
            AutoResume = runtimeConfig.AutoResume,
            AutoScaleVram = runtimeConfig.AutoScaleVram,
            TargetVramUtilization = runtimeConfig.TargetVramUtilization,
            ReserveVramMb = runtimeConfig.ReserveVramMb,
            LocalStateDirectory = runtimeConfig.LocalStateDirectory,
            HiddenSize = Math.Max(payload.Config.HiddenSize, runtimeConfig.HiddenSize)
        };

        var model = new GenesisNeuralModel(effectiveConfig);
        var routeWeights = IsUsableRouteSnapshot(payload.RouteWeights, payload.RouteBias, payload.Embeddings.Cols)
            ? payload.RouteWeights!.ToMatrix()
            : null;
        var routeBias = routeWeights is not null ? payload.RouteBias!.ToArray() : null;
        var snapshot = new ModelSnapshot(
            payload.Embeddings.ToMatrix(),
            payload.OutputWeights.ToMatrix(),
            payload.OutputBias,
            routeWeights,
            routeBias,
            // Restore the edit head + shared GRU. Import validates shapes (HasUsableEditHead /
            // HasUsableGru) and reinitialises gracefully if a pre-GRU checkpoint omitted them.
            EditWeights: payload.EditWeights?.ToMatrix(),
            EditBias: payload.EditBias,
            GruWih: payload.GruWih?.ToMatrix(),
            GruWhh: payload.GruWhh?.ToMatrix(),
            GruBih: payload.GruBih,
            GruBhh: payload.GruBhh,
            QueryOpWeights: payload.QueryOpWeights?.ToMatrix(),
            QueryOpBias: payload.QueryOpBias,
            QueryOperandWeights: payload.QueryOperandWeights?.ToMatrix(),
            QueryOperandBias: payload.QueryOperandBias,
            PlanWeights: payload.PlanWeights?.ToMatrix(),
            PlanBias: payload.PlanBias,
            TrunkWeights: payload.TrunkWeights?.ToMatrix(),
            TrunkBias: payload.TrunkBias);

        // Hidden-size growth can't reshape the GRU/edit head — Import rejects the mismatch and
        // reinitialises them; the embeddings/output/route heads still expand.
        if (effectiveConfig.HiddenSize > payload.Config.HiddenSize)
            snapshot = ExpandSnapshot(snapshot, effectiveConfig.HiddenSize);

        model.Import(snapshot);

        return (effectiveConfig, tokenizer, model, payload.PlatonicSpace, payload.Conversation, payload.AutonomousTraining, payload.TrainerLearningStateJson);
    }

    private static ModelSnapshot ExpandSnapshot(ModelSnapshot snapshot, int hiddenSize)
    {
        if (snapshot.Embeddings.GetLength(1) >= hiddenSize)
            return snapshot;

        var expandedEmb = ExpandColumns(snapshot.Embeddings, hiddenSize, fill: 0.0);
        var expandedOut = ExpandRows(snapshot.OutputWeights, hiddenSize, fill: 0.0);
        var expandedRoute = snapshot.RouteWeights is not null
            ? ExpandRows(snapshot.RouteWeights, hiddenSize, fill: 0.0)
            : null;
        return new ModelSnapshot(
            expandedEmb,
            expandedOut,
            snapshot.OutputBias.ToArray(),
            expandedRoute,
            snapshot.RouteBias?.ToArray());
    }

    private static bool IsUsableRouteSnapshot(MatrixSnapshot? routeWeights, double[]? routeBias, int hiddenSize)
    {
        if (routeWeights is null || routeBias is null)
            return false;

        return routeWeights.Rows == hiddenSize
            && routeWeights.Rows > 0
            && routeWeights.Cols > 0
            && routeWeights.Values.Length == routeWeights.Rows * routeWeights.Cols
            && routeBias.Length == routeWeights.Cols;
    }

    private static double[,] ExpandColumns(double[,] matrix, int targetCols, double fill)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var expanded = new double[rows, targetCols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
                expanded[r, c] = matrix[r, c];
            for (var c = cols; c < targetCols; c++)
                expanded[r, c] = fill;
        }
        return expanded;
    }

    private static double[,] ExpandRows(double[,] matrix, int targetRows, double fill)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var expanded = new double[targetRows, cols];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                expanded[r, c] = matrix[r, c];

        for (var r = rows; r < targetRows; r++)
            for (var c = 0; c < cols; c++)
                expanded[r, c] = fill;

        return expanded;
    }
}
