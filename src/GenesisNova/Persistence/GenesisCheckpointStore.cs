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
        WriteIndented = true
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
            PlatonicSpace: platonicSpace,
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
            Version: GenesisCheckpoint.CurrentVersion);

        var json = JsonSerializer.Serialize(payload, JsonOptions);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);
        ExecuteWithCheckpointLock(path, () =>
        {
            var tempPath = $"{path}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
            try
            {
                File.WriteAllText(tempPath, json);
                File.Move(tempPath, path, overwrite: true);
            }
            finally
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        });
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicMemorySnapshot? PlatonicSpace, GenesisConversationSnapshot? Conversation, GenesisAutonomousTrainingSnapshot? AutonomousTraining) Load(string path)
    {
        var payload = ReadPayload(path);
        var loaded = CreateRuntimePayload(payload, payload.Config);
        return (loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation, loaded.AutonomousTraining);
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicMemorySnapshot? PlatonicSpace, GenesisConversationSnapshot? Conversation, GenesisAutonomousTrainingSnapshot? AutonomousTraining, string? TrainerLearningStateJson) LoadForRuntime(
        string path,
        GenesisNovaConfig runtimeConfig)
    {
        var payload = ReadPayload(path);
        return CreateRuntimePayload(payload, runtimeConfig);
    }

    private static GenesisCheckpoint ReadPayload(string path)
    {
        var json = ExecuteWithCheckpointLock(path, () => File.ReadAllText(path));
        return JsonSerializer.Deserialize<GenesisCheckpoint>(json, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint.");
    }

    public static void CopyCheckpointFile(string sourcePath, string destinationPath)
    {
        ExecuteWithCheckpointLock(sourcePath, () =>
        {
            var dir = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);

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
            GruBhh: payload.GruBhh);

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
