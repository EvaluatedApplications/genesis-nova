using System.Text.Json;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;

namespace GenesisNova.Persistence;

public static class GenesisCheckpointStore
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public static void Save(
        string path,
        GenesisNovaConfig config,
        WhitespaceGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicCognitionSnapshot? cognition = null,
        GenesisConversationSnapshot? conversation = null)
    {
        var snapshot = model.Export();
        var payload = new GenesisCheckpoint(
            Config: config,
            Vocabulary: tokenizer.Vocabulary.ToArray(),
            Embeddings: MatrixSnapshot.From(snapshot.Embeddings),
            RouteWeights: new MatrixSnapshot(0, 0, []),
            RouteBias: [],
            OutputWeights: MatrixSnapshot.From(snapshot.OutputWeights),
            OutputBias: snapshot.OutputBias,
            Cognition: cognition,
            Conversation: conversation);

        var json = JsonSerializer.Serialize(payload, JsonOptions);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);
        File.WriteAllText(path, json);
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicCognitionSnapshot? Cognition, GenesisConversationSnapshot? Conversation) Load(string path)
    {
        var payload = ReadPayload(path);
        return CreateRuntimePayload(payload, payload.Config);
    }

    public static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicCognitionSnapshot? Cognition, GenesisConversationSnapshot? Conversation) LoadForRuntime(
        string path,
        GenesisNovaConfig runtimeConfig)
    {
        var payload = ReadPayload(path);
        return CreateRuntimePayload(payload, runtimeConfig);
    }

    private static GenesisCheckpoint ReadPayload(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<GenesisCheckpoint>(json, JsonOptions)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint.");
    }

    private static (GenesisNovaConfig Config, WhitespaceGenesisTokenizer Tokenizer, GenesisNeuralModel Model, PlatonicCognitionSnapshot? Cognition, GenesisConversationSnapshot? Conversation) CreateRuntimePayload(
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
        var snapshot = new ModelSnapshot(
            payload.Embeddings.ToMatrix(),
            payload.OutputWeights.ToMatrix(),
            payload.OutputBias);

        if (effectiveConfig.HiddenSize > payload.Config.HiddenSize)
            snapshot = ExpandSnapshot(snapshot, effectiveConfig.HiddenSize);

        model.Import(snapshot);

        return (effectiveConfig, tokenizer, model, payload.Cognition, payload.Conversation);
    }

    private static ModelSnapshot ExpandSnapshot(ModelSnapshot snapshot, int hiddenSize)
    {
        if (snapshot.Embeddings.GetLength(1) >= hiddenSize)
            return snapshot;

        var vocab = snapshot.Embeddings.GetLength(0);
        var expandedEmb = ExpandColumns(snapshot.Embeddings, hiddenSize, fill: 0.0);
        var expandedOut = ExpandRows(snapshot.OutputWeights, hiddenSize, fill: 0.0);
        return new ModelSnapshot(expandedEmb, expandedOut, snapshot.OutputBias.ToArray());
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
