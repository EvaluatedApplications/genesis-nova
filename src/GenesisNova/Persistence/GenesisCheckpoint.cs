using GenesisNova.Core;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Persistence;

public sealed record GenesisCheckpoint(
    GenesisNovaConfig Config,
    string[] Vocabulary,
    MatrixSnapshot Embeddings,
    MatrixSnapshot OutputWeights,
    double[] OutputBias,
    PlatonicMemorySnapshot? PlatonicSpace = null,
    GenesisConversationSnapshot? Conversation = null,
    GenesisAutonomousTrainingSnapshot? AutonomousTraining = null,
    MatrixSnapshot? RouteWeights = null,
    double[]? RouteBias = null,
    string? TrainerLearningStateJson = null,
    // Edit-head + shared GRU gate weights. Optional/null on pre-GRU checkpoints (the model
    // reinitialises them on load). Without these, the trained GRU — which EVERY head reads via
    // hInput — was silently dropped on save, so a loaded model behaved like an untrained one.
    MatrixSnapshot? EditWeights = null,
    double[]? EditBias = null,
    MatrixSnapshot? GruWih = null,
    MatrixSnapshot? GruWhh = null,
    double[]? GruBih = null,
    double[]? GruBhh = null,
    // Learned detokenization spacing statistics. Null on older checkpoints (the model relearns from training
    // text on load). Lets the spacing model survive a reload instead of resetting to the heuristic prior.
    SpacingModelSnapshot? SpacingModel = null,
    // Learned detokenization casing statistics (folded token -> surface spelling). Null on older checkpoints
    // (relearns on load). Lets restored casing ("WhitespaceGenesisTokenizer") survive a reload.
    CasingModelSnapshot? CasingModel = null,
    // Platonic-query construction heads (op classifier + operand scorer) and the composer PLAN head. Null on
    // older checkpoints (then lazily reinitialised on load). Persisted so these TRAINED heads survive a reload
    // instead of resetting each time — the drop-on-load gap previously fixed for the GRU/edit heads.
    MatrixSnapshot? QueryOpWeights = null,
    double[]? QueryOpBias = null,
    MatrixSnapshot? QueryOperandWeights = null,
    double[]? QueryOperandBias = null,
    MatrixSnapshot? PlanWeights = null,
    double[]? PlanBias = null,
    int Version = 0)
{
    public const int CurrentVersion = 4;
}

public sealed record MatrixSnapshot(int Rows, int Cols, double[] Values)
{
    public static MatrixSnapshot From(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var values = new double[rows * cols];
        var k = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                values[k++] = matrix[i, j];
        return new MatrixSnapshot(rows, cols, values);
    }

    public double[,] ToMatrix()
    {
        var matrix = new double[Rows, Cols];
        var k = 0;
        for (var i = 0; i < Rows; i++)
            for (var j = 0; j < Cols; j++)
                matrix[i, j] = Values[k++];
        return matrix;
    }
}

public sealed record GenesisConversationSnapshot(
    string Summary,
    ConversationTurnSnapshot[] RecentTurns,
    int ResetCount,
    int CompactionCount,
    double BranchTrust,
    DateTimeOffset? LastResetAtUtc = null);

public sealed record ConversationTurnSnapshot(
    DateTimeOffset TimestampUtc,
    string Role,
    string Content,
    bool IsResetSignal,
    string? Note = null);

public sealed record GenesisAutonomousTrainingSnapshot(
    GenesisAutonomousTrainingRound[] History);
