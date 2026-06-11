using GenesisNova.Core;
using GenesisNova.Cognition;
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
    int Version = 0)
{
    public const int CurrentVersion = 2;
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
