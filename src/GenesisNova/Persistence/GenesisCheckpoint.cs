using GenesisNova.Core;
using GenesisNova.Cognition;

namespace GenesisNova.Persistence;

public sealed record GenesisCheckpoint(
    GenesisNovaConfig Config,
    string[] Vocabulary,
    MatrixSnapshot Embeddings,
    MatrixSnapshot RouteWeights,
    double[] RouteBias,
    MatrixSnapshot OutputWeights,
    double[] OutputBias,
    PlatonicCognitionSnapshot? Cognition = null,
    GenesisConversationSnapshot? Conversation = null);

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
