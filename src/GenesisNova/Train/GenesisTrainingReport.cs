using GenesisNova.Data;

namespace GenesisNova.Train;

public sealed record GenesisExampleProgress(
    string ExampleKey,
    string InputPreview,
    string OutputPreview,
    int SeenCount,
    int SuccessCount,
    double SuccessRate,
    double LastTokenLoss,
    double AverageTokenLoss,
    double BestTokenLoss);

public sealed record GenesisCreatorProgress(
    string CreatorName,
    GenesisTrainingExampleKind TrainingKind,
    int SeenCount,
    int SuccessCount,
    double SuccessRate,
    double LastTokenLoss,
    double AverageTokenLoss,
    double BestTokenLoss);

public sealed record GenesisTrainingReport(
    int Epochs,
    int ExampleCount,
    GenesisStepLoss AverageLoss,
    double ContradictionRate,
    double ConservationDrift,
    double MemoryOverwriteRate,
    int IntrospectionCycles,
    int PendingQueueDepth,
    int SpaceManagementCycles = 0,
    int NodesPruned = 0,
    int RelationsPruned = 0,
    int FinalNodeCount = 0,
    int FinalRelationCount = 0,
    double SpaceNoiseRatio = 0.0,
    int CorrectExampleCount = 0,
    int IncorrectExampleCount = 0,
    double ExampleSuccessRate = 0.0,
    int SkippedCorrectExampleCount = 0,
    int PromptAnswerExampleCount = 0,
    int WindowedTextExampleCount = 0,
    double PromptAnswerAverageTokenLoss = 0.0,
    double WindowedTextAverageTokenLoss = 0.0,
    IReadOnlyList<GenesisExampleProgress>? WeakExamples = null,
    IReadOnlyList<GenesisCreatorProgress>? CreatorProgress = null);
