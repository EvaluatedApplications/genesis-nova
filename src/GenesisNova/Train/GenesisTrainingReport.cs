namespace GenesisNova.Train;

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
    double SpaceNoiseRatio = 0.0);
