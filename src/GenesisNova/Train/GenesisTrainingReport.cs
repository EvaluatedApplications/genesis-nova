namespace GenesisNova.Train;

public sealed record GenesisTrainingReport(
    int Epochs,
    int ExampleCount,
    GenesisStepLoss AverageLoss,
    double ContradictionRate,
    double ConservationDrift,
    double MemoryOverwriteRate,
    int IntrospectionCycles,
    int PendingQueueDepth);
