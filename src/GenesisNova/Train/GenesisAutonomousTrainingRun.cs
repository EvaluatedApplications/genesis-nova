namespace GenesisNova.Train;

public sealed record GenesisAutonomousTrainingRun(
    GenesisAutonomousTrainingRequest Request,
    IReadOnlyList<GenesisAutonomousTrainingRound> Rounds,
    GenesisTrainingReport? FinalReport);
