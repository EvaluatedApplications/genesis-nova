namespace GenesisNova.Train;

public sealed record GenesisAutonomousTrainingRound(
    int Round,
    string CreatorName,
    int SampleCount,
    int Difficulty,
    int Epochs,
    GenesisTrainingReport Report);
