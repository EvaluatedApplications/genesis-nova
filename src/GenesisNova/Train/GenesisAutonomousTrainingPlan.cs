namespace GenesisNova.Train;

public sealed record GenesisAutonomousTrainingPlan(
    string CreatorName,
    int SampleCount,
    int TrainCount,
    int Difficulty,
    int Epochs,
    string Reason);
