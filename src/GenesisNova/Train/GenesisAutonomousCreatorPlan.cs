namespace GenesisNova.Train;

public sealed record GenesisAutonomousCreatorPlan(
    string CreatorName,
    int SampleCount,
    int TrainCount,
    int Difficulty,
    int Epochs,
    double Priority,
    string Reason);
