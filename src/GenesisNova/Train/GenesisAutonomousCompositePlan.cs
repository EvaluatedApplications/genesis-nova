namespace GenesisNova.Train;

public sealed record GenesisAutonomousCompositePlan(
    int Round,
    int Epochs,
    IReadOnlyList<GenesisAutonomousCreatorPlan> CreatorPlans,
    string Reason);
