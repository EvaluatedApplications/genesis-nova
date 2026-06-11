namespace GenesisNova.Cognition;

public sealed record PlatonicNodeSnapshot(
    string Name,
    double[] PositiveFace,
    double[] NegativeFace,
    int ObservationCount,
    int UseCount = 0,
    int SuccessCount = 0,
    int FailureCount = 0,
    long LastUsedStep = 0);

public sealed record PlatonicRelationSnapshot(
    string Left,
    string Right,
    double ThesisContradiction,
    double LastObservedContradiction,
    double SynthesisContradiction,
    int ObservationCount,
    int UseCount = 0,
    int SuccessCount = 0,
    int FailureCount = 0,
    long LastUsedStep = 0);

public sealed record PlatonicMemorySnapshot(
    int FaceDimension,
    PlatonicNodeSnapshot[] Nodes,
    PlatonicRelationSnapshot[] Relations);
