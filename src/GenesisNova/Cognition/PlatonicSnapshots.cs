namespace GenesisNova.Cognition;

public sealed record PlatonicNodeSnapshot(
    string Name,
    double[] PositiveFace,
    double[] NegativeFace,
    int ObservationCount);

public sealed record PlatonicRelationSnapshot(
    string Left,
    string Right,
    double ThesisContradiction,
    double LastObservedContradiction,
    double SynthesisContradiction,
    int ObservationCount);

public sealed record PlatonicMemorySnapshot(
    int FaceDimension,
    PlatonicNodeSnapshot[] Nodes,
    PlatonicRelationSnapshot[] Relations);

public sealed record NoveltyEventSnapshot(
    string Kind,
    string Input,
    string Output,
    int? RouteId,
    double Confidence,
    DateTime CreatedAtUtc,
    string[] Concepts,
    double NoveltyScore);

public sealed record NoveltyLearnerSnapshot(
    Dictionary<string, double> ConceptWeights);

public sealed record PlatonicQueueSnapshot(
    int Capacity,
    NoveltyEventSnapshot[] Events,
    NoveltyLearnerSnapshot Learner);

public sealed record PlatonicCognitionSnapshot(
    PlatonicMemorySnapshot Memory,
    PlatonicQueueSnapshot Queue);

public sealed record IntrospectionCycleDetail(
    int Cycle,
    string Kind,
    double NoveltyScore,
    int ConceptCount,
    string[] Concepts,
    double BaseContradiction,
    double EnergyBefore,
    double EnergyAfter,
    double Yield,
    int QueueDepthAfter);

public sealed record IntrospectionRunReport(
    int RequestedCycles,
    int Processed,
    int QueueDepthBefore,
    int QueueDepthAfter,
    double AverageYield,
    IntrospectionCycleDetail[] Details);
