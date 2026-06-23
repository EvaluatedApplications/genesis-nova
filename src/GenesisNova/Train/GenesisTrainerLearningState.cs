using GenesisNova.Core;

namespace GenesisNova.Train;

public sealed record GenesisTrainerLearningState(
    IReadOnlyList<SpaceDecisionJournalEntry>? SpaceDecisionJournalEntries,
    IReadOnlyList<MasteredRehearsalExample>? MasteredRehearsalRing,
    IReadOnlyList<SpacePolicyTransitionSnapshot>? SpacePolicyTrajectory,
    Dictionary<string, int>? ConceptCoverageCounts,
    GenesisTrainerTelemetryState? Telemetry,
    int SpacePolicyStepCounter,
    IReadOnlyList<ConceptPlanDecisionJournalEntry>? ConceptPlanDecisionJournalEntries,
    // Appended (optional) checkpoint-persistence fields. Old JSON without these deserializes to null,
    // and existing positional construction remains valid by passing these last by name.
    TransformAccumulatorSnapshot? TransformAccumulator = null,
    FoldPathDiscoverySnapshot? FoldPaths = null);

public sealed record MasteredRehearsalExample(
    string Input,
    string Output,
    int? RouteLabel);

public sealed record SpacePolicyTransitionSnapshot(
    string StateEncoding,
    int ActionId,
    double NoiseRatio,
    double AverageBridgeConfidence,
    double RelationPressure,
    int StepIndex);

public sealed record ConceptPlanDecisionJournalEntry(
    int Step,
    string Prompt,
    string Target,
    IReadOnlyList<string> SelectedLatentConcepts);

public sealed record GenesisTrainerTelemetryState(
    int BiasAppliedCount,
    int BiasAppliedCorrectCount,
    int BiasNotAppliedCount,
    int BiasNotAppliedCorrectCount,
    int PlatonicConfidenceCorrectCount,
    int PlatonicConfidenceIncorrectCount,
    double PlatonicConfidenceCorrectSum,
    double PlatonicConfidenceIncorrectSum,
    int FallbackCount,
    int FallbackCorrectCount,
    int TelemetryObservationCount,
    int SpaceToolParseFailureCount,
    int ConceptPlanCalls,
    int ConceptPlanDirectCount,
    int ConceptPlanMergedCount,
    int ConceptPlanFallbackCount,
    double ConceptPlanCoverageSum,
    double ConceptPlanNoveltySum,
    int MasteredSkippedCount,
    int MasteredRehearsalCount,
    int MasteredInterleavedRehearsalCount,
    int SpacePolicyRetrospectiveCreditCount,
    int ConceptPlanRetrospectiveCreditCount);
