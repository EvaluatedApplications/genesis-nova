namespace GenesisNova.Train;

public sealed record GenesisTrainerLearningState(
    IReadOnlyList<SpaceDecisionJournalEntry>? SpaceDecisionJournalEntries,
    IReadOnlyList<MasteredRehearsalExample>? MasteredRehearsalRing,
    Dictionary<int, int>? SpacePolicyActionCounters,
    IReadOnlyList<SpacePolicyTransitionSnapshot>? SpacePolicyTrajectory,
    Dictionary<string, int>? ConceptCoverageCounts,
    GenesisTrainerTelemetryState? Telemetry,
    int SpacePolicyStepCounter,
    int PriorSpaceActionId,
    IReadOnlyList<ConceptPlanDecisionJournalEntry>? ConceptPlanDecisionJournalEntries);

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
