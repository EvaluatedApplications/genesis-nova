using GenesisNova.Cognition;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed record GenesisTrainTaskData(
    string FilePath,
    int Epochs,
    int? IntrospectionCyclesPerEpoch = null,
    string? SavePath = null,
    string? LogPath = null,
    IReadOnlyList<GenesisExample>? Examples = null,
    GenesisTrainingReport? Report = null);

public sealed record GenesisTrainOneTaskData(
    GenesisExample Example,
    GenesisStepLoss? Loss = null,
    int QueueDepth = 0);

public sealed record GenesisPredictTaskData(
    string Input,
    int MaxNewTokens = 48,
    bool EnableIntrospection = true,
    GenerationResult? Result = null,
    int IntrospectionProcessed = 0,
    int QueueDepth = 0);

public sealed record GenesisIntrospectTaskData(
    int Cycles,
    int Processed = 0,
    int QueueDepth = 0);

public sealed record GenesisRelateTaskData(
    string Left,
    string Right,
    double Contradiction,
    int QueueDepth = 0);

public sealed record GenesisConceptTaskData(
    string Concept,
    string Description = "");

public sealed record GenesisSaveTaskData(string Path, bool Saved = false);
public sealed record GenesisLoadTaskData(string Path, bool Loaded = false, PlatonicCognitionSnapshot? Cognition = null);
public sealed record GenesisConversationTaskData(
    string UserInput,
    string AssistantOutput,
    bool ResetSignal = false,
    string? Note = null,
    string? ContextBrief = null,
    int RecentTurnCount = 0);

public sealed record GenesisCompactConversationTaskData(
    string? Note = null,
    bool Compacted = false,
    string? ContextBrief = null,
    int RecentTurnCount = 0);

public sealed record GenesisEvaluationReport(
    int SampleCount,
    int ExactMatchCount,
    int RouteLabeledCount,
    int RouteCorrectCount,
    double ExactMatchAccuracy,
    double RouteAccuracy);
