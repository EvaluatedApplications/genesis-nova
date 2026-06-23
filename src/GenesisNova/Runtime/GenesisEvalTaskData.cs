using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Train;
using System.Collections.Immutable;

namespace GenesisNova.Runtime;

public sealed record GenesisTrainTaskData(
    string FilePath,
    int Epochs,
    string? SavePath = null,
    string? LogPath = null,
    Action<string>? UiLogger = null,
    IReadOnlyList<GenesisExample>? Examples = null,
    GenesisTrainingReport? Report = null);

public sealed record GenesisTrainOneTaskData(
    GenesisExample Example,
    GenesisStepLoss? Loss = null);

public sealed record GenesisPredictTaskData(
    string Input,
    int MaxNewTokens = 48,
    GenerationResult? Result = null);

public sealed record GenesisSaveTaskData(string Path, bool Saved = false);
public sealed record GenesisLoadTaskData(string Path, bool Loaded = false);
public sealed record GenesisEvaluationReport(
    int SampleCount,
    int ExactMatchCount,
    int RouteLabeledCount,
    int RouteCorrectCount,
    double ExactMatchAccuracy,
    double RouteAccuracy);

public sealed record GenesisAutonomousTrainTaskData(
    GenesisAutonomousTrainingRequest Request,
    IReadOnlyList<GenesisAutonomousTrainingRound> History,
    int RoundIndex,
    CancellationToken CancellationToken = default,
    Action<string>? UiLogger = null,
    GenesisAutonomousCompositePlan? Plan = null,
    IReadOnlyDictionary<string, ImmutableArray<GenesisExample>>? CandidatePools = null,
    IReadOnlyList<GenesisExample>? TrainingExamples = null,
    IReadOnlyList<GenesisAutonomousTrainingRound>? CreatorRounds = null,
    GenesisTrainingReport? Report = null);
