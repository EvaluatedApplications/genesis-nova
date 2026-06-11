namespace GenesisNova.Data;

public sealed record GenesisExample(
    string Input,
    string Output,
    GenesisTrainingExampleKind TrainingKind = GenesisTrainingExampleKind.PromptAnswer,
    string? SourceCreatorName = null,
    int? RouteLabel = null);
