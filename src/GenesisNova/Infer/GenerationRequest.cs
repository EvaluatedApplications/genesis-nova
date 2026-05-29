namespace GenesisNova.Infer;

public sealed record GenerationRequest(
    string Input,
    int MaxNewTokens = 48,
    int ChunkTokenBudget = 16);
