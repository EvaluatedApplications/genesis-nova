using GenesisNova.Data;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed record GenesisTrainingRun(
    string FilePath,
    int Epochs,
    int? IntrospectionCyclesPerEpoch,
    IReadOnlyList<GenesisExample>? Examples = null,
    GenesisTrainingReport? Report = null);

