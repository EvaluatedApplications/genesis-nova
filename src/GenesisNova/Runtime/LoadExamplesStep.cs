using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class LoadExamplesStep
{
    public GenesisTrainTaskData Execute(GenesisTrainTaskData data)
        => data with { Examples = GenesisTrainingDataLoader.LoadFromFile(data.FilePath) };
}
