namespace GenesisNova.Runtime;

internal sealed class RunTrainingStep
{
    private readonly GenesisRuntimeState _state;

    public RunTrainingStep(GenesisRuntimeState state)
    {
        _state = state;
    }

    public GenesisTrainTaskData Execute(GenesisTrainTaskData data)
    {
        StreamWriter? writer = null;
        Action<string>? logger = null;
        if (!string.IsNullOrWhiteSpace(data.LogPath))
        {
            var dir = Path.GetDirectoryName(data.LogPath);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);
            writer = new StreamWriter(data.LogPath, append: false);

            if (data.UiLogger != null)
            {
                logger = line =>
                {
                    writer.WriteLine(line);
                    writer.Flush();
                    data.UiLogger?.Invoke(line);
                };
            }
            else
            {
                logger = line =>
                {
                    writer.WriteLine(line);
                    writer.Flush();
                };
            }
        }
        else if (data.UiLogger != null)
        {
            logger = data.UiLogger;
        }

        try
        {
            var report = _state.Orchestrator.Train(
                data.Examples ?? [],
                data.Epochs,
                logger);
            return data with { Report = report };
        }
        finally
        {
            writer?.Dispose();
        }
    }
}
