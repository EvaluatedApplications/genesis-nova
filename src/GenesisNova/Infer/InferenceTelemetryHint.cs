namespace GenesisNova.Infer;

public readonly record struct InferenceTelemetryHint(
    double BiasScale,
    bool EnableContextBias)
{
    public static InferenceTelemetryHint Default { get; } = new(1.0, true);
}
