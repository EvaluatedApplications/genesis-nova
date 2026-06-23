namespace GenesisNova.Infer;

public readonly record struct InferenceTelemetryHint(
    double BiasScale)
{
    public static InferenceTelemetryHint Default { get; } = new(1.0);
}
