namespace GenesisNova.Runtime;

internal sealed class BestLossTracker
{
    public double BestLoss { get; set; } = double.MaxValue;
}
