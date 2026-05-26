namespace GenesisNova.Axioms;

public sealed record GenesisCompositeObjective(
    double TokenWeight,
    double RouteWeight,
    double ConsistencyWeight,
    double ConservationWeight,
    double MemoryWeight)
{
    public double ComputeTotal(
        double tokenLoss,
        double routeLoss,
        double consistencyLoss,
        double conservationLoss,
        double memoryLoss)
    {
        return (TokenWeight * tokenLoss)
            + (RouteWeight * routeLoss)
            + (ConsistencyWeight * consistencyLoss)
            + (ConservationWeight * conservationLoss)
            + (MemoryWeight * memoryLoss);
    }
}

