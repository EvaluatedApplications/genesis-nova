namespace GenesisNova.Axioms;

/// <summary>
/// Composite training objective. NOTE (loss-record cleanup): only the token term reflects a real
/// gradient. The route CE + L2 are backpropagated INSIDE the model before the loss record is built,
/// so the route/consistency/conservation/memory terms that used to be folded in here were decorative
/// (computed after the autograd graph was severed; never backpropped). They are no longer summed into
/// TotalLoss. The weights/parameters are retained for API/serialization compatibility but are inert:
/// ComputeTotal returns exactly the (weighted) token loss — the only genuinely trained signal.
/// </summary>
public sealed record GenesisCompositeObjective(
    double TokenWeight,
    double RouteWeight,
    double ConsistencyWeight,
    double ConservationWeight,
    double MemoryWeight)
{
    /// <summary>
    /// Total trained loss. The non-token arguments are accepted for source compatibility with existing
    /// call sites but are intentionally ignored — they never contributed a gradient. Total = token only.
    /// </summary>
    public double ComputeTotal(
        double tokenLoss,
        double routeLoss = 0.0,
        double consistencyLoss = 0.0,
        double conservationLoss = 0.0,
        double memoryLoss = 0.0)
    {
        // Only the token term carries a real gradient; the rest were decorative and are dropped.
        return TokenWeight * tokenLoss;
    }
}
