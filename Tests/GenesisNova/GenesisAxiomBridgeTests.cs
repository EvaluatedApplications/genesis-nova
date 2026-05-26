using GenesisNova.Axioms;

namespace GenesisNova.Tests;

public sealed class GenesisAxiomBridgeTests
{
    [Fact]
    public void WhenBridgeIsLoaded_ThenAllSixAxiomsAreMapped()
    {
        var mapped = GenesisAxiomBridge.Entries.Select(e => e.Axiom).Distinct().OrderBy(x => (int)x).ToArray();
        var expected = Enum.GetValues<GenesisAxiom>().OrderBy(x => (int)x).ToArray();

        Assert.Equal(expected, mapped);
    }

    [Fact]
    public void WhenCompositeObjectiveIsComputed_ThenAllAxiomLossesContribute()
    {
        var objective = new GenesisCompositeObjective(
            TokenWeight: 1.0,
            RouteWeight: 0.5,
            ConsistencyWeight: 0.2,
            ConservationWeight: 0.3,
            MemoryWeight: 0.1);

        var total = objective.ComputeTotal(
            tokenLoss: 2.0,
            routeLoss: 1.0,
            consistencyLoss: 0.5,
            conservationLoss: 0.5,
            memoryLoss: 1.0);

        Assert.Equal(2.85, total, precision: 6);
    }
}
