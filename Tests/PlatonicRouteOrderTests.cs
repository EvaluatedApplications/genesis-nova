using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Characterization test for the OCP route-registry refactor: the platonic reasoning ladder's ORDER is behavior
/// (glider-plan must precede the relational walks; the edge-following routes must come last). This pins the exact
/// priority sequence so a future reorder/extension of `_platonicRoutes` is a deliberate, visible change.
/// </summary>
public sealed class PlatonicRouteOrderTests
{
    [Fact]
    public void PlatonicLadder_PreservesPriorityOrder()
    {
        var state = NewState();
        Assert.Equal(
            new[]
            {
                "glider-plan",
                "expression-chain",
                "gru-query",
                "learned-function",
                "geometric-retrieval",
                "relation-edge",
                "concept-chain",
            },
            state.Inference.PlatonicRouteOrder);
    }

    private static GenesisRuntimeState NewState() =>
        new(new GenesisNovaConfig(Backend: ComputeBackend.Cpu, AutoPersist: false, AutoResume: false));
}
