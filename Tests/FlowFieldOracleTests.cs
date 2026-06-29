using System.Collections.Generic;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE NAVIGATOR FLOW-FIELD ORACLE (PLATONIC_NAVIGATOR.md §7 / §11.1). Proves the deterministic teacher — backward
/// Dijkstra from a known answer over the platonic RELATION graph — fills cost[] (dense reward, defined everywhere
/// reachable) and next[] (expert policy at EVERY node, the DAgger-for-free property). NO neural net, no training.
/// Built on a small hand-planted fact graph in a fresh DialecticalSpace (dim 512); facts are planted via the
/// substrate's native learn (ObserveContradiction, contradiction 0 → relation strength 1.0).
/// </summary>
public sealed class FlowFieldOracleTests
{
    private readonly ITestOutputHelper _out;
    public FlowFieldOracleTests(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024; // production face: address-space layout active, orbital [416,1024) = 608 learned dims

    // Plant a relation edge a↔b (contradiction 0 → strength 1.0). Relations index both endpoints, so the edge is
    // traversable in reverse — exactly what backward Dijkstra needs.
    private static void Relate(DialecticalSpace s, string a, string b) => s.ObserveContradiction(a, b, 0.0);

    // ── 1. One-hop entity → category ────────────────────────────────────────────────────────────────────────────────
    [Fact]
    public void OneHop_EntitiesPointAtCategory_CostOne()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var entities = new[] { "apple", "banana", "pear", "grape" };
        foreach (var e in entities) Relate(space, e, "fruit"); // entity — category

        var field = FlowFieldOracle.Compute(space, "fruit");

        Assert.True(field.TryCost("fruit", out var cAns) && cAns == 0.0, "Cost[answer] must be 0");
        foreach (var e in entities)
        {
            Assert.True(field.TryNext(e, out var nxt), $"{e} must have a Next");
            Assert.Equal("fruit", nxt);                       // step toward the answer in one hop
            Assert.True(field.TryCost(e, out var c) && c == 1.0, $"{e} should be 1 hop, got {(field.TryCost(e, out var cc) ? cc : double.NaN)}");
            _out.WriteLine($"  {e}: Next={nxt} Cost={c}");
        }
        Assert.False(field.Truncated);
    }

    // ── 2. Two-hop chain: the field gives the right NEXT step from a NON-adjacent node (what A* would not) ───────────
    [Fact]
    public void TwoHopChain_NextStepFromNonAdjacentNode()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        Relate(space, "alpha", "beta");   // a — b
        Relate(space, "beta", "gamma");   // b — c   (answer = gamma)

        var field = FlowFieldOracle.Compute(space, "gamma");

        Assert.True(field.TryNext("alpha", out var na) && na == "beta",  "from the non-adjacent node alpha, step to beta");
        Assert.True(field.TryNext("beta",  out var nb) && nb == "gamma", "from beta, step to gamma");
        Assert.True(field.TryCost("alpha", out var ca) && ca == 2.0, $"alpha is 2 hops, got {ca}");
        Assert.True(field.TryCost("beta",  out var cb) && cb == 1.0, $"beta is 1 hop, got {cb}");
        Assert.True(field.TryCost("gamma", out var cg) && cg == 0.0, "gamma is the answer");
        _out.WriteLine($"  alpha: Next={na} Cost={ca} | beta: Next={nb} Cost={cb} | gamma Cost={cg}");
    }

    // ── 3. Coverage: every connected node gets a Cost+Next; a disconnected node gets neither ────────────────────────
    [Fact]
    public void Coverage_ReachesConnected_SkipsDisconnected()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        // Connected component flowing to "home": near — home, far — near.
        Relate(space, "near", "home");
        Relate(space, "far", "near");
        // A SEPARATE component with no path to home.
        Relate(space, "island", "lagoon");

        var field = FlowFieldOracle.Compute(space, "home");

        foreach (var n in new[] { "near", "far" })
        {
            Assert.True(field.TryCost(n, out _), $"{n} is connected → must have a Cost");
            Assert.True(field.TryNext(n, out _), $"{n} is connected → must have a Next");
        }
        Assert.False(field.TryCost("island", out _), "disconnected node has no Cost");
        Assert.False(field.TryNext("island", out _), "disconnected node has no Next");
        Assert.False(field.TryCost("lagoon", out _), "disconnected node has no Cost");
        _out.WriteLine($"  reached: near={field.Cost.ContainsKey("near")} far={field.Cost.ContainsKey("far")} | island reached={field.Cost.ContainsKey("island")}");
    }

    // ── 4. Follow-the-field walk: applying Next[] reaches the answer within Cost[start] steps ───────────────────────
    [Fact]
    public void FollowField_ReachesAnswer_WithinCostSteps()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        // A 4-deep chain so the walk is non-trivial: n1 — n2 — n3 — n4 — answer.
        Relate(space, "n1", "n2");
        Relate(space, "n2", "n3");
        Relate(space, "n3", "n4");
        Relate(space, "n4", "answer");

        var field = FlowFieldOracle.Compute(space, "answer");
        Assert.True(field.TryCost("n1", out var startCost));
        Assert.Equal(4.0, startCost);

        var node = "n1";
        var steps = 0;
        var budget = (int)startCost + 1; // must arrive within Cost[start] steps
        var path = new List<string> { node };
        while (node != "answer" && steps < budget)
        {
            Assert.True(field.TryNext(node, out var nxt), $"no Next at {node}");
            node = nxt;
            path.Add(node);
            steps++;
        }
        _out.WriteLine($"  walk: {string.Join(" -> ", path)} ({steps} steps, Cost[start]={startCost})");
        Assert.Equal("answer", node);
        Assert.True(steps <= startCost, $"reached the answer in {steps} steps, must be <= Cost[start]={startCost}");
    }

    // ── 5. Truncation is flagged (a silent cap must never pass unnoticed — repo norm) ───────────────────────────────
    [Fact]
    public void Truncation_IsFlagged_AndLogged()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        // A chain long enough to exceed a tiny cap.
        for (var i = 0; i < 20; i++) Relate(space, $"c{i}", $"c{i + 1}");

        string? note = null;
        var field = FlowFieldOracle.Compute(space, "c20", maxNodes: 5, onTruncate: m => note = m);

        Assert.True(field.Truncated, "field beyond the cap must be flagged truncated");
        Assert.NotNull(note);
        _out.WriteLine($"  truncation note: {note}");
    }
}
