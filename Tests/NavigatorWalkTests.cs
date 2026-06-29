using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE NAVIGATOR WALK LOOP (PLATONIC_NAVIGATOR.md §2 / §5.1 / §5.2 / §8). Proves the ENVIRONMENT a policy drives to
/// walk the platonic address space to an answer, with the deterministic FLOW-FIELD ORACLE as the policy (no NN). The
/// walk emits a target coordinate, the lattice (TryLand) lands the step, and it repeats until it stands on the answer,
/// halts, or abstains (budget / dead-end / cycle). Fact graphs are hand-planted via ObserveContradiction (contradiction
/// 0 → relation strength 1.0), the same style as FlowFieldOracleTests. Dim 512 (the decodable address space).
/// </summary>
public sealed class NavigatorWalkTests
{
    private readonly ITestOutputHelper _out;
    public NavigatorWalkTests(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024; // production face: address space active, orbital [416,1024) = 608 learned dims

    private static void Relate(DialecticalSpace s, string a, string b) => s.ObserveContradiction(a, b, 0.0);

    // Drive a walk toward `answer` from `start` using the flow-field oracle as the policy.
    private static NavWalkResult OracleWalk(DialecticalSpace space, string start, string answer, NavWalkOptions options)
    {
        var field = FlowFieldOracle.Compute(space, answer);
        var policy = new FlowFieldPolicy(field, space);
        space.TryGetConceptFace(answer, out var goalFace);
        return new NavigatorWalk().Walk(space, start, goalFace, answer, policy, options);
    }

    // ── 1. Oracle walk reaches the answer along a multi-hop chain, in exactly Cost[start] steps, following Next[] ─────
    [Fact]
    public void OracleWalk_MultiHopChain_ReachesAnswer_StepsEqualCost()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        // a — b — c — answer (so Cost[a]=3, Cost[b]=2, Cost[c]=1).
        Relate(space, "a", "b");
        Relate(space, "b", "c");
        Relate(space, "c", "answer");

        var field = FlowFieldOracle.Compute(space, "answer");

        foreach (var (start, expectedCost, expectedPath) in new[]
        {
            ("a", 3.0, new[] { "a", "b", "c", "answer" }),
            ("b", 2.0, new[] { "b", "c", "answer" }),
            ("c", 1.0, new[] { "c", "answer" }),
        })
        {
            var result = OracleWalk(space, start, "answer", new NavWalkOptions(MaxSteps: 16));
            _out.WriteLine($"  from {start}: reached={result.Reached} steps={result.Steps} (cost={expectedCost}) " +
                           $"path={string.Join("->", result.Trajectory)} final={result.FinalSymbol}");

            Assert.True(result.Reached, $"walk from {start} must reach the answer");
            Assert.Equal("answer", result.FinalSymbol);
            Assert.Equal((int)expectedCost, result.Steps);                       // steps == Cost[start]
            Assert.True(field.TryCost(start, out var c) && (int)c == result.Steps, "steps must equal the field cost");
            Assert.Equal(expectedPath, result.Trajectory.ToArray());             // trajectory follows Next[]
        }
    }

    // ── 2. One-hop facts: several entities → category, each reaches the category in 1 step ──────────────────────────
    [Fact]
    public void OracleWalk_OneHopFacts_ReachCategoryInOneStep()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var entities = new[] { "apple", "banana", "pear", "grape" };
        foreach (var e in entities) Relate(space, e, "fruit");

        foreach (var e in entities)
        {
            var result = OracleWalk(space, e, "fruit", new NavWalkOptions(MaxSteps: 8));
            _out.WriteLine($"  {e}: reached={result.Reached} steps={result.Steps} path={string.Join("->", result.Trajectory)}");
            Assert.True(result.Reached, $"{e} must reach fruit");
            Assert.Equal("fruit", result.FinalSymbol);
            Assert.Equal(1, result.Steps);
            Assert.Equal(new[] { e, "fruit" }, result.Trajectory.ToArray());
        }
    }

    // ── 3. Abstain on a disconnected start: no path to the answer → Reached=false, no loop, no exception ─────────────
    [Fact]
    public void OracleWalk_DisconnectedStart_Abstains()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        Relate(space, "near", "home");      // connected component → home
        Relate(space, "far", "near");
        Relate(space, "island", "lagoon");  // a SEPARATE component, no path to home

        var result = OracleWalk(space, "island", "home", new NavWalkOptions(MaxSteps: 16));
        _out.WriteLine($"  island->home: reached={result.Reached} steps={result.Steps} final={result.FinalSymbol}");

        Assert.False(result.Reached, "a disconnected start must abstain");
        Assert.NotEqual("home", result.FinalSymbol);
        Assert.True(result.Steps <= 16, "must not loop");
    }

    // ── 4. MaterialiseOnSuccess: a walk through a LATENT coordinate materialises it (genesis-tick growth, §5.2) ──────
    //  The flow-field oracle only traverses STORED relations, so to put a genuinely-latent coordinate on the path we
    //  use a tiny custom policy that emits the (homomorphic, never-stored) face of the number 141 — a latent coordinate
    //  the lattice decodes & lands on exactly (§5.1). Before the walk 141 is absent; after a MaterialiseOnSuccess walk
    //  every trajectory concept (incl. 141) is present.
    [Fact]
    public void Walk_MaterialiseOnSuccess_RealisesLatentTrajectoryCoordinate()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        Relate(space, "start", "origin"); // make 'start' a real concept to stand on

        Assert.False(space.ContainsConcept("141"), "141 must be latent (never stored) before the walk");
        Assert.True(space.TryGetConceptFace("141", out var latentFace), "a number's homomorphic face is always available");

        var policy = new EmitThenHaltPolicy("141", latentFace);
        var result = new NavigatorWalk().Walk(
            space, "start", latentFace, "141", policy,
            new NavWalkOptions(MaxSteps: 8, MaterialiseOnSuccess: true));

        _out.WriteLine($"  before: contains(141)=False | reached={result.Reached} final={result.FinalSymbol} " +
                       $"path={string.Join("->", result.Trajectory)} | after: contains(141)={space.ContainsConcept("141")}");

        Assert.True(result.Reached, "walk must reach the latent target 141");
        Assert.Equal("141", result.FinalSymbol);
        Assert.True(space.ContainsConcept("141"), "141 must be MATERIALISED (realised) after the successful walk");
        foreach (var sym in result.Trajectory)
            Assert.True(space.ContainsConcept(sym), $"every trajectory concept must be present after materialise: {sym}");
    }

    // ── 5. Budget halt: a long chain with MaxSteps < its length → Reached=false, Steps<=MaxSteps (light-cone, §8) ────
    [Fact]
    public void OracleWalk_BudgetSmallerThanChain_HaltsWithoutReaching()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        // A 6-hop chain: h0 — h1 — ... — h6 (Cost[h0]=6).
        for (var i = 0; i < 6; i++) Relate(space, $"h{i}", $"h{i + 1}");

        const int budget = 3;
        var result = OracleWalk(space, "h0", "h6", new NavWalkOptions(MaxSteps: budget));
        _out.WriteLine($"  h0->h6 budget={budget}: reached={result.Reached} steps={result.Steps} " +
                       $"final={result.FinalSymbol} path={string.Join("->", result.Trajectory)}");

        Assert.False(result.Reached, "the answer is beyond the budget → must not reach");
        Assert.True(result.Steps <= budget, $"steps {result.Steps} must be <= budget {budget}");
        Assert.NotEqual("h6", result.FinalSymbol);
    }

    // A minimal non-oracle policy: emit a fixed target face until standing on the goal, then halt. Used only to put a
    // latent coordinate on the path for the materialise test (the oracle cannot route through unstored coordinates).
    private sealed class EmitThenHaltPolicy : INavPolicy
    {
        private readonly string _goal;
        private readonly double[] _target;
        public EmitThenHaltPolicy(string goal, double[] target) { _goal = goal; _target = target; }

        public NavDecision Decide(NavState state) =>
            string.Equals(state.CurrentSymbol, _goal, StringComparison.Ordinal)
                ? new NavDecision(Array.Empty<double>(), Halt: true)
                : new NavDecision(_target, Halt: false);
    }
}
