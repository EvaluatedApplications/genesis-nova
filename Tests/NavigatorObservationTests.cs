using System;
using System.IO;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE READ-ONLY OBSERVATION SURFACES for the platonic navigator (the REPL `/nav` walk + the Inspect "Navigator" panel).
/// There is no headless WinForms message loop, so this covers the UNDERLYING logic the UI calls: the gated diagnostics
/// accessor (<see cref="GenesisEvalAppRuntime.GetNavigatorDiagnostics"/>) returns sane values after a few training
/// cycles, and the gated walk helper (<see cref="GenesisEvalAppRuntime.WalkNavigator"/>) returns a real trajectory for a
/// planted concept, a clear message for an unknown one, and never throws. PURELY OBSERVATIONAL — no training/persistence
/// is exercised here beyond seeding the navigator so there is something to observe. Mirrors NavigatorGymTrainTests' setup.
/// </summary>
public sealed class NavigatorObservationTests
{
    private readonly ITestOutputHelper _out;
    public NavigatorObservationTests(ITestOutputHelper o) => _out = o;

    private const int ObservePerEdge = 4;
    private const int MaxMembers = 64;

    private static GenesisNovaConfig Config(string dir) => new(
        HiddenSize: 64,                 // tiny model NN — only the navigator observation surfaces are under test
        Backend: ComputeBackend.Cpu,
        AutoPersist: false,
        AutoResume: false,
        LocalStateDirectory: dir);

    // Same 2-level taxonomy (strictly-increasing degree up the levels) as NavigatorGymTrainTests.
    private static void PlantTaxonomy(DialecticalSpace ds)
    {
        const string root = "thing";
        var domains = new[] { "animal", "plant" };
        var generaByDomain = new System.Collections.Generic.Dictionary<string, string[]>
        {
            ["animal"] = new[] { "mammal", "bird" },
            ["plant"] = new[] { "tree", "flower" },
        };
        var membersByGenus = new System.Collections.Generic.Dictionary<string, string[]>
        {
            ["mammal"] = new[] { "dog" },
            ["bird"] = new[] { "robin" },
            ["tree"] = new[] { "oak" },
            ["flower"] = new[] { "rose" },
        };

        void Edge(string child, string parent)
        {
            for (var i = 0; i < ObservePerEdge; i++) ds.ObserveContradiction(child, parent, 0.0);
        }

        foreach (var domain in domains)
        {
            Edge(domain, root);
            foreach (var genus in generaByDomain[domain])
            {
                Edge(genus, domain);
                foreach (var m in membersByGenus[genus]) Edge(m, genus);
            }
        }
        foreach (var misc in new[] { "rock", "water", "cloud", "fire" }) Edge(misc, root);
        ds.FlushCloudBatch();
    }

    [Fact]
    public void Diagnostics_AndWalk_ObserveTheTrainedNavigator_NoThrow()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navobs-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(Config(dir));
            // Match the production app (WithProductionMechanisms turns self-conditioning on) so the `/nav` walk reads the
            // live self — the bare test config defaults it off.
            runtime.State.Inference.SelfConditionsCognition = true;
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            PlantTaxonomy(ds);

            // Seed the navigator with a few light cycles. M3: gym nav training is READ-ONLY w.r.t. the engine self — it
            // does NOT fold traversed (category-hub) concepts into the shared _selfField (that was the pollution that
            // forced DAgger off). The self is shaped only by INFERENCE resolutions, exercised below by explicit priming.
            NavTrainCycleResult last = default;
            for (var c = 0; c < 4; c++) last = runtime.TrainNavigatorCycle(maxMembers: MaxMembers, epochs: 2);
            Assert.True(last.Queries > 0, "training must have sampled cued queries from the live space");

            // ── DIAGNOSTICS ACCESSOR (what the Inspect panel reads) ──
            var diag = runtime.GetNavigatorDiagnostics();
            _out.WriteLine($"diag: loss={diag.LastLoss:F4} queries={diag.LastQueries} resolve={diag.LastResolvePct:P0} " +
                           $"abstain={diag.LastAbstainPct:P0} running={diag.RunningResolvePct:P0} selfMag={diag.SelfMagnitude:F3} " +
                           $"selfCond={diag.SelfConditions} focus={diag.SelfFocus.Count}");
            Assert.Equal(last.Loss, diag.LastLoss, 6);
            Assert.Equal(last.Queries, diag.LastQueries);
            Assert.True(diag.RunningResolvePct >= 0.0 && diag.RunningResolvePct <= 1.0);
            Assert.True(diag.LastAbstainPct is >= 0.0 and <= 1.0);
            Assert.True(diag.SelfConditions, "self-conditioning is on by default in the living-field mode");
            // M3 INVARIANT: gym nav training did NOT write the shared self (no pollution) → the self is still empty.
            Assert.Equal(0.0, diag.SelfMagnitude);
            Assert.Empty(diag.SelfFocus);

            // Now PRIME the self the way inference does (the mind dwelling on concepts) so the Inspect "self focus" has
            // something to show — proving the diagnostics surface the live self once it is non-empty.
            runtime.State.Inference.PerceiveSelf("dog");
            runtime.State.Inference.PerceiveSelf("robin");
            var primed = runtime.GetNavigatorDiagnostics();
            Assert.True(primed.SelfMagnitude > 0.0, "after priming, the engine self is non-empty (unit length)");
            Assert.NotEmpty(primed.SelfFocus);
            Assert.All(primed.SelfFocus, f => Assert.False(string.IsNullOrWhiteSpace(f.Concept)));
            // Cosine similarities are well-formed and ranked descending.
            Assert.All(primed.SelfFocus, f => Assert.InRange(f.Similarity, -1.0001, 1.0001));
            for (var i = 1; i < primed.SelfFocus.Count; i++)
                Assert.True(primed.SelfFocus[i - 1].Similarity >= primed.SelfFocus[i].Similarity - 1e-9, "focus must be ranked");

            // ── WALK HELPER (what `/nav <concept> [cue]` runs) — a planted concept yields a real trajectory ──
            var walk = runtime.WalkNavigator("dog", "genus");
            _out.WriteLine($"walk: anchor={walk.Anchor} cue={walk.Cue} answer={walk.Answer} reached={walk.Reached} " +
                           $"self={walk.SelfConditioned} traj=[{string.Join("→", walk.Trajectory)}] msg={walk.Message}");
            Assert.Equal("dog", walk.Anchor);
            Assert.Equal(GenesisNova.Cognition.Navigator.NavCue.Genus, walk.Cue);
            Assert.NotEmpty(walk.Trajectory);
            Assert.Equal("dog", walk.Trajectory[0]);                 // the walk always starts where it stands
            Assert.False(string.IsNullOrWhiteSpace(walk.Answer));
            Assert.False(string.IsNullOrWhiteSpace(walk.Message));
            Assert.True(walk.SelfConditioned, "the live (now non-empty) engine self conditions the walk");

            // The walk is stashed for the Inspect panel's "last walk" line.
            var afterWalk = runtime.GetNavigatorDiagnostics();
            Assert.NotNull(afterWalk.LastWalk);
            Assert.Equal("dog", afterWalk.LastWalk!.Value.Anchor);

            // Default cue parses to GENUS; an unrecognised cue keyword also falls to GENUS.
            Assert.Equal(GenesisNova.Cognition.Navigator.NavCue.Genus, runtime.WalkNavigator("dog").Cue);
            Assert.Equal(GenesisNova.Cognition.Navigator.NavCue.Genus, runtime.WalkNavigator("dog", "wat").Cue);
            Assert.Equal(GenesisNova.Cognition.Navigator.NavCue.Domain, runtime.WalkNavigator("dog", "domain").Cue);
            Assert.Equal(GenesisNova.Cognition.Navigator.NavCue.Root, runtime.WalkNavigator("dog", "root").Cue);

            // ── UNKNOWN CONCEPT — a clear message, no crash ──
            var unknown = runtime.WalkNavigator("zzznotaconcept", null);
            Assert.False(unknown.Reached);
            Assert.Contains("not a live concept", unknown.Message, StringComparison.OrdinalIgnoreCase);
            Assert.NotEmpty(unknown.Trajectory);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
