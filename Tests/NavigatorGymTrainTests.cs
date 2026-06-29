using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// PER-CYCLE NAVIGATOR TRAINING ON THE LIVE SPACE (the overnight gym hook). Plants a small REAL-ish relational space
/// (entity → genus → domain → root, a 2-level taxonomy with strictly-increasing relational degree up the levels) into a
/// real <see cref="GenesisEvalAppRuntime"/>, then calls <see cref="GenesisEvalAppRuntime.TrainNavigatorCycle"/> several
/// times — exactly as <c>GymLoopAsync</c> does each cycle. Asserts the light per-cycle step actually LEARNS (cross-entropy
/// trends down, resolve% does not regress) and that it is GATED (no exception when a predict runs concurrently — both
/// share the one model-ops gate). Production face dim (1024) but a tiny model NN (HiddenSize 64) and tiny work knobs keep
/// it fast.
/// </summary>
public sealed class NavigatorGymTrainTests
{
    private readonly ITestOutputHelper _out;
    public NavigatorGymTrainTests(ITestOutputHelper o) => _out = o;

    private const int ObservePerEdge = 4;
    // maxMembers larger than the whole qualifying set ⇒ every cycle samples the SAME full set (no rotation window),
    // so the trainer converges on a fixed query set and the loss trend is clean to assert.
    private const int MaxMembers = 64;

    private static GenesisNovaConfig Config(string dir) => new(
        HiddenSize: 64,                 // tiny model NN — only the navigator is under test
        Backend: ComputeBackend.Cpu,
        AutoPersist: false,
        AutoResume: false,
        LocalStateDirectory: dir);
    // FaceDimensionOverride defaults to the production 1024 (the substrate face layout needs ≥ 202), navigator hidden 2048.

    // A 2-level taxonomy with strictly-increasing degree up the levels so the navigator's degree-guided ancestor climb
    // (categories are hubs) recovers GENUS → DOMAIN → ROOT without any hardcoded taxonomy.
    //   member (deg 1) → genus (deg 2) → domain (deg 3) → root (deg 4, padded with misc leaves to top the domains).
    private static void PlantTaxonomy(DialecticalSpace ds)
    {
        const string root = "thing";
        var domains = new[] { "animal", "plant" };
        var generaByDomain = new Dictionary<string, string[]>
        {
            ["animal"] = new[] { "mammal", "bird" },
            ["plant"] = new[] { "tree", "flower" },
        };
        var membersByGenus = new Dictionary<string, string[]>
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
        // Pad the root's degree above the domains (2 domains → deg 2) so the climb reaches it as the ROOT.
        foreach (var misc in new[] { "rock", "water", "cloud", "fire" }) Edge(misc, root);

        ds.FlushCloudBatch();
    }

    [Fact]
    public async Task TrainNavigatorCycle_OnLiveSpace_TrendsDown_AndIsGated()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navgym-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Config(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(cfg);
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            PlantTaxonomy(ds);

            // ── Several light cycles, exactly as the gym loop calls it. Collect the metrics. ──
            var results = new List<NavTrainCycleResult>();
            for (var cycle = 0; cycle < 6; cycle++)
            {
                var r = runtime.TrainNavigatorCycle(maxMembers: MaxMembers, epochs: 2);
                results.Add(r);
                _out.WriteLine($"cycle {cycle}: loss={r.Loss:F4} queries={r.Queries} resolve={r.ResolvePct:P0} abstain={r.AbstainPct:P0} runningResolve={runtime.NavResolveRunningPct:P0}");
            }

            // It actually sampled the live graph and trained (the climb derived cued ancestors from real relation depth).
            Assert.All(results, r => Assert.True(r.Queries > 0, "each cycle must sample cued queries from the live space"));
            var first = results.First();
            var last = results.Last();
            Assert.True(first.Loss > 0, "the first cycle has a real (positive) cross-entropy to descend from");

            // LEARNS: cross-entropy trends DOWN over the cycles (a fixed query set, cumulative weights).
            Assert.True(last.Loss < first.Loss,
                $"navigator CE must trend down across cycles (first={first.Loss:F4}, last={last.Loss:F4})");
            // Does not REGRESS the on-policy resolve rate.
            Assert.True(last.ResolvePct >= first.ResolvePct,
                $"resolve% must not regress (first={first.ResolvePct:P0}, last={last.ResolvePct:P0})");
            // The running EMA was populated (a later inspect tab reads it).
            Assert.True(runtime.NavResolveRunningPct >= 0.0);
            Assert.Equal(last, runtime.LastNavTrain);

            // ── GATED: a predict running concurrently must not throw (both take the one model-ops gate → serialized). ──
            var predict = Task.Run(async () =>
            {
                for (var i = 0; i < 4; i++) { try { await runtime.PredictAsync("dog", 4); } catch (OperationCanceledException) { } }
            });
            var ex = await Record.ExceptionAsync(async () =>
            {
                for (var i = 0; i < 4; i++) runtime.TrainNavigatorCycle(maxMembers: MaxMembers, epochs: 1);
                await predict;
            });
            Assert.Null(ex);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
