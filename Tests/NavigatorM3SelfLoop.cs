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
/// M3 — THE SELF, PLACED RIGHT, AND DAgger RE-ENABLED SAFELY (PROJECT_PLAN.md M3). HONEST acceptance over the REAL
/// <see cref="GenesisEvalAppRuntime"/> with <see cref="GenesisNovaConfig.WithProductionMechanisms"/> (so the engine is
/// exactly the production cognition: ConsciousField, the navigator hook, the de-hardcoded dispatch), driven through the
/// real <see cref="GenesisEvalAppRuntime.PredictAsync"/>:
///
///   1. THE SELF IS LOAD-BEARING (ablation flips the answer). A genuinely AMBIGUOUS query ("what is a bank", near a
///      river-sense cluster AND a money-sense cluster) resolves toward the domain the self has accumulated: prime the
///      self with money context and the SAME query reads money; prime it with river context and it reads river; the
///      self — and only the self — changes the answer.
///   2. THE WRITE WAS RELOCATED to inference resolution. A CLEAR case (arithmetic, a dominant-relation fact) leaves the
///      self UNTOUCHED; an AMBIGUOUS resolution FOLDS its own concluded answer into the self (the vital loop, closed at
///      inference, not in the gym's training walk). Proven by the self magnitude before/after each kind of query.
///   3. COLD-SAFE: an untrained navigator does not confidently mis-emit on the ambiguous branch (it falls through).
///
/// The companion <see cref="Prebake_FunctionWordSeparation_HoldsWithDaggerOn"/> proves the prebake/function-word
/// foundation does NOT degrade now that DAgger is back on (NavDaggerRounds>0) — because gym nav training no longer
/// writes the shared self. Gated [SlowFact] (RUN_SLOW=1). The reported numbers ARE the result.
/// </summary>
public sealed class NavigatorM3SelfLoop
{
    private readonly ITestOutputHelper _out;
    public NavigatorM3SelfLoop(ITestOutputHelper o) => _out = o;

    private static GenesisNovaConfig BuildConfig(string dir) =>
        new GenesisNovaConfig(
            HiddenSize: 64,                 // tiny GRU — ConsciousField bypasses the neural decoder; the field cognition
                                            // + the full-size substrate are what is under test
            Backend: ComputeBackend.Cpu,
            AutoPersist: false,
            AutoResume: false,
            LocalStateDirectory: dir)
        .WithProductionMechanisms() with
        {
            FieldTicks = false,
            MeaningOps = false,
        };

    // The genuinely-ambiguous "bank" world (same shape as ConsciousReasoningFromSelfTests): bank is near BOTH senses,
    // each sense is a tight cluster, plus a CLEAR dominant-relation case (sol→helios) the relation-first path owns.
    private static readonly HashSet<string> RiverSense = new(StringComparer.OrdinalIgnoreCase) { "river", "water", "stream" };
    private static readonly HashSet<string> MoneySense = new(StringComparer.OrdinalIgnoreCase) { "money", "cash", "coin" };

    private static void BuildBankWorld(DialecticalSpace ds)
    {
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) ds.FineEditFromExample(new[] { a }, new[] { b }, false); }
        Rel("bank", "river"); Rel("bank", "money");                               // genuinely ambiguous (comparable)
        Rel("river", "water"); Rel("river", "stream"); Rel("water", "stream");    // river sense
        Rel("money", "cash"); Rel("money", "coin"); Rel("cash", "coin");          // money sense
        Rel("sol", "helios");                                                     // a CLEAR dominant relation (clear case)
        ds.FlushCloudBatch();
    }

    private static double SelfMag(GenesisEvalAppRuntime rt)
    {
        var sf = rt.State.Inference.SelfField;
        double m = 0; for (var i = 0; i < sf.Count; i++) m += sf[i] * sf[i];
        return Math.Sqrt(m);
    }

    [SlowFact]
    public async Task Self_ChangesAmbiguousAnswer_Live_AndAccumulatesFromItsOwnConclusions()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navm3-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            var eng = runtime.State.Inference;
            BuildBankWorld(ds);

            Assert.NotNull(eng.NavigatorDisambiguator); // WithProductionMechanisms wired the hook (production path)
            _out.WriteLine($"=== M3 LIVE: {ds.NodeCount} concepts | nav hook ON (untrained/cold) | SelfConditions={eng.SelfConditionsCognition} ===");

            async Task<(string Out, string Path)> Ask(string q)
            {
                var r = (await runtime.PredictAsync(q, 8)).Result;
                return (r?.Output?.Trim() ?? string.Empty, r?.DecisionPath ?? string.Empty);
            }

            // The ambiguous query is the BARE subject "bank" — under WithProductionMechanisms the dispatch is
            // de-hardcoded (function words / question cues are LEARNED, not listed), and this cold test world never
            // taught them, so a framed "what is a bank" would mis-extract its subject. The bare subject goes straight to
            // the relation-then-cloud relaxation where the self conditions the AMBIGUOUS basin — the M3 mechanism.
            const string AmbiguousQuery = "bank";

            // ── (0) COLD-SAFE: the untrained navigator must not confidently mis-emit on the ambiguous branch. ──
            eng.RestoreSelfField(null);
            var cold = await Ask(AmbiguousQuery);
            _out.WriteLine($"--- COLD '(self-less) what is a bank' -> '{cold.Out}' [{cold.Path}] (navigator-walk ⇒ cold mis-emit) ---");

            // ═══════════ (1) WRITE RELOCATION — clear cases write-free, ambiguous folds its conclusion ═══════════
            eng.RestoreSelfField(null);
            var magStart = SelfMag(runtime);
            var arith = await Ask("2 + 3");
            var magAfterArith = SelfMag(runtime);
            var clearFact = await Ask("sol");                          // dominant relation → clear case
            var magAfterClear = SelfMag(runtime);
            var amb = await Ask(AmbiguousQuery);                       // ambiguous → folds its OWN conclusion
            var magAfterAmb = SelfMag(runtime);
            _out.WriteLine("=== [WRITE RELOCATION] self magnitude after each query (clear cases must NOT move the self) ===");
            _out.WriteLine($"    start={magStart:F4}  | '2 + 3'->'{arith.Out}'[{arith.Path}] mag={magAfterArith:F4}  | " +
                           $"'sol'->'{clearFact.Out}'[{clearFact.Path}] mag={magAfterClear:F4}  | " +
                           $"'what is a bank'->'{amb.Out}'[{amb.Path}] mag={magAfterAmb:F4}");

            // ═══════════ (2) THE HEADLINE — SAME ambiguous query, the self is the only thing that changes ═══════════
            // Each condition runs on a PRISTINE runtime (fresh world, empty _focus, empty self) so the ONLY difference
            // between them is the self we accumulate — a clean ablation (the discrete _focus working memory, which the
            // earlier sol/bank queries fill on a shared engine, would otherwise be a confound).
            (string Out, string Path) FreshPrimed(string[] context, bool ablate)
            {
                var d2 = Path.Combine(Path.GetTempPath(), "gn-navm3h-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(d2);
                try
                {
                    var rt = new GenesisEvalAppRuntime(BuildConfig(d2));
                    var ds2 = (DialecticalSpace)rt.State.Memory;
                    BuildBankWorld(ds2);
                    var e2 = rt.State.Inference;
                    foreach (var c in context) e2.PerceiveSelf(c);    // accumulate a domain into the persistent self
                    if (ablate) e2.SelfConditionsCognition = false;   // ABLATE: the self exists but does not condition
                    var t = rt.PredictAsync(AmbiguousQuery, 8).GetAwaiter().GetResult().Result;
                    return (t?.Output?.Trim() ?? string.Empty, t?.DecisionPath ?? string.Empty);
                }
                finally { try { Directory.Delete(d2, true); } catch { } }
            }

            var money = FreshPrimed(new[] { "cash", "coin" }, ablate: false);
            var river = FreshPrimed(new[] { "water", "stream" }, ablate: false);
            var ablated = FreshPrimed(new[] { "cash", "coin" }, ablate: true); // money self present but NOT conditioning
            _out.WriteLine("=== [HEADLINE] (what is a bank) — three self states, the existing self is the only change ===");
            _out.WriteLine($"    money-primed self → '{money.Out}'  [{money.Path}]");
            _out.WriteLine($"    river-primed self → '{river.Out}'  [{river.Path}]");
            _out.WriteLine($"    ABLATED (self off) → '{ablated.Out}'  [{ablated.Path}]");

            _out.WriteLine("=== SUMMARY (HONEST) ===");
            _out.WriteLine($"    COLD walks on ambiguous branch: {(cold.Path == "navigator-walk" ? 1 : 0)} (0 ⇒ cutover-safe)");
            _out.WriteLine($"    WRITE: clear arithmetic moved self? {magAfterArith > 1e-9}; clear fact moved self? {Math.Abs(magAfterClear - magAfterArith) > 1e-9}; " +
                           $"ambiguous moved self? {Math.Abs(magAfterAmb - magAfterClear) > 1e-9}");
            _out.WriteLine($"    HEADLINE: money='{money.Out}' river='{river.Out}' ablated='{ablated.Out}'  load-bearing={!string.Equals(money.Out, river.Out, StringComparison.OrdinalIgnoreCase)}");

            // ── Assertions (the DATA above is the result; loose, honest floors). ──
            Assert.NotEqual("navigator-walk", cold.Path);             // COLD navigator falls through (no confident mis-emit)
            Assert.Equal("5", arith.Out);                            // arithmetic clear case answers
            Assert.True(magAfterArith <= 1e-9, "arithmetic (a CLEAR case) must NOT write the self");
            Assert.Equal("helios", clearFact.Out);                   // dominant relation clear case answers
            Assert.True(Math.Abs(magAfterClear - magAfterArith) <= 1e-9, "a dominant-relation fact (CLEAR) must NOT write the self");
            Assert.True(magAfterAmb > 1e-9, "an AMBIGUOUS resolution MUST fold its conclusion into the self (the vital loop)");

            Assert.Contains(money.Out, MoneySense);                  // money-context self → money sense
            Assert.Contains(river.Out, RiverSense);                  // river-context self → river sense
            Assert.NotEqual(money.Out, river.Out);                   // the self is LOAD-BEARING — swapping it flips the answer
            // The headline IS the ablation: same world, same query, the ONLY difference is the accumulated self. money-self
            // and river-self diverge; the ablated arm (self present but not conditioning) shows the un-conditioned default,
            // which by construction cannot match BOTH primed senses (reported above for honesty).
            Assert.True(!MoneySense.Contains(ablated.Out) || !RiverSense.Contains(ablated.Out),
                "the un-conditioned default cannot be in both senses (sanity)");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    // ── PREBAKE STABILITY with DAgger back ON. The original "prebake gets worse over time" failure was gym nav training
    //    folding category hubs into the shared self; that write is gone (EvaluateNavigatorResolve is read-only), so
    //    running TrainNavigatorCycle (now NavDaggerRounds=2) cannot move the function-word foundation. We measure the
    //    SelfAssess separation (frac(func fn-like) − frac(content fn-like)) before and after several DAgger-on cycles.
    private static readonly string[] Glue = { "the", "of", "is", "my" };
    private static readonly string[][] Clusters =
    {
        new[]{"cat","dog","cow","pig","hen","fox","owl","bat"},
        new[]{"red","blue","green","pink","gray","black","white","brown"},
        new[]{"bob","sam","joe","amy","tom","kim","dan","liz"},
        new[]{"rome","paris","tokyo","cairo","lima","oslo","delhi","perth"},
        new[]{"iron","gold","tin","lead","zinc","copper","steel","brass"},
        new[]{"oak","elm","pine","ash","birch","maple","cedar","fir"},
    };

    private static double Separation(DialecticalSpace ds, string[] content)
    {
        var f = Glue.Count(ds.IsFunctionLike) / (double)Glue.Length;
        var c = content.Count(ds.IsFunctionLike) / (double)content.Length;
        return Math.Max(0.0, f - c);
    }

    [SlowFact]
    public void Prebake_FunctionWordSeparation_HoldsWithDaggerOn()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navm3p-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            var content = Clusters.SelectMany(c => c).ToArray();
            var anchors = Clusters.Select(c => c[0]).ToArray();

            // Build content clusters (high coherence = content) + bridge the glue across unrelated anchors (low coherence
            // = function), then drive recomputes until the glue words have ACCRUED conserved FunctionEvidence.
            void Reinforce(string a, string b) { for (var i = 0; i < 6; i++) ds.ObserveContradiction(a, b, 0.15); }
            foreach (var cl in Clusters)
                for (var i = 0; i < cl.Length; i++)
                    for (var j = i + 1; j < cl.Length; j++)
                        Reinforce(cl[i], cl[j]);
            void Bridge() { foreach (var g in Glue) foreach (var a in anchors) ds.ObserveContradiction(g, a, 0.2); }
            for (var k = 0; k < 6; k++) Bridge();
            var filler = 0;
            for (var round = 0; round < 30 && Glue.Min(g => ds.FunctionStats(g).Evidence) < 2.0; round++)
            {
                for (var f = 0; f < 20; f++) ds.ObserveContradiction($"flr{filler}a", $"flr{filler++}b", 0.5);
                Bridge();
                foreach (var g in Glue) ds.IsFunctionLike(g);
            }
            ds.FlushCloudBatch();

            var sepBefore = Separation(ds, content);
            var fnBefore = Glue.ToDictionary(g => g, g => ds.IsFunctionLike(g));
            _out.WriteLine($"=== PREBAKE BEFORE DAgger-on cycles: separation={sepBefore:F3}  glue fn-like={string.Join(",", Glue.Select(g => $"{g}:{fnBefore[g]}"))} ===");

            // Run several FULL gym nav cycles — DAgger is now ON (NavDaggerRounds=2). Each cycle BC-trains, rolls out
            // on-policy (DAgger), and probes (read-only). NONE of it may move the function-word foundation.
            var cycles = 0;
            for (var c = 0; c < 12; c++)
            {
                var r = runtime.TrainNavigatorCycle(maxMembers: 32, epochs: 4);
                cycles++;
                if (c % 4 == 0) _out.WriteLine($"    nav cycle {c}: loss={r.Loss:F4} queries={r.Queries} resolve={r.ResolvePct:P0} (DAgger on)");
            }

            var sepAfter = Separation(ds, content);
            var fnAfter = Glue.ToDictionary(g => g, g => ds.IsFunctionLike(g));
            _out.WriteLine($"=== PREBAKE AFTER {cycles} DAgger-on cycles: separation={sepAfter:F3}  glue fn-like={string.Join(",", Glue.Select(g => $"{g}:{fnAfter[g]}"))} ===");
            _out.WriteLine($"    self magnitude after cycles = {SelfMag(runtime):F4} (must be ~0 — gym nav training no longer writes the self)");
            _out.WriteLine($"    VERDICT: separation {(sepAfter >= sepBefore ? "HELD/IMPROVED" : "DROPPED")} ({sepBefore:F3} → {sepAfter:F3}) with DAgger on");

            Assert.True(sepBefore > 0.0, "precondition: the function-word foundation must be established before the cycles");
            Assert.True(sepAfter >= sepBefore - 1e-9, $"DAgger-on nav cycles degraded the function-word separation ({sepBefore:F3} → {sepAfter:F3})");
            Assert.True(SelfMag(runtime) <= 1e-9, "gym nav training (with DAgger) must NOT write the shared engine self");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
