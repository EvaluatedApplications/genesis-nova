using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// THE END-TO-END PROOF (user: "does it actually remember the name? warm via the prebake, then teach+recall").
// The bare unit tests proved the Merge LOGIC. The earlier gym-mix warm could NOT warm the LEARNED function-word
// signal (IsFunctionLike) — only 2 framings per skill, so the clustering signal never separated → the parser
// couldn't tell "my"/"is" from content → field-abstain. THIS warms with the TIER-1 PREBAKE (the curriculum built
// to warm exactly that signal, proven to separate glue 0.21 vs content 0.40 at scale), then teaches + recalls a
// name THROUGH PredictAsync — the real REPL path — in the PRODUCTION config (WithProductionMechanisms =
// LearnedCuesOnly, no hardcoded word list anywhere in the dispatch). It prints (1) whether the signal actually
// warmed (is/my/the read function-like; a real content word does not) and (2) the recall verdict + DecisionPath,
// so the truth is visible either way. If the signal warmed AND recall works, the whole arc closes.
public sealed class MergeGymFactMemory
{
    private readonly ITestOutputHelper _out;
    public MergeGymFactMemory(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task DoesItRememberTheName_AfterPrebakeWarm()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-mergegym-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // TIER 1 — warm the LEARNED function-word signal. This is the prerequisite the whole thing gates on.
            // Default generous (the signal needs a populous space to separate — Active~2.8k worked, ~600 did not).
            var curriculum = new PrebakeLanguageCurriculum(trainPerCycle: 128, seed: 7);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = false, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("MERGEGYM_SECONDS"), out var ss) ? ss : 420.0;
            _out.WriteLine($"== warming the function-word signal via TIER-1 PREBAKE ({seconds}s) ==");
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            // (1) DID THE SIGNAL WARM? glue should read function-like; a real (untrained) content word should not.
            var ds = (DialecticalSpace)rt.State.Memory;
            var (_, mean, std, thresh, floor, active, _) = ds.FunctionStats("the");
            _out.WriteLine($"\n-- function-word signal --  Active={active}  mean={mean:F3} std={std:F3}  fn-like if coh<={thresh:F3} (and <={floor:F2})");
            foreach (var g in new[] { "the", "is", "my", "of", "a", "and" })
            { var s = ds.FunctionStats(g); _out.WriteLine($"   GLUE    {g,-6} coh={s.Centrality,6:F3} deg={s.MinWarm,4}  fn?={ds.IsFunctionLike(g)}"); }
            foreach (var c in new[] { "name", "color", "sam", "blue" })
            { var s = ds.FunctionStats(c); _out.WriteLine($"   CONTENT {c,-6} coh={s.Centrality,6:F3} deg={s.MinWarm,4}  fn?={ds.IsFunctionLike(c)}"); }

            // (2) TEACH + RECALL through the production PredictAsync path — the real REPL path a person hits.
            async Task<(string Out, string Path)> P(string s)
            {
                var r = (await rt.PredictAsync(s, 12)).Result;
                var o = r?.Output?.Trim() ?? ""; var path = r?.DecisionPath ?? "";
                _out.WriteLine($"   '{s}' -> '{o}' [{path}]");
                return (o, path);
            }

            _out.WriteLine($"\n== teach + recall in the PRODUCTION runtime ==");
            await P("my name is sam");
            var name = await P("what is my name");
            await P("my favorite color is blue");
            var color = await P("what is my favorite color");

            _out.WriteLine($"\nVERDICT: name='{name.Out}' [{name.Path}]  color='{color.Out}' [{color.Path}]");
            // Honest check — does it actually remember after the prebake warmed the signal it depends on?
            Assert.True(name.Out.Contains("sam", StringComparison.OrdinalIgnoreCase),
                $"PRODUCTION name recall after prebake warm: got '{name.Out}' [{name.Path}] — if the signal warmed (above) but this abstained, the gap is in the parse/recall route, not the signal");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
