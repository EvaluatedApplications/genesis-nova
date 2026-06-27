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

// SMOKE for TIER 1 PREBAKE: does the function-word curriculum actually WARM the learned filler signal? Train it briefly,
// then check DialecticalSpace.IsFunctionLike — the GLUE words (the/is/my/of…) should read function-like (clouds collapsed
// toward the centroid) while CONTENT (cat/apple/zibble…) does not. This validates the prerequisite warms BEFORE a long
// run — the recognition is 100% learned (centrality), no dispatch list. Reports rates so we iterate from data.
public sealed class PrebakeFunctionWordsSmoke
{
    private readonly ITestOutputHelper _out;
    public PrebakeFunctionWordsSmoke(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task PrebakeWarmsTheFunctionWordSignal()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-prebake-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);
            var curriculum = new PrebakeLanguageCurriculum(trainPerCycle: 128, seed: 7);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = false, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("PREBAKE_SECONDS"), out var ss) ? ss : 150.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            var ds = (DialecticalSpace)rt.State.Memory;
            var glue = new[] { "the", "of", "is", "my", "in", "and", "with", "from" };
            var content = curriculum.SampleContent(8).ToArray(); // the ACTUAL trained (procedural) content words

            var (_, mean, std, thresh, floor, active, _) = ds.FunctionStats("the");
            _out.WriteLine($"== PREBAKE neighbour-coherence (b) ({seconds}s) ==  Active={active}  mean={mean:F3} std={std:F3}  function-like if coh≤{thresh:F3} (and ≤{floor:F2})");
            foreach (var g in glue) { var s = ds.FunctionStats(g); _out.WriteLine($"  GLUE    {g,-8} coh={s.Centrality,6:F3} deg={s.MinWarm,4}  fn?={ds.IsFunctionLike(g)}"); }
            foreach (var c in content) { var s = ds.FunctionStats(c); _out.WriteLine($"  CONTENT {c,-8} coh={s.Centrality,6:F3} deg={s.MinWarm,4}  fn?={ds.IsFunctionLike(c)}"); }
            var glueFn = glue.Count(g => ds.IsFunctionLike(g));
            var contentFn = content.Count(c => ds.IsFunctionLike(c));
            var glueAvg = glue.Select(g => ds.FunctionStats(g).Centrality).Average();
            var contentAvg = content.Select(c => ds.FunctionStats(c).Centrality).Average();
            _out.WriteLine($"  RESULT: glue function-like {glueFn}/{glue.Length}  content {contentFn}/{content.Length}  |  glue-coh {glueAvg:F3} vs content-coh {contentAvg:F3} (want glue LOWER)");
            Assert.True(glueFn >= glue.Length / 2, $"(b): a majority of function words should read function-like (got {glueFn}/{glue.Length})");
            Assert.True(contentFn <= content.Length / 3, $"content must mostly NOT be function-like (got {contentFn}/{content.Length})");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
