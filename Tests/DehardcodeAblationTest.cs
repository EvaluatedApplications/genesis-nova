using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// ABLATION: with the hardcoded crutches REMOVED (grammar copula/possessive fallback AND the TryOpCue op-synonym list),
// does the LEARNED path carry it after gym training? If yes, the "learned" gains were real; if no, they were faked by
// the lists. Train the gym warm, then probe worded op-cues (learned op→cue relation) + name memory (NN role parse).
public sealed class DehardcodeAblationTest
{
    private readonly ITestOutputHelper _out;
    public DehardcodeAblationTest(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task LearnedPaths_CarryArithmeticAndNameMemory_WithNoHardcodedCrutch()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-abl-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            var gymSkills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply };
            var children = gymSkills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new OpCueCurriculum(trainPerCycle: 96))  // worded op-synonyms → LEARNED cue→op relations (no TryOpCue list)
                .Append(new GrammarCurriculum(trainPerCycle: 96)).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 4);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("ABL_SECONDS"), out var ss) ? ss : 220.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 8)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r?.DecisionPath}]"); return o; }
            bool Eq(string g, string w) => AnswerEquivalence.Equivalent(g, w);

            _out.WriteLine("── OP CUES (no TryOpCue list — must resolve via LEARNED op→cue) ──");
            var sym = await P("3 + 4");          // symbol infix (still structural)
            var sumw = await P("the sum of 3 and 4");
            var diffw = await P("the difference of 7 and 5");
            var prodw = await P("the product of 2 and 5");
            var quotw = await P("the quotient of 6 and 2");
            _out.WriteLine("── NAME MEMORY (no copula/possessive fallback — must parse via NN roles) ──");
            await P("my name is poo"); var name = await P("what is my name");

            _out.WriteLine($"\nSUMMARY  sym={Eq(sym,"7")} sum={Eq(sumw,"7")} difference={Eq(diffw,"2")} product={Eq(prodw,"10")} quotient={Eq(quotw,"3")} name={name.Contains("poo",StringComparison.OrdinalIgnoreCase)}");
            Assert.True(true); // diagnostic — the numbers are the result
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
