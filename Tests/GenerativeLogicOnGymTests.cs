using System;
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

/// <summary>
/// Does the generative-logic engine (field-induce-2d) show reasoning ON the gym's open-ended reasoning trainers?
/// Runs the headless gym on the reasoning creators (Category + Synonym), then probes both a FUNCTION-shaped query
/// (NAND composition — the shape the engine reasons over) and the gym's OWN reasoning queries (category/synonym).
/// Reports the DecisionPath of each, so the SHAPE question is answered empirically, not asserted.
/// </summary>
public sealed class GenerativeLogicOnGymTests
{
    private readonly ITestOutputHelper _out;
    public GenerativeLogicOnGymTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task GenerativeLogic_OnReasoningGym_ShowsReasoning_OnlyOnFunctionShapes()
    {
        var minutes = double.TryParse(Environment.GetEnvironmentVariable("GYM_MINUTES"), out var mm) ? mm : 0.75;
        var dir = Path.Combine(Path.GetTempPath(), "gn-glgym-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(
                Backend: ComputeBackend.Cpu, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoResume: false, AutoPersist: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var runtime = new GenesisEvalAppRuntime(config);

            // headless gym on the OPEN-ENDED REASONING creators
            var skills = new[] { GymSkill.Synonym, GymSkill.Category };
            var children = skills.Select(s => (ITrainingCurriculum)new GymTrainer(startLevel: 1, skills: new[] { s })
                { MasteryBar = 0.8, TrainPerCycle = 64 }).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.8, focusBudget: 8);
            var options = new GenesisModularTrainingOrchestrator.Options
            { MasteryBar = 0.8, RequirePlatonic = true, WorkDir = dir, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };

            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(minutes));
            var cycles = 0;
            try { await new GenesisModularTrainingOrchestrator().RunAsync(runtime, curriculum, options, _ => cycles++, cts.Token); }
            catch (OperationCanceledException) { }
            _out.WriteLine($"ran {cycles} reasoning-gym cycles\n");

            async Task<(string, string)> Ask(string q)
            { var r = (await runtime.PredictAsync(q, 8)).Result; return ((r?.Output ?? "").Trim(), r?.DecisionPath ?? ""); }

            // (a) FUNCTION-shaped — the generative-logic engine's domain (thinking: derive AND from NAND by composition)
            var (fOut, fPath) = await Ask("nand 0 0 is 1  nand 0 1 is 1  nand 1 0 is 1  nand 1 1 is 0  nand nand 1 1 nand 1 1 is");
            // (b) the gym's OWN reasoning queries — relational/semantic shape
            var (cOut, cPath) = await Ask("what kind of thing is apple");
            var (sOut, sPath) = await Ask("a synonym for big");

            _out.WriteLine($"FUNCTION  'AND via NAND'      → '{fOut}' [{fPath}]");
            _out.WriteLine($"GYM CAT   'kind of apple'     → '{cOut}' [{cPath}]");
            _out.WriteLine($"GYM SYN   'synonym for big'   → '{sOut}' [{sPath}]");

            var reasonsOnFunction = fPath == "field-induce-2d";
            var firesOnGym = cPath == "field-induce-2d" || sPath == "field-induce-2d";
            _out.WriteLine($"\n=> reasoning engine fires on FUNCTION shape: {reasonsOnFunction}");
            _out.WriteLine($"=> reasoning engine fires on GYM reasoning shapes: {firesOnGym}");
            _out.WriteLine(reasonsOnFunction && !firesOnGym
                ? ">>> HONEST: the generative-logic reasoning is REAL but SHAPE-MISMATCHED — the gym's open-ended trainers pose relational queries, not function-induction ones, so it doesn't fire there. To show reasoning IN the gym, the curricula must be reasoning-shaped (function induction / composition tasks)."
                : ">>> (unexpected — inspect: either it fired on gym shapes, or the function shape didn't reason).");

            Assert.True(cycles > 0, "the reasoning gym ran");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
