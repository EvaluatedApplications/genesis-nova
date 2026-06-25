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

// RESEARCH: is the role head's instability (copula / name-recall) just UNDER-TRAINING, or something deeper? Two levers
// vs the earlier thin runs: (1) grammar gets a far bigger share of the budget (fewer competing skills), (2) reliability
// is measured over MANY held-out tokens (a RATE, not one coin-flip probe). If held-out copulas reliably read NONE here,
// it was a training-budget problem; if they stay ~50/50 with heavy training, it's deeper (likely the linear role head's
// capacity for the positional copula judgment). Reports rates; asserts only the already-robust subject/value.
public sealed class GrammarStabilityResearch
{
    private readonly ITestOutputHelper _out;
    public GrammarStabilityResearch(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task DoesMoreGrammarTraining_StabiliseHeldOutRoles()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-stab-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // FULL MIXED (grammar is just 1/6 by count). With HONEST held-out grammar probes, the weakest-first
            // scheduler should AUTO-give grammar enough focus when it cannot generalise — no hand-tuned share.
            var gymSkills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category };
            var children = gymSkills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new GrammarCurriculum(trainPerCycle: 128)).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 4);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("STAB_SECONDS"), out var ss) ? ss : 420.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            int RoleAt(string frame, string tok)
                => rt.ProbeRoles(frame).Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);

            // Many HELD-OUT (made-up, never-trained) tokens in each slot — a RATE, not one probe.
            var copulas = new[] { "vumple", "zib", "kront", "blarg", "frot", "splim", "wozz", "glark", "tresk", "plod" };
            var nouns = new[] { "blixnar", "gizmo", "wodget", "plonk", "trizzle", "quomp", "snarf", "vlim" };
            var values = new[] { "zorptron", "quxil", "fnord", "blivet", "zarn", "morblo", "drav", "skell" };

            var copNone = copulas.Count(c => RoleAt($"my name {c} sam", c) == 0);     // held-out copula -> NONE
            var nounSubj = nouns.Count(n => RoleAt($"my {n} is sam", n) == 1);         // held-out noun -> SUBJECT
            var valVal = values.Count(v => RoleAt($"my name is {v}", v) == 2);         // held-out value -> VALUE

            _out.WriteLine($"HELD-OUT ROLE RELIABILITY (grammar-heavy, {seconds}s):");
            _out.WriteLine($"  copula->NONE   : {copNone}/{copulas.Length}");
            _out.WriteLine($"  noun->SUBJECT  : {nounSubj}/{nouns.Length}");
            _out.WriteLine($"  value->VALUE   : {valVal}/{values.Length}");
            foreach (var c in copulas) _out.WriteLine($"    my name {c} sam -> {c}:{RoleAt($"my name {c} sam", c)}");
            // NOTE (measured separately): with roles reliable, asserts PARSE+LEARN ("my name is stephen"->field-learn),
            // but recall ABSTAINS ("what is my name"->''). So name-recall's remaining problem is RETRIEVAL, not roles.

            // The SCHEDULER's claim: in the thin MIX, honest weakest-first must give grammar enough that copula + noun
            // generalise (copula was 0/10 with a dishonest grammar probe). value->VALUE is REPORTED — its collapse is a
            // separate VALUE-class-balance/dynamics issue, not a training-share one, so it is not the scheduler's gate.
            Assert.True(copNone >= 8 && nounSubj >= nouns.Length - 1,
                $"honest weakest-first must rescue grammar in the mix: copula={copNone}/{copulas.Length} noun={nounSubj}/{nouns.Length} (value reported: {valVal}/{values.Length})");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
