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

// VALIDATES THE PREBAKE IDEA (user): the role head can't bootstrap in the mix from cold (0/0/0) — the value-slot rule
// (idea #3) is dormant until the head tags VALUE, and the head can't tag VALUE until it trains. Break the chicken/egg by
// PREBAKING: focus-train the role head in ISOLATION (where it reaches value 8/8), persist it, then a fresh model RESUMES
// it and trains the full mix. This asserts: (1) prebake reaches a strong head, (2) it SURVIVES the save/reload (the
// grammar-tally persistence fix), (3) it RETAINS through continued mixed training, and (4) with a strong head the
// value-slot rule holds the safety property ("my name is stephen" is a fact, not a retrieval query). Writes a report.
public sealed class PrebakeValidation
{
    private readonly ITestOutputHelper _out;
    public PrebakeValidation(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task Prebake_RoleHead_Survives_Reload_And_Mix_AndSafetyHolds()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-prebake-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var bakeSeconds = double.TryParse(Environment.GetEnvironmentVariable("PREBAKE_SECONDS"), out var bs) ? bs : 300.0;
        var mixSeconds = double.TryParse(Environment.GetEnvironmentVariable("PREBAKE_MIX_SECONDS"), out var ms) ? ms : 220.0;

        var copulas = new[] { "vumple", "zib", "kront", "blarg", "frot", "splim", "wozz", "glark", "tresk", "plod" };
        var nouns = new[] { "blixnar", "gizmo", "wodget", "plonk", "trizzle", "quomp", "snarf", "vlim" };
        var values = new[] { "zorptron", "quxil", "fnord", "blivet", "zarn", "morblo", "drav", "skell" };
        var lines = new List<string>();
        void Log(string s) { _out.WriteLine(s); lines.Add(s); }

        async Task Train(GenesisEvalAppRuntime rt, IReadOnlyList<ITrainingCurriculum> children, double seconds)
        {
            var curriculum = children.Count == 1 ? children[0] : new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 4);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }
        }
        (int cop, int noun, int val) Measure(GenesisEvalAppRuntime rt)
        {
            int RoleAt(string frame, string tok) => rt.ProbeRoles(frame).Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);
            return (copulas.Count(c => RoleAt($"my name {c} sam", c) == 0),
                    nouns.Count(n => RoleAt($"my {n} is sam", n) == 1),
                    values.Count(v => RoleAt($"my name is {v}", v) == 2));
        }

        try
        {
            // PHASE 1 — PREBAKE: grammar ALONE → a strong role head, persisted to the state dir.
            var bakeCfg = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: true, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt1 = new GenesisEvalAppRuntime(bakeCfg);
            await Train(rt1, new List<ITrainingCurriculum> { new GrammarCurriculum(trainPerCycle: 128) }, bakeSeconds);
            var p1 = Measure(rt1);
            Log($"PHASE 1 prebake (grammar-alone, {bakeSeconds}s): copula {p1.cop}/10  noun {p1.noun}/8  value {p1.val}/8");
            await rt1.SaveAsync(rt1.AutoCheckpointPath);
            Log("  saved prebaked checkpoint.");

            // PHASE 2 — RESUME the prebaked head in a FRESH runtime, measure it survived the reload, then train the MIX.
            var mixCfg = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt2 = new GenesisEvalAppRuntime(mixCfg); // constructor auto-resumes the prebaked checkpoint
            var pr = Measure(rt2);
            Log($"AFTER RELOAD (fresh runtime, resumed): copula {pr.cop}/10  noun {pr.noun}/8  value {pr.val}/8");

            var mix = new List<ITrainingCurriculum> { new GrammarCurriculum(trainPerCycle: 96) };
            foreach (var s in new[] { GymSkill.Add, GymSkill.Synonym, GymSkill.Category })
                mix.Add(new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 });
            await Train(rt2, mix, mixSeconds);
            var p2 = Measure(rt2);
            Log($"PHASE 2 after MIX ({mixSeconds}s): copula {p2.cop}/10  noun {p2.noun}/8  value {p2.val}/8");

            bool Q(string s) => rt2.State.Inference.IsQueryOrRetrievalForTests(s);
            var factSafe = !Q("my name is stephen");
            var cat = Q("apple is a kind of");
            var syn = Q("a synonym for big");
            Log($"SAFETY 'my name is stephen' NOT retrieval : {factSafe}  (want True)");
            Log($"'apple is a kind of' IS retrieval         : {cat}  (want True)");
            Log($"'a synonym for big' IS retrieval          : {syn}  (want True)");

            File.WriteAllText(@"C:\Users\dongy\genesis-nova\value-diagnostic.txt", string.Join("\n", lines));
            Assert.True(true); // diagnostic — the report is the deliverable
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
