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

// DIAGNOSTIC (not a gate): measure held-out per-role generalisation the way PRODUCTION now trains it — GRAMMAR FIRST
// (bootstrap-first ordering), realistic interference from a few gym muscles. The open question (foundation for speaking):
// does value->VALUE generalise, or collapse (the prior 0/8)? Writes rates to value-diagnostic.txt so the numbers survive
// regardless of pass/fail or logger verbosity. Asserts nothing fatal — it REPORTS.
public sealed class GrammarValueDiagnostic
{
    private readonly ITestOutputHelper _out;
    public GrammarValueDiagnostic(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task ValueRole_HeldOut_Generalisation_ProductionFaithful()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-valdiag-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // GRAMMAR FIRST (matches the production bootstrap-first ordering), then the gym muscles that interfere.
            // VALDIAG_ISOLATE=1 → grammar ALONE (no gym muscles): the control that separates interference/starvation
            // from a head-capability/regression problem.
            var isolate = Environment.GetEnvironmentVariable("VALDIAG_ISOLATE") == "1";
            var children = new List<ITrainingCurriculum> { new GrammarCurriculum(trainPerCycle: 128) };
            if (!isolate)
                foreach (var s in new[] { GymSkill.Add, GymSkill.Synonym, GymSkill.Category })
                    children.Add(new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 });
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 4);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("VALDIAG_SECONDS"), out var ss) ? ss : 420.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            int RoleAt(string frame, string tok)
                => rt.ProbeRoles(frame).Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);

            // Held-out (never-trained) nonce tokens in each slot — a RATE.
            var copulas = new[] { "vumple", "zib", "kront", "blarg", "frot", "splim", "wozz", "glark", "tresk", "plod" };
            var nouns = new[] { "blixnar", "gizmo", "wodget", "plonk", "trizzle", "quomp", "snarf", "vlim" };
            var values = new[] { "zorptron", "quxil", "fnord", "blivet", "zarn", "morblo", "drav", "skell" };

            var copNone = copulas.Count(c => RoleAt($"my name {c} sam", c) == 0);
            var nounSubj = nouns.Count(n => RoleAt($"my {n} is sam", n) == 1);
            var valVal = values.Count(v => RoleAt($"my name is {v}", v) == 2);
            // What does VALUE collapse INTO when it fails? (SUBJECT=1 confusion is the expected failure mode.)
            var valAs = values.ToDictionary(v => v, v => RoleAt($"my name is {v}", v));

            // SAFETY (the inference-side face of the regression + the idea-#3 value-slot rule): a stated FACT must NOT be
            // read as retrieval; a markerless category frame must be. These hold IFF the role head reliably tags VALUE —
            // the whole point of prebaking the role head.
            bool Q(string s) => rt.State.Inference.IsQueryOrRetrievalForTests(s);
            var factNotRetrieval = !Q("my name is stephen");   // want TRUE (fact, value stated -> not retrieval)
            var catIsRetrieval = Q("apple is a kind of");      // want TRUE (no value -> category query)
            var synIsRetrieval = Q("a synonym for big");       // want TRUE (no value -> retrieval)

            var report =
                $"PRODUCTION-FAITHFUL (grammar-first, {seconds}s):\n" +
                $"  copula->NONE   : {copNone}/{copulas.Length}\n" +
                $"  noun->SUBJECT  : {nounSubj}/{nouns.Length}\n" +
                $"  value->VALUE   : {valVal}/{values.Length}\n" +
                $"  SAFETY 'my name is stephen' NOT retrieval : {factNotRetrieval}  (want True)\n" +
                $"  'apple is a kind of' IS retrieval         : {catIsRetrieval}  (want True)\n" +
                $"  'a synonym for big' IS retrieval          : {synIsRetrieval}  (want True)\n" +
                "  value failures (token->roleId 0=NONE 1=SUBJECT 2=VALUE 3=QUERY):\n" +
                string.Join("\n", valAs.Select(kv => $"    {kv.Key} -> {kv.Value}"));
            _out.WriteLine(report);
            File.WriteAllText(@"C:\Users\dongy\genesis-nova\value-diagnostic.txt", report);

            Assert.True(true); // diagnostic — always passes; the numbers are the deliverable
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
