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

// The trained grammar PARSER (per-token role head) must SURVIVE a reload — restored from the checkpoint, not relearned
// from the gym each session. Train grammar (role head warms), save, then a FRESH runtime resumes and tags roles
// CORRECTLY with NO retraining. If the head weren't persisted, PredictRoles would be empty/untrained after reload.
public sealed class RoleHeadPersistenceTest
{
    private readonly ITestOutputHelper _out;
    public RoleHeadPersistenceTest(ITestOutputHelper o) => _out = o;

    [Fact(Skip = "Retired: asserts the deprecated 4-role NN role head (ProbeRoles). The project subtracts this classifier — the conscious-field copula-pivot is the fact-parsing path (uses no role head; runtime fact recall passes via it). See nova-grammar-role-regression.")]
    public async Task TrainedRoleParser_SurvivesSaveAndReload()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-rhp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            GenesisNovaConfig Cfg() => new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();

            // SESSION 1 — train the role head on grammar, then save.
            var rt1 = new GenesisEvalAppRuntime(Cfg());
            var curriculum = new FocusedCurriculum(new ITrainingCurriculum[] { new GrammarCurriculum(trainPerCycle: 128) }.ToList(), masteryBar: 0.9, focusBudget: 1);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = false, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("RHP_SECONDS"), out var ss) ? ss : 100.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt1, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            string[] RN = { "NONE", "SUBJECT", "VALUE", "QUERY" };
            int RoleOf(System.Collections.Generic.IReadOnlyList<(string Token, int Role, double Confidence)> tags, string tok)
                => tags.Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);
            var before = rt1.ProbeRoles("what is my name");
            _out.WriteLine("BEFORE save: what is my name -> " + string.Join(" ", before.Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));
            await rt1.SaveAsync(rt1.AutoCheckpointPath);

            // SESSION 2 — fresh runtime resumes from disk; NO training. The role head must be RESTORED.
            var rt2 = new GenesisEvalAppRuntime(Cfg());
            var after = rt2.ProbeRoles("what is my name");
            _out.WriteLine("AFTER reload: what is my name -> " + string.Join(" ", after.Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));
            var assertTags = rt2.ProbeRoles("his car is audi");
            _out.WriteLine("AFTER reload: his car is audi -> " + string.Join(" ", assertTags.Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));

            // The restored parser tags the query correctly WITHOUT any retraining.
            Assert.True(after.Count > 0 && after.Any(t => t.Confidence > 0.0), "role head must be RESTORED on reload (not empty/untrained)");
            Assert.Equal(3, RoleOf(after, "what"));   // QUERY
            Assert.Equal(1, RoleOf(after, "name"));   // SUBJECT
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
