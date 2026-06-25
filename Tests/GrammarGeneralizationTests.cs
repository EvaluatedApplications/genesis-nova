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

// THE ADVERSARIAL / SKEPTICAL TEST (am I overfitting?). The role head is trained ONLY on the GrammarCurriculum's
// fixed vocabulary. Real generalisation = it tags tokens it NEVER SAW in grammar frames by their STRUCTURAL POSITION,
// not by token identity. So we probe held-out, made-up tokens ("blixnar"/"zorptron"/"vumple" — in no curriculum and
// no list) in subject/value/copula slots. If they get the right role, the NN learned grammar; if not, it memorised
// the curriculum and the head should be SUBTRACTED, not extended. Roles: 0=NONE 1=SUBJECT 2=VALUE 3=QUERY.
public sealed class GrammarGeneralizationTests
{
    private readonly ITestOutputHelper _out;
    public GrammarGeneralizationTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task RoleHead_GeneralisesToUnseenVocabulary_OrItIsMemorising()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-gen-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            var gymSkills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord };
            var children = gymSkills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new GrammarCurriculum(trainPerCycle: 96)).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 6);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("GEN_SECONDS"), out var ss) ? ss : 260.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            string[] RN = { "NONE", "SUBJECT", "VALUE", "QUERY" };
            int RoleOf(IReadOnlyList<(string Token, int Role, double Confidence)> tags, string tok)
                => tags.Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);
            void Dump(string s) => _out.WriteLine($"  '{s}' -> " + string.Join("  ", rt.ProbeRoles(s).Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}({t.Confidence:F2})")));

            // SANITY: trained vocabulary (should already work).
            _out.WriteLine("── trained vocab (sanity) ──");
            Dump("my name is sam");
            Dump("whats my name");

            // GENERALISATION: held-out, never-seen tokens in grammar slots. "blixnar"/"zorptron"/"vumple" appear in NO
            // curriculum and NO list — only their STRUCTURAL POSITION can drive the role.
            _out.WriteLine("── HELD-OUT vocab (the real test) ──");
            var assertHO = rt.ProbeRoles("my blixnar is zorptron");   // unseen noun + unseen value, trained copula "is"
            var queryHO = rt.ProbeRoles("whats my blixnar");          // unseen noun in a query
            var copulaHO = rt.ProbeRoles("my name vumple sam");       // unseen COPULA between trained subject + value
            Dump("my blixnar is zorptron");
            Dump("whats my blixnar");
            Dump("my name vumple sam");

            var subjOK = RoleOf(assertHO, "blixnar") == 1;   // unseen noun -> SUBJECT by structure
            var valOK = RoleOf(assertHO, "zorptron") == 2;   // unseen value -> VALUE by structure
            var qSubjOK = RoleOf(queryHO, "blixnar") == 1;   // unseen noun -> SUBJECT in a query
            var copOK = RoleOf(copulaHO, "vumple") == 0;     // unseen copula -> NONE (the hardest)
            _out.WriteLine($"\nGENERALISES: heldout-subject={subjOK} heldout-value={valOK} heldout-querysubject={qSubjOK} heldout-copula={copOK}");

            // The verdict. Require the CONTENT roles to transfer to unseen vocabulary (subject + value), which is the
            // meaningful claim ("my <newthing> is <newvalue>" parses). If this fails, the head is memorising — cut it.
            Assert.True(subjOK && valOK,
                $"role head must generalise to UNSEEN vocab by structure (else it's memorising): subj={subjOK} val={valOK}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
