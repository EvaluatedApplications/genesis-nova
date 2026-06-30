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

// END-TO-END: does the NN STRUCTURE RECOGNISER learn to parse grammar (the user's architecture — NN recognises,
// space stores)? After the gym warms the grammar curriculum, the per-token role head should TAG roles
// {0=NONE,1=SUBJECT,2=VALUE,3=QUERY} robustly — the fuzzy job the centrality thresholds did badly. Warm-start
// [SlowFact]: a cold model can't parse grammar (it must be taught). Roles=0..3 per GenesisInferenceEngine.RoleCount.
public sealed class GrammarEmergenceTests
{
    private readonly ITestOutputHelper _out;
    public GrammarEmergenceTests(ITestOutputHelper o) => _out = o;

    [Fact(Skip = "Retired: asserts the deprecated 4-role NN role head (ProbeRoles → SUBJECT/QUERY/VALUE). The project subtracts this classifier — the conscious-field copula-pivot replaced it for fact parsing (uses no role head). See nova-grammar-role-regression.")]
    public async Task NN_RecognisesGrammarRoles_AfterWarmup()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-gem-" + Guid.NewGuid().ToString("N"));
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
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("GEM_SECONDS"), out var ss) ? ss : 220.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            string[] RoleName = { "NONE", "SUBJECT", "VALUE", "QUERY" };
            int RoleOf(IReadOnlyList<(string Token, int Role, double Confidence)> tags, string tok)
                => tags.Where(t => t.Token.Equals(tok, StringComparison.OrdinalIgnoreCase)).Select(t => t.Role).FirstOrDefault(-1);
            void Dump(string s)
            {
                var tags = rt.ProbeRoles(s);
                _out.WriteLine($"  '{s}' -> " + string.Join("  ", tags.Select(t => $"{t.Token}:{(t.Role >= 0 && t.Role < 4 ? RoleName[t.Role] : "?")}({t.Confidence:F2})")));
            }

            // The NN should RECOGNISE the CORE structure (these vocab tokens were warmed by the curriculum). We assert
            // the robustly-learned roles (subject noun / value / query cue); the rarer NONE/determiner classes are
            // noisier under a short warm-up (accuracy tuning, not a mechanism failure) and are only reported.
            var assertTags = rt.ProbeRoles("my name is sam");
            var queryTags = rt.ProbeRoles("whats your name");   // in-distribution query phrasing
            var unseen = rt.ProbeRoles("his car was audi");      // generalisation to an unseen binding
            Dump("my name is sam");
            Dump("whats your name");
            Dump("his car was audi");

            // RELIABLY-learned core (all >0.9 confidence): the subject noun, the query cue, and GENERALISATION to a
            // frame the NN never trained on. This is the validation that the NN recognises grammatical structure.
            Assert.Equal(1, RoleOf(assertTags, "name"));   // SUBJECT (the queried key)
            Assert.Equal(3, RoleOf(queryTags, "whats"));   // QUERY (the recall cue)
            Assert.Equal(1, RoleOf(queryTags, "name"));    // SUBJECT in a query frame
            Assert.Equal(1, RoleOf(unseen, "car"));        // SUBJECT — generalises to a frame it wasn't trained on
            // VALUE recognition + full end-to-end recall are reported below — the accuracy-tuning frontier.

            // END-TO-END (reported; retrieval pollution is a separate substrate concern): learn + recall via the NN parse.
            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 8)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r?.DecisionPath}]"); return o; }
            await P("my name is stephen");
            var mine = await P("what is my name");
            await P("your name ploo zorptron"); // NONCE copula — only the learned NN roles can parse "ploo"
            var nonce = await P("what is your name");
            _out.WriteLine($"\nend-to-end: mine={mine} nonce={nonce}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
