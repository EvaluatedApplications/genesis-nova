using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// THE WHOLE STACK AT ONCE (user: "turn that shit on and test it all together"): gym skills + the NN grammar
// recogniser + the rude persona, trained together exactly like the app, then probe EVERY capability. The guard:
// turning grammar on must NOT break the robust core (arithmetic via the homomorphism, the in-character persona).
// Everything else (synonym/category/number-word/name-memory/roles) is reported so we see the real combined behaviour.
public sealed class EverythingTogetherTests
{
    private readonly ITestOutputHelper _out;
    public EverythingTogetherTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task EverythingTogether_Gym_Persona_Grammar()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-all-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // PERSONA: seeded reply chunks + talk route on (as the app does).
            var persona = new PersonalityCurriculum();
            rt.SeedConversationalChunks(persona.Repertoire);
            rt.SetConversationalMode(true);

            // CURRICULUM: the full gym mix + the grammar recogniser, focused with rehearsal (as the app does).
            var gymSkills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply,
                                    GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord, GymSkill.Predicate, GymSkill.WordedAdd };
            var children = gymSkills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new GrammarCurriculum(trainPerCycle: 96)).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 8);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("ALL_SECONDS"), out var ss) ? ss : 320.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 8)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r?.DecisionPath}]"); return o; }
            bool Eq(string got, string want) => AnswerEquivalence.Equivalent(got, want);
            var personaReplies = new HashSet<string>(persona.Repertoire.Select(p => p.Reply), StringComparer.OrdinalIgnoreCase);
            // In character = a known persona reply OR any rude marker (the persona's own grading set).
            bool InCharacter(string s) => personaReplies.Contains(s) || PersonalityCurriculum.RudeMarkers.Any(m => s.Contains(m, StringComparison.OrdinalIgnoreCase));

            _out.WriteLine("── ARITHMETIC (homomorphism — must hold with everything on) ──");
            var add = await P("7 + 5"); var sub = await P("9 - 4"); var mul = await P("6 * 3"); var wadd = await P("what is 8 plus 2");
            _out.WriteLine("── RETRIEVAL ──");
            var nw = await P("ten"); var syn = await P("a synonym for big"); var cat = await P("what kind of thing is apple");
            _out.WriteLine("── PERSONA (rude, in-character) ──");
            var greet = await P("hello"); var thanks = await P("thanks"); var help = await P("can you help me");
            _out.WriteLine("── NAME MEMORY (NN-parsed grammar) ──");
            await P("my name is stephen"); var mine = await P("what is my name");
            await P("your name ploo zorptron"); var nonce = await P("what is your name");
            _out.WriteLine("── NN GRAMMAR ROLES ──");
            string[] RN = { "NONE", "SUBJECT", "VALUE", "QUERY" };
            _out.WriteLine("  whats your name -> " + string.Join("  ", rt.ProbeRoles("whats your name").Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));

            _out.WriteLine($"\nSUMMARY add={Eq(add,"12")} sub={Eq(sub,"5")} mul={Eq(mul,"18")} wadd={Eq(wadd,"10")} nw={Eq(nw,"10")} syn='{syn}' cat='{cat}' greet_ok={InCharacter(greet)} thanks_ok={InCharacter(thanks)} name='{mine}' nonce='{nonce}'");

            // GUARD the robust core survives training everything together:
            var arith = new[] { Eq(add, "12"), Eq(sub, "5"), Eq(mul, "18") }.Count(b => b);
            Assert.True(arith >= 2, $"arithmetic (homomorphism) must hold with grammar+persona on; {arith}/3");
            Assert.True(InCharacter(greet) || InCharacter(thanks) || InCharacter(help), "the persona must stay in character with everything on");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
