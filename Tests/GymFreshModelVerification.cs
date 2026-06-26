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

// END-TO-END: train a FRESH model through the PRODUCTION gym path with the de-hardcoded dispatch LIVE
// (WithProductionMechanisms → DeHardcodedDispatch → codec+lists OFF), then verify the de-hardcoded capabilities work
// having been LEARNED from zero — number-words (lexicon), worded op-cues, compare cue+output, alongside the
// homomorphic arithmetic. This is the "does the whole thing actually train and work" check, no warming shortcuts.
public sealed class GymFreshModelVerification
{
    private readonly ITestOutputHelper _out;
    public GymFreshModelVerification(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task FreshModel_GymTrains_DeHardcodedCapabilities()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-fresh-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // FRESH model (AutoResume off, fresh dir), PRODUCTION mechanisms → de-hardcoded dispatch is LIVE.
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // The production gym de-hardcoding block: gym skills + op-cues + number-words + grammar (FocusedCurriculum,
            // weakest-first — the empty lexicon/cues start WRONG so they get prioritised and bootstrap).
            var children = new List<ITrainingCurriculum>();
            foreach (var s in new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category, GymSkill.Predicate, GymSkill.WordedAdd })
                children.Add(new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 });
            children.Add(new OpCueCurriculum(trainPerCycle: 64));
            children.Add(new NumberWordCurriculum(trainPerCycle: 64));
            children.Add(new GrammarCurriculum(trainPerCycle: 64));
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 5);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("FRESH_SECONDS"), out var ss) ? ss : 480.0; // 300 = too short for digit->word ∘tow; 480 reliably bootstraps the learned paths
            _out.WriteLine($"training a FRESH model for {seconds}s with de-hardcoded dispatch LIVE...");
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            _out.WriteLine($"learned: {rt.State.Inference.LearnedNumberWordAtomCount} number-word atoms");
            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 12)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r?.DecisionPath}]"); return o; }
            bool Eq(string g, string w) => AnswerEquivalence.Equivalent(g, w);

            _out.WriteLine("== de-hardcoded capabilities, LEARNED from a fresh model (codec+lists off) ==");
            var add = await P("3 + 4");                  // homomorphism (always)
            var prod = await P("the product of 2 and 5"); // LEARNED op-cue
            var sum = await P("the sum of 6 and 1");       // LEARNED op-cue
            var nw = await P("7 in words");                // LEARNED number-word lexicon
            var nw2 = await P("12 in words");
            var wd = await P("nineteen as a number");      // LEARNED number-word (word->digit)
            var cmp = await P("8 compared to 3");          // LEARNED compare cue + output word
            var cmp2 = await P("2 compared to 9");

            var pass = new[] {
                ("3 + 4 = 7", Eq(add, "7")),
                ("product 2,5 = 10", Eq(prod, "10")),
                ("sum 6,1 = 7", Eq(sum, "7")),
                ("7 in words = seven", Eq(nw, "seven")),
                ("12 in words = twelve", Eq(nw2, "twelve")),
                ("nineteen = 19", Eq(wd, "19")),
                ("8>3 = greater", cmp.Contains("greater")),
                ("2<9 = less", cmp2.Contains("less")),
            };
            foreach (var (label, ok) in pass) _out.WriteLine($"  [{(ok ? "OK" : "MISS")}] {label}");
            var hits = pass.Count(p => p.Item2);
            _out.WriteLine($"\nDE-HARDCODED CAPABILITIES LEARNED FROM A FRESH GYM: {hits}/{pass.Length}");

            // The homomorphism is guaranteed; the rest must have been LEARNED (codec/lists are off). Require the learned
            // paths to have genuinely bootstrapped — a strong majority, allowing for a short-run tail.
            Assert.True(Eq(add, "7"), "arithmetic homomorphism");
            Assert.True(rt.State.Inference.LearnedNumberWordAtomCount >= 20, $"number-word lexicon bootstrapped from the gym ({rt.State.Inference.LearnedNumberWordAtomCount} atoms)");
            Assert.True(hits >= 6, $"the de-hardcoded LEARNED capabilities work after a fresh gym run ({hits}/{pass.Length})");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
