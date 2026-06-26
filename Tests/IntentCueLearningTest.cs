using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// De-hardcoding #3/#4: the IsCompareCue / IsToWordCue / IsToDigitCue word-lists become LEARNED intent cues (cue→intent
// "∘" anchor, same mechanism as op-cues). Feed the intents by STRUCTURE, flip the hardcoded lists OFF (LearnedCuesOnly),
// and confirm compare / to-word / to-digit still route — purely from what was learned.
public sealed class IntentCueLearningTest
{
    private readonly ITestOutputHelper _out;
    public IntentCueLearningTest(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void LearnedIntentCues_Route_WithoutHardcodedLists()
    {
        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: 128, Seed: 7).WithProductionMechanisms());
        var eng = nova.Inference;
        string GT(long v) => NumberWordVocabulary.ToWords(v);

        // 1. number-word atoms so LearnIntentCue can TYPE number-word outputs (and the routes can answer).
        for (long v = 0; v <= 20; v++) eng.LearnNumberWord($"{v} in words", GT(v));
        // 2. intent observations (varied frames so each cue word is reinforced; framing spread → competing → abstain).
        foreach (var v in new long[] { 5, 12, 7, 3, 15, 9, 18 })
        {
            eng.LearnIntentCue($"{v} in words", GT(v));                 // ToWord cue: "in"/"words"
            eng.LearnIntentCue($"spell out {v}", GT(v));                // ToWord cue: "spell"/"out"
            eng.LearnIntentCue($"{GT(v)} as a number", v.ToString());  // ToDigit cue: "as"/"a"/"number"
            eng.LearnIntentCue($"{GT(v)} as a numeral", v.ToString());  // ToDigit cue: "numeral"
        }
        foreach (var (a, b) in new[] { (5, 3), (2, 8), (4, 4), (9, 1), (3, 7), (6, 6) })
            eng.LearnIntentCue($"{a} compared to {b}", a > b ? "greater" : a < b ? "less" : "equal"); // Compare cue: "compared"/"to"

        eng.LearnedCuesOnly = true;       // de-hardcoded: no IsToWordCue/IsToDigitCue/IsCompareCue lists
        eng.LearnedNumberWordsOnly = true; // and no codec — the value mapping is the learned lexicon

        string P(string s) { var r = eng.Generate(new GenerationRequest(s, 12)); var o = r.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r.DecisionPath}]"); return o; }

        Assert.Equal("seven", P("7 in words"));               // ToWord cue learned + lexicon spells it
        Assert.Equal("eleven", P("11 in words"));
        Assert.Equal("13", P("thirteen as a number"));        // ToDigit cue learned + lexicon parses it
        Assert.Equal("8", P("eight as a numeral"));
        Assert.Contains("greater", P("8 compared to 3"));     // Compare cue learned (route fires; output word is the glider's)
        Assert.Contains("less", P("2 compared to 9"));
    }

    // #2: the retrieval marker (kind/type/synonym/...) becomes a LEARNED ∘ret cue, learned from examples whose ANSWER is
    // a CATEGORY HUB (structural — a concept many things point to; genesis Law S), NOT a word-list. Train Category (forms
    // hubs + learns the cue) + Grammar/Synonym (warm the role head + the learned function-word signal so the shared
    // copula "is" is EXCLUDED from ∘ret). Then a markerless category frame is detected as retrieval; a fact is not.
    [SlowFact]
    public async Task LearnedRetrievalCue_DetectsCategory_NotFact()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-ret-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);
            var children = new[] { GymSkill.Category, GymSkill.Synonym }
                .Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new GrammarCurriculum(trainPerCycle: 64)).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 3);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("RET_SECONDS"), out var ss) ? ss : 220.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            rt.State.Inference.LearnedCuesOnly = true; // de-hardcoded: no IsRetrievalMarker / IsQuestionCue lists
            bool Q(string s) { var r = rt.State.Inference.IsQueryOrRetrievalForTests(s); _out.WriteLine($"  query/retrieval? '{s}' -> {r}"); return r; }

            // markerless category frame (no wh-word) — caught only by the LEARNED ∘ret cue, not the role head's QUERY:
            Assert.True(Q("apple is a kind of"));      // "kind" -> ∘ret (learned from category-HUB answers)
            // THE SAFETY PROPERTY: a personal FACT must NOT be gated as retrieval (the shared copula "is", tagged Filler
            // by the role head, is excluded from ∘ret — else every "X is Y" fact would be mis-read as a category query):
            Assert.False(Q("my name is stephen"));
            _out.WriteLine($"  (informational, coverage-dependent) 'rose is a sort of' -> {Q("rose is a sort of")}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
