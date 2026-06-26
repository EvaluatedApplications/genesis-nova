using System;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Runtime;
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
}
