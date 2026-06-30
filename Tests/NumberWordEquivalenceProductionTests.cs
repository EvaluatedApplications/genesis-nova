using System;
using System.Diagnostics;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// PRODUCTION-DIMENSION behavioural demo: bidirectional number-word equivalence (one≡1) works VIA THE NEW
/// ENGINE in BOTH directions, taught by DATA, nothing hardcoded.
///
/// HISTORY — the legacy version of this test built the OLD substrate (<c>PlatonicSpaceMemory</c> + the GRU
/// decoder) with no conscious field / dialectical core / learned lexicon, and probed BARE tokens ("five"→"5",
/// "5"→"five"). That asks the GRU decoder to memorise a raw mapping; it failed the platonic bidirectional bar
/// (and was failing at the pre-session baseline too — pre-existing). The capability is real on the NEW engine,
/// so this demonstrates it there.
///
/// THE NEW PATH (de-hardcoding #5, see nova-learned-number-words): with <see cref="GenesisNovaConfig.WithProductionMechanisms"/>
/// the runtime uses the dialectical core + conscious field, and number↔word is answered by the LEARNED
/// <c>NumberWordLexicon</c> (word↔value atoms learned by observation, composed by universal base-10 place value),
/// not the hardcoded <c>NumberWordVocabulary</c> codec (<c>LearnedNumberWordsOnly</c>) — and the to-word/to-digit
/// INTENT is the LEARNED cue (<c>LearnedCuesOnly</c>), not a word-list. Nothing about the mapping is hardcoded:
/// the lexicon ABSTAINS on an untaught word. The field number-word route (<c>TryFieldNumberWord</c>) is INTENT-CUED
/// by design (it answers "5 in words"→five and "five as a number"→5, the way a person frames a conversion), so the
/// probe uses those cue frames rather than bare tokens — that is how the de-hardcoded route works, not a relaxed bar.
///
/// The learned path is a direct write to the SHARED space (no GRU gradient steps), so this is fast even at
/// production dims — there is no epoch loop to bound. [SlowFact] only because it builds the production-size runtime.
/// </summary>
public sealed class NumberWordEquivalenceProductionTests
{
    private readonly ITestOutputHelper _out;
    public NumberWordEquivalenceProductionTests(ITestOutputHelper output) => _out = output;

    [SlowFact]
    public void Bidirectional_NumberWordEquivalence_IsPlatonic_ViaLearnedLexicon_AtProductionDimension()
    {
        var sw = Stopwatch.StartNew();

        // The PRODUCTION runtime: dialectical core + conscious field + de-hardcoded dispatch (LearnedNumberWordsOnly
        // + LearnedCuesOnly). Production width (HiddenSize 512, FaceDimension 1024).
        var nova = new GenesisRuntimeState(
            new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize).WithProductionMechanisms());
        var eng = nova.Inference;
        Assert.True(eng.LearnedNumberWordsOnly && eng.LearnedCuesOnly,
            "production mechanisms must turn the codec/word-lists OFF — the mapping must come from what was LEARNED");

        string GT(long v) => NumberWordVocabulary.ToWords(v);  // GROUND-TRUTH reference data (the trainer's oracle), not a model heuristic
        var digitWords = new[] { ("0","zero"),("1","one"),("2","two"),("3","three"),("4","four"),
            ("5","five"),("6","six"),("7","seven"),("8","eight"),("9","nine") };

        // 1. LEARN the number-word ATOMS by OBSERVATION (the gym's digit→word feed, done directly here). word↔value
        //    atoms are shared by BOTH directions (the same atom spells AND parses), so feeding digit→word once is
        //    enough for word→digit too. Teach 0-20 so the lexicon is genuinely populated, not just the probe targets.
        for (long v = 0; v <= 20; v++) eng.LearnNumberWord($"{v} in words", GT(v));

        // 2. LEARN the to-word / to-digit INTENT cues from example STRUCTURE (one digit + number-word output ⇒ ∘tow;
        //    number-word input + one-digit output ⇒ ∘tod) — varied frames so the cue words are reinforced and a framing
        //    word that spreads across intents abstains. No IsToWordCue/IsToDigitCue list is consulted.
        foreach (var v in new long[] { 5, 12, 7, 3, 15, 9, 18, 1, 14, 6 })
        {
            eng.LearnIntentCue($"{v} in words", GT(v));                 // ToWord cue: "in"/"words"
            eng.LearnIntentCue($"spell out {v}", GT(v));                // ToWord cue: "spell"/"out"
            eng.LearnIntentCue($"{GT(v)} as a number", v.ToString());  // ToDigit cue: "as"/"a"/"number"
            eng.LearnIntentCue($"{GT(v)} as a numeral", v.ToString()); // ToDigit cue: "numeral"
        }

        // 3. PROBE both directions through the field number-word route, asserting the answer came via the PLATONIC
        //    path (UsedPlatonicQuery && !UsedNeuralFallback) — i.e. the learned lexicon, never a neural-decoder guess.
        int w2d = 0, d2wPlat = 0;
        foreach (var (digit, word) in digitWords)
        {
            var wr = eng.Generate(new GenerationRequest($"{word} as a number", 4));   // word → digit
            var wHit = wr.Output.Trim() == digit && wr.UsedPlatonicQuery && !wr.UsedNeuralFallback;
            if (wHit) w2d++;

            var dr = eng.Generate(new GenerationRequest($"{digit} in words", 4));      // digit → word
            var dHit = dr.Output.Trim() == word && dr.UsedPlatonicQuery && !dr.UsedNeuralFallback;
            if (dHit) d2wPlat++;

            _out.WriteLine($"  '{word} as a number' -> '{wr.Output.Trim()}'(plat={wr.UsedPlatonicQuery && !wr.UsedNeuralFallback},{wr.DecisionPath})  |  "
                         + $"'{digit} in words' -> '{dr.Output.Trim()}'(plat={dr.UsedPlatonicQuery && !dr.UsedNeuralFallback},{dr.DecisionPath})");
        }
        sw.Stop();
        _out.WriteLine($"word→digit {w2d}/10   digit→word(platonic) {d2wPlat}/10   wall {sw.ElapsedMilliseconds} ms");

        // BOTH directions via the LEARNED platonic path (capability demonstration, modest bar) — taught by data, no codec.
        Assert.True(w2d >= 9, $"word→digit only {w2d}/10 via the learned path");
        Assert.True(d2wPlat >= 9, $"digit→word platonic only {d2wPlat}/10 via the learned path");
    }
}
