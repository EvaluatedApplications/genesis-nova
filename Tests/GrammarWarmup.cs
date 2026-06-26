using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Runtime;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

/// <summary>
/// Warms the per-token ROLE head (the NN structure recogniser) so a bare/cold engine parses ordinary assertions
/// ("my name is X") the way the gym-trained PRODUCTION model does. This is NOT a hardcoded fallback — the role parser
/// is LEARNED; the engine deliberately has no copula/possessive word-list. Production warms it on startup via the gym's
/// <see cref="GrammarCurriculum"/>; bare-engine unit tests must do the same to reflect reality.
///
/// Crucially this trains ONLY the head (<see cref="GenesisNeuralModel.TrainExample"/> with role labels) — it does NOT
/// route through the trainer's <c>ObserveLearningSignals</c>, which would call <c>ObservePlatonicSpace</c> and write the
/// curriculum's NONCE facts into the space, diluting the test's own real fact ("my name"→X) via hub competition. So:
/// accumulate the role-alignment counters (<see cref="GenesisInferenceEngine.ObserveGrammar"/>, counters only, no space
/// writes), derive the self-supervised labels, and train the head. The space stays clean.
/// </summary>
internal static class GrammarWarmup
{
    public static void WarmRoleHead(IGenesisTokenizer tok, GenesisNeuralModel model, GenesisInferenceEngine engine, int cycles = 60)
    {
        var grammar = new GrammarCurriculum(trainPerCycle: 48, seed: 1234); // SEEDED → deterministic convergence (no flaky threshold)
        for (var c = 0; c < cycles; c++)
            foreach (var (input, output) in grammar.NextTrainBatch())
            {
                engine.ObserveGrammar(input, output);        // accumulate role-learning counters — NO space writes
                var roleLabels = engine.DeriveRoleLabels(input);
                if (roleLabels is null) continue;            // not a settled grammar frame yet
                var inTok = tok.Encode(input);
                var tgtTok = tok.Encode(output, addEos: true); // encode FIRST (may grow vocab), THEN size the model — else a new token id exceeds the decoder's class count (CUDA CE assert)
                model.EnsureVocabularySize(tok.VocabularySize);
                model.TrainExample(inTok, tgtTok, tok.BosTokenId, roleLabels: roleLabels);
                model.CloneParametersToBreakGraph();
            }
    }

    /// <summary>Teach the WORDED op-cues ("plus"→add, "product"→multiply, …) by LEARNED cue→op relations — the engine
    /// has no hardcoded TryOpCue list. FAST: no GRU training, just relation edits (proven sound in 3 examples/op,
    /// OpCueLearningDirectTest), so a caller can stay [Fact]. Production learns these via <see cref="OpCueCurriculum"/>.</summary>
    public static void WarmOpCues(GenesisInferenceEngine engine, int cycles = 3)
    {
        var opcue = new OpCueCurriculum(trainPerCycle: 48);
        for (var c = 0; c < cycles; c++)
            foreach (var (input, output) in opcue.NextTrainBatch())
                engine.LearnArithmeticCue(input, output);
    }

    /// <summary>Warm the role head AFTER training the GRU on real gym skills, so its per-token features are
    /// DISCRIMINATIVE — the role head reads those features, and on a near-random GRU it generalises only to POSSESSIVE
    /// subjects ("my name is X"); with a trained GRU it tags BARE/"the" subjects ("alice is doctor") too. This is what
    /// production does and what DurableMechanismTests proves. Heavier than the head-only warm (it trains the GRU via
    /// TrainStep, which also writes the gym/grammar facts into the space — tolerable, same as Durable), so callers stay
    /// [SlowFact]. Interleaves gym (features) + seeded grammar (role labels) each cycle.</summary>
    public static void WarmRoleHeadWithGym(IGenesisTokenizer tok, GenesisNeuralModel model, IPlatonicSpace space,
        GenesisNovaConfig config, int gymCycles = 12)
    {
        // Train the GRU on a RICH vocabulary (arithmetic + synonym/category/number WORDS) so its per-token features are
        // discriminative enough for the role head to tag arbitrary BARE subjects ("alice") — the real-word diversity is
        // what a pure-arithmetic warm lacks. CRUCIALLY this is HEAD-ONLY (model.TrainExample, NOT the trainer's
        // TrainStep), so ObservePlatonicSpace never runs and NO relation edge (synonym/category fact) is written — the
        // richness is the token diversity, not stored facts, so a scale-recall test sees ONLY its own facts (no hubs).
        var gym = new GymTrainer(startLevel: 1, seed: 12345, skills: new[]
            { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord });
        for (var c = 0; c < gymCycles; c++)
            foreach (var (i, o) in gym.NextTrainBatch())
            {
                var inTok = tok.Encode(i);
                var tgt = tok.Encode(o, addEos: true); // encode BEFORE EnsureVocabularySize (a new token id must not overrun the decoder)
                model.EnsureVocabularySize(tok.VocabularySize);
                model.TrainExample(inTok, tgt, tok.BosTokenId);
                model.CloneParametersToBreakGraph();
            }
        // Then warm the role head on grammar — also head-only, so still no space writes. Extra cycles so multi-word
        // (3-token) subject spans tag consistently between assert and recall.
        var engine = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        WarmRoleHead(tok, model, engine, cycles: 80);
    }

    public static void WarmRoleHead(GenesisRuntimeState s, int cycles = 60)
        => WarmRoleHead(s.Tokenizer, s.Model, s.Inference, cycles);

    public static void WarmRoleHead(GenesisEvalAppRuntime rt, int cycles = 60)
        => WarmRoleHead(rt.State, cycles);
}
