using GenesisNova.Core;
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

    public static void WarmRoleHead(GenesisRuntimeState s, int cycles = 60)
        => WarmRoleHead(s.Tokenizer, s.Model, s.Inference, cycles);

    public static void WarmRoleHead(GenesisEvalAppRuntime rt, int cycles = 60)
        => WarmRoleHead(rt.State, cycles);
}
