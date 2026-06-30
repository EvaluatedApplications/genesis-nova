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

    /// <summary>Teach the number-word LEXICON atoms (#5) and the to-word/to-digit/compare intent cues (#3/#4) the same
    /// fast way as <see cref="WarmOpCues"/> (relations only, no GRU training) — for tests that probe number-words /
    /// worded arithmetic / predicate with the codec+lists flipped OFF (the learned defaults). Lexicon first so the cue
    /// learner can type the outputs.</summary>
    public static void WarmNumberWordsAndCues(GenesisInferenceEngine engine)
    {
        string GT(long v) => GenesisNova.Core.NumberWordVocabulary.ToWords(v);
        for (long v = 0; v <= 99; v++) engine.LearnNumberWord($"{v} in words", GT(v));          // lexicon atoms
        foreach (var v in new long[] { 100, 113, 147, 250, 1000, 1234 }) engine.LearnNumberWord($"{v} in words", GT(v)); // scales
        foreach (var v in new long[] { 5, 12, 7, 3, 15, 9, 18 })
        {
            engine.LearnIntentCue($"{v} in words", GT(v));               // ToWord cue
            engine.LearnIntentCue($"write {v} in words", GT(v));
            engine.LearnIntentCue($"{GT(v)} as a number", v.ToString()); // ToDigit cue
        }
        // Compare cue — teach ALL the predicate frame wordings (compared/compare/larger/smaller/bigger/next/with) so
        // every gym predicate frame resolves, not just "compared".
        string[] cmpFrames = { "{0} compared to {1}", "how does {0} compare to {1}", "is {0} larger or smaller than {1}",
                               "which is bigger, {0} or {1}", "put {0} next to {1}", "compare {0} with {1}" };
        foreach (var (a, b) in new[] { (5, 3), (2, 8), (4, 4), (9, 1), (3, 7), (6, 6), (8, 2), (1, 5) })
            foreach (var f in cmpFrames)
                engine.LearnIntentCue(string.Format(f, a, b), a > b ? "greater" : a < b ? "less" : "equal");
    }

    public static void WarmRoleHead(GenesisRuntimeState s, int cycles = 60)
        => WarmRoleHead(s.Tokenizer, s.Model, s.Inference, cycles);

    /// <summary>Warm the runtime the way the PRODUCTION (de-hardcoded) path actually needs it. Under
    /// <see cref="GenesisNovaConfig.WithProductionMechanisms"/> the engine runs with <c>DeHardcodedDispatch</c> →
    /// <c>LearnedCuesOnly</c>, so the fact parser (copula pivot / interrogative / retrieval-frame) reads the LEARNED
    /// distributional signals — the function-word centrality (<c>DialecticalSpace.IsFunctionLike</c>) and the ∘qst
    /// interrogative cue — NOT the hardcoded word-lists a bare engine falls back to. The head-only
    /// <see cref="WarmRoleHead(GenesisRuntimeState,int)"/> writes NOTHING to the space, so those signals stay COLD and
    /// every personal-fact assertion <c>field-abstain</c>s (the copula "is" can't be told from content). Production warms
    /// them with the gym/corpus; this reproduces that with a synthetic corpus exactly as the accepted
    /// <c>QuestionCueDeHardcodingTest</c> does — content CLUSTERS plus function-word BRIDGES that co-occur with every
    /// cluster, so glue/possessive/interrogative tokens read function-like and content words don't, then a few
    /// <c>LearnQuestionCue</c> examples for ∘qst. This is the learned signal, taught by DATA — no hardcoded copula /
    /// possessive / name routine, and no <c>LearnIntentCue</c> (whose ∘ret hub heuristic mislabels grammar recall
    /// frames, polluting "my"/"what" and re-breaking the parse).</summary>
    public static void WarmRoleHead(GenesisEvalAppRuntime rt, int cycles = 60)
    {
        WarmLearnedCues(rt.State);
        WarmRoleHead(rt.State, cycles);
    }

    /// <summary>Corpus-style warm of the LEARNED function-word + interrogative signals the de-hardcoded path consumes
    /// (see <see cref="WarmRoleHead(GenesisEvalAppRuntime,int)"/>). Substrate-driven: the signals EMERGE from the
    /// co-occurrence distribution, nothing is hardcoded at inference.</summary>
    public static void WarmLearnedCues(GenesisRuntimeState s, int steps = 5000, int seed = 11)
    {
        if (s.Memory is not GenesisNova.Cognition.Platonic.DialecticalSpace ds) return;
        // Content clusters: members co-occur with their own kind (high coherence ⇒ content). Bridges: glue / possessive
        // / interrogative tokens that co-occur with EVERY cluster (low coherence ⇒ function-like). This is what windowed
        // corpus text produces; the words live in the DATA, the field stays general (a nonce bridge warms the same way).
        string[][] clusters =
        {
            new[]{"cat","dog","cow","pig","hen","fox","owl","bat","ant","elk"},
            new[]{"red","blue","green","pink","gray","black","white","brown","gold","teal"},
            new[]{"bob","sam","joe","amy","tom","kim","dan","liz","ben","eve"},
            new[]{"name","color","age","city","job","pet","car","book","song","food"},
            new[]{"rome","paris","tokyo","cairo","lima","oslo","delhi","perth","kyoto","nice"},
        };
        string[] bridges = { "what", "who", "where", "when", "which", "is", "are", "was", "the", "a", "an", "of", "to",
                             "my", "your", "his", "her", "their", "this", "that" };
        var rng = new Random(seed);
        for (var step = 0; step < steps; step++)
        {
            var cl = clusters[rng.Next(clusters.Length)];
            var w1 = cl[rng.Next(cl.Length)]; var w2 = cl[rng.Next(cl.Length)];
            if (w1 != w2) ds.ObserveContradiction(w1, w2, 0.15);                                   // intra-cluster bond
            var any = clusters[rng.Next(clusters.Length)][rng.Next(10)];
            ds.ObserveContradiction(bridges[rng.Next(bridges.Length)], any, 0.2);                  // bridge ↔ anything
        }
        // Teach the interrogative cue (∘qst): the answer is ABSENT from the input (a question retrieves it) and the
        // sentence is fronted by a function-like token — that token IS the interrogative. No wh-list (LearnQuestionCue
        // self-checks the learned function-word signal). A handful of frames generalise to any wh question by position.
        foreach (var (q, a) in new[] { ("what is my name", "bob"), ("who is my pet", "dog"), ("where is the cat", "fox"),
                                       ("what is your color", "blue"), ("which is the city", "rome") })
            s.Inference.LearnQuestionCue(q, a);
    }
}
