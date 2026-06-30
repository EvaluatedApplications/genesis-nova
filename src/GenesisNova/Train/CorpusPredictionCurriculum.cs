using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Data.Creators;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// GENESIS-NATIVE "BASE MODEL" OBJECTIVE — self-supervised MASKED / CLOZE prediction over a text corpus, so corpus
/// text actively BUILDS knowledge into the substrate (predict → error → learn) instead of only warming the function-
/// word signal passively (the <see cref="CorpusWarmCurriculum"/> peer). This is the founding PLATONIC_MIND thesis —
/// "reasoning = the field relaxing its own surprise" (free-energy / predictive coding) — used as the LLM pre-training
/// loss. The genesis-native FORM is MASKED, not autoregressive: the field is associative / relaxational, so "fill the
/// held-out token from the surrounding context" is precisely what relaxation already does (a query cloud settling into
/// the lowest-surprise basin), whereas left-to-right next-token generation is not.
///
/// HOW (all on the EXISTING substrate — no new gradient invented):
///  • From corpus windows we hold out ONE CONTENT word (the function-word skeleton + the other content words stay as
///    context) and ask the field to PREDICT it. Function words are the skeleton; content carries the knowledge, so we
///    mask content, never function words.
///  • PREDICT = <see cref="DialecticalSpace.Reason"/> over the visible CONTENT context. Reason relaxes the context
///    cloud to a settled basin and EXCLUDES the anchors from the candidates — i.e. it returns a DIFFERENT concept that
///    best fits the context, which is exactly masked fill. (The engine's TryFieldRelax route relaxes from a SINGLE
///    discriminative anchor, so it cannot do full-context cloze; its underlying primitive Reason accepts the whole
///    context and is what we reuse. No new route is added.)
///  • LEARN = the prediction ERROR drives <see cref="DialecticalSpace.FineEditFromExample"/>(context → target) — the
///    SAME substrate write the production trainer's ObservePlatonicSpace uses to couple input→output — reinforcing the
///    correct fill so its cloud overlaps the context; the wrong pull weakens relatively. Correctness-gated (only the
///    currently-wrong examples are reinforced) so a mastered window stops eroding the rest, mirroring the gym's
///    TrainOnFailureOnly anti-erosion regime.
///  • GRADE = HELD-OUT cloze accuracy (windows whose (context→target) pair was never directly reinforced); a held-out
///    fill must GENERALISE from the distributional clouds the train pairs built. This is the SelfAssess signal the gym
///    watches climb.
///
/// Wiring: a NEW peer curriculum — it does NOT touch <see cref="CorpusWarmCurriculum"/>. In the orchestrator loop,
/// <see cref="NextTrainBatch"/> emits the masked (skeleton→target) pairs so the standard TrainAsync observe path
/// relates context→target (and, because the skeleton is retained, ALSO warms the function-word signal — so this could
/// later SUBSUME the passive CorpusWarm prebake); <see cref="SelfAssess"/> grades held-out cloze. The bounded research
/// proof (CorpusPredictionTests) drives the SAME substrate write directly via <see cref="ObserveTrainSplit"/> (no GPU
/// step) and reads the same <see cref="HeldOutClozeAccuracy"/>, so the climb is a property of the substrate, not the NN.
/// </summary>
public sealed class CorpusPredictionCurriculum : ITrainingCurriculum
{
    // Cold-start function-word skeleton (English). Only used to DECIDE which token is maskable CONTENT and to drop
    // skeleton words from the relaxation anchors — a deterministic, warm-state-independent split so the held-out set is
    // STABLE across cycles. Once the space is warm the learned IsFunctionLike signal agrees with this; we keep the list
    // ONLY as the fixed masking key (not as a routing decision). Mirrors CorpusWarmCurriculum.Func.
    private static readonly HashSet<string> ColdFunctionWords = new(StringComparer.Ordinal)
    {
        "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "and", "or", "but", "not", "no", "this", "that", "these", "those", "it",
        "its", "he", "she", "they", "we", "you", "i", "his", "her", "their", "my", "your", "our", "has", "have",
        "had", "will", "would", "can", "could", "do", "does", "did", "so", "than", "then", "if", "into", "out",
        "up", "down", "over", "under", "near", "across", "through", "past", "about", "who", "what", "which",
        "where", "when", "why", "how", "there", "here", "all", "any", "some", "more", "most", "such", "also",
    };

    private readonly List<MaskedExample> _train = new();
    private readonly List<MaskedExample> _heldOut = new();
    private readonly HashSet<string> _targetVocab = new(StringComparer.Ordinal);
    private readonly PublicTextCorpusCreator? _corpus;
    private readonly int _trainPerCycle;
    private double _lastScore;

    /// <summary>One cloze item: the function-word SKELETON of the window with the target removed (for the orchestrator's
    /// surface train pair), the CONTENT context tokens (the relaxation anchors), and the held-out TARGET content word.</summary>
    private readonly record struct MaskedExample(string Skeleton, IReadOnlyList<string> Context, string Target);

    /// <summary>Build directly from a fixed set of source sentences (the bounded research proof / offline use). A masked
    /// example is generated for every maskable CONTENT position; each UNIQUE (context → target) pair is assigned wholly
    /// to TRAIN or HELD-OUT by a stable hash, so no held-out pair is ever directly reinforced.</summary>
    public CorpusPredictionCurriculum(IEnumerable<string> sentences, double heldOutFraction = 0.3, int trainPerCycle = 64)
    {
        _trainPerCycle = Math.Max(8, trainPerCycle);
        BuildExamples(sentences ?? Array.Empty<string>(), Math.Clamp(heldOutFraction, 0.1, 0.5));
    }

    /// <summary>Production peer-curriculum form: stream window text from a real corpus (same source as CorpusWarm).</summary>
    public CorpusPredictionCurriculum(PublicTextCorpusCreator corpus, double heldOutFraction = 0.3, int trainPerCycle = 64)
    {
        _corpus = corpus ?? throw new ArgumentNullException(nameof(corpus));
        _trainPerCycle = Math.Max(8, trainPerCycle);
        RefreshFromCorpus(Math.Clamp(heldOutFraction, 0.1, 0.5));
    }

    public string Name => "predict:cloze";
    public int Difficulty => 1;                  // one job: predict masked corpus tokens; mastery is the cloze bar
    public bool IsMastered => _lastScore >= 0.60;

    /// <summary>Distinct held-out target words — the chance baseline is 1 / this (a random fill).</summary>
    public int TargetVocabulary => _targetVocab.Count;
    public int TrainCount => _train.Count;
    public int HeldOutCount => _heldOut.Count;
    public double ChanceLevel => _targetVocab.Count > 0 ? 1.0 / _targetVocab.Count : 0.0;

    // ── ORCHESTRATOR WIRING ─────────────────────────────────────────────────────────────────────────────────────
    // The masked (skeleton → target) SURFACE pairs for the standard loop: TrainAsync's ObservePlatonicSpace relates the
    // skeleton's content cue → the target (the same FineEditFromExample coupling we drive directly below), AND — because
    // the function-word skeleton is kept in the input — warms the function-word signal as a side effect (so this can
    // later subsume the passive CorpusWarm prebake). Refreshes the corpus stream when corpus-backed.
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        if (_corpus is not null) { try { RefreshFromCorpus(0.3); } catch { } }
        if (_train.Count == 0) return Array.Empty<(string, string)>();
        var rng = Random.Shared;
        return Enumerable.Range(0, Math.Min(_trainPerCycle, _train.Count))
            .Select(_ => _train[rng.Next(_train.Count)])
            .Select(m => (Input: m.Skeleton, Output: m.Target))
            .Where(p => p.Input.Length > 0)
            .ToArray();
    }

    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>(); // graded by SelfAssess (cloze), not surface echo

    /// <summary>The gym's climb signal: HELD-OUT cloze accuracy (read-only). Returns null when the live memory is not a
    /// dialectical space. The orchestrator uses this as the unit's cycle accuracy and skips surface probe-grading.</summary>
    public double? SelfAssess(GenesisEvalAppRuntime runtime)
    {
        if (runtime.State.Memory is not DialecticalSpace ds) return null;
        var acc = HeldOutClozeAccuracy(ds);
        _lastScore = acc;
        return acc;
    }

    public void RecordCycle(CycleGrade grade) { }

    // ── SUBSTRATE PREDICT → ERROR → LEARN (reused directly; what the bounded proof drives, no GPU) ───────────────

    /// <summary>The masked-prediction LEARN step on the substrate: for each TRAIN cloze item, PREDICT the held-out token
    /// by relaxation (<see cref="DialecticalSpace.Reason"/>) over the visible content context; when the fill is WRONG,
    /// the error drives <see cref="DialecticalSpace.FineEditFromExample"/>(context → target) — reinforcing the correct
    /// fill so the target's cloud overlaps the context (the wrong pull weakens relatively). Correctness-gated: an
    /// already-correct item is left untouched (anti-erosion). Returns this pass's TRAIN cloze accuracy (pre-reinforce).</summary>
    public double ObserveTrainSplit(DialecticalSpace ds, int? maxExamples = null)
    {
        if (ds is null || _train.Count == 0) return 0.0;
        var n = Math.Min(maxExamples ?? _train.Count, _train.Count);
        var correct = 0;
        for (var i = 0; i < n; i++)
        {
            var m = _train[i];
            if (m.Context.Count == 0) continue;
            var pred = ds.Reason(m.Context).Symbol;
            if (string.Equals(pred, m.Target, StringComparison.Ordinal)) { correct++; continue; }
            // PREDICTION ERROR → reinforce the correct fill via the substrate's own example-coupling primitive.
            ds.FineEditFromExample(m.Context, new[] { m.Target }, isNegativeExample: false);
        }
        return n > 0 ? correct / (double)n : 0.0;
    }

    /// <summary>HELD-OUT cloze accuracy: the fraction of held-out items whose masked token the field fills EXACTLY by
    /// relaxation. A held-out (context → target) pair is NEVER directly reinforced, so a hit must GENERALISE from the
    /// distributional clouds the train pairs built. Read-only.</summary>
    public double HeldOutClozeAccuracy(DialecticalSpace ds)
    {
        if (ds is null || _heldOut.Count == 0) return 0.0;
        var correct = 0;
        foreach (var m in _heldOut)
        {
            if (m.Context.Count == 0) continue;
            if (string.Equals(ds.Reason(m.Context).Symbol, m.Target, StringComparison.Ordinal)) correct++;
        }
        return correct / (double)_heldOut.Count;
    }

    /// <summary>Diagnostic split of held-out grading: EXACT fill, plus a SEMANTIC-NEAR credit (the settled fill is a
    /// strong relational neighbour of the true target — the right cluster, wrong member). Lets the proof report both.</summary>
    public (double Exact, double Near) HeldOutClozeDetailed(DialecticalSpace ds)
    {
        if (ds is null || _heldOut.Count == 0) return (0.0, 0.0);
        int exact = 0, near = 0;
        foreach (var m in _heldOut)
        {
            if (m.Context.Count == 0) continue;
            var pred = ds.Reason(m.Context).Symbol;
            if (string.Equals(pred, m.Target, StringComparison.Ordinal)) { exact++; near++; continue; }
            if (string.IsNullOrEmpty(pred)) continue;
            if (ds.GetNeighbors(m.Target, PlatonicNeighborhoodType.Relational, 8, 0.0)
                  .Any(nb => string.Equals(nb.Concept, pred, StringComparison.Ordinal)))
                near++;
        }
        return (exact / (double)_heldOut.Count, near / (double)_heldOut.Count);
    }

    // ── MASKED-EXAMPLE GENERATION ────────────────────────────────────────────────────────────────────────────────

    private void RefreshFromCorpus(double heldOutFraction)
    {
        if (_corpus is null) return;
        // The corpus yields WINDOWED (context, next-chunk) pairs; the Input field is a real multi-word text window — use
        // those windows as the source "sentences" to mask. forTraining=false → deterministic, no growth pressure.
        var batch = _corpus.GenerateAsync(Math.Max(32, _trainPerCycle), difficulty: 4, forTraining: false)
            .GetAwaiter().GetResult();
        var windows = batch.IsDefaultOrEmpty
            ? Array.Empty<string>()
            : batch.Select(p => $"{p.Input} {p.Output}").ToArray();
        BuildExamples(windows, heldOutFraction);
    }

    private void BuildExamples(IEnumerable<string> sentences, double heldOutFraction)
    {
        _train.Clear(); _heldOut.Clear(); _targetVocab.Clear();
        var seen = new HashSet<string>(StringComparer.Ordinal); // dedupe identical (context|target) pairs across windows
        var cut = (int)Math.Round(heldOutFraction * 100);
        foreach (var sentence in sentences)
        {
            var tokens = Tokenize(sentence);
            if (tokens.Count < 4) continue;
            var contentIdx = Enumerable.Range(0, tokens.Count).Where(i => IsContent(tokens[i])).ToList();
            if (contentIdx.Count < 3) continue; // need a target plus enough surrounding content to relax from
            foreach (var p in contentIdx)
            {
                var target = tokens[p];
                var context = contentIdx.Where(i => i != p).Select(i => tokens[i]).ToList();
                if (context.Count < 2) continue;
                var skeleton = string.Join(' ', tokens.Where((_, i) => i != p)); // skeleton keeps the function words
                var key = target + "" + string.Join(' ', context);
                if (!seen.Add(key)) continue;
                var example = new MaskedExample(skeleton, context, target);
                // Stable per-pair split: the SAME pair always lands in the SAME split, and a pair is never in BOTH —
                // so a held-out fill is genuinely un-reinforced (tests generalisation, not memorisation).
                if (StableBucket(key) < cut) _heldOut.Add(example);
                else { _train.Add(example); _targetVocab.Add(target); }
            }
        }
        // A target that only ever appears held-out has no cloud to be retrieved from → drop those held-out items so the
        // chance baseline and the score reflect genuinely learnable fills (the target must have been a TRAIN context).
        var learnable = new HashSet<string>(_train.SelectMany(m => m.Context).Concat(_train.Select(m => m.Target)), StringComparer.Ordinal);
        _heldOut.RemoveAll(m => !learnable.Contains(m.Target));
        foreach (var m in _heldOut) _targetVocab.Add(m.Target);
    }

    private static bool IsContent(string t) => t.Length >= 2 && t.All(char.IsLetter) && !ColdFunctionWords.Contains(t);

    private static List<string> Tokenize(string? text)
        => (text ?? string.Empty)
            .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(t => new string(t.Where(c => char.IsLetterOrDigit(c) || c is '\'' or '-').ToArray()).ToLowerInvariant())
            .Where(t => t.Length > 0)
            .ToList();

    // Deterministic 0..99 bucket from a stable FNV-1a hash (no Random — the split must be reproducible run-to-run).
    private static int StableBucket(string key)
    {
        var h = 2166136261u;
        foreach (var c in key) { h ^= c; h *= 16777619u; }
        return (int)(h % 100u);
    }
}
