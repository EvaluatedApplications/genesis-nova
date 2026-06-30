using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// Warms the function-word signal from a REAL TEXT CORPUS instead of synthetic templates. The synthetic prebake works,
/// but only after hand-tuning the context breadth of every function word one at a time (prepositions need locatives,
/// copulas need things, conjunctions need cross-cluster spread — miss one and it sticks). Natural text gives that
/// breadth for free: "the"/"of"/"in" precede thousands of different words, so they bridge automatically and read
/// relational, while content words stay clustered by genuine semantics. There is no uniform-random averaging because
/// the distribution is real. This is the prerequisite the Merge parse / fact memory rest on (see [[nova-merge-substrate-plan]]).
///
/// It streams windowed text from a <see cref="PublicTextCorpusCreator"/> (hydrated from HuggingFace, cached to disk) as
/// the training batch, and grades by the SAME function-word-vs-content separation (a property of the space), so we can
/// watch whether the signal warms and HOLDS on real text.
/// </summary>
public sealed class CorpusWarmCurriculum : ITrainingCurriculum
{
    // A clean, broad English corpus (confirmed reachable). Hydrates on first use; small fallback if offline.
    private static readonly PublicTextCorpusCreator Wikipedia = new(
        name: "warm:wikipedia", estimatedComplexity: 1,
        datasetName: "wikimedia/wikipedia", config: "20231101.en", split: "train", textField: "text",
        maxRemoteRows: 0, allowRemoteFetch: true, requireHydration: false,
        fallbackSnippets:
        [
            "The river flows through the valley and past the old stone bridge near the village.",
            "She opened the book on the table and read the first page by the light of the lamp.",
            "A group of scientists studied the way the cells divide and form new tissue over time.",
        ]);

    // Closed-class function words: should read function-like once warm (they bridge everything).
    private static readonly string[] Func =
    {
        "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "and", "or", "but", "not", "this", "that", "it", "he", "she", "they", "we", "you", "his",
        "her", "its", "their", "my", "your", "has", "have", "had", "will", "would", "can",
    };
    // Common open-class content words that occur in any corpus: should NOT read function-like.
    private static readonly string[] Content =
    {
        "water", "house", "world", "people", "time", "city", "year", "music", "river", "book", "school", "family",
        "country", "system", "language", "science", "history", "art", "war", "land", "church", "money", "art", "story",
    };

    /// <summary>The SHARED Wikipedia corpus source (same on-disk cache / hydration) so a peer corpus curriculum —
    /// e.g. <see cref="CorpusPredictionCurriculum"/> (masked cloze) — streams from the exact same text, not a second
    /// download. The standalone <see cref="CorpusWarmCurriculum"/> falls back to it too.</summary>
    public static PublicTextCorpusCreator SharedCorpus => Wikipedia;

    private readonly PublicTextCorpusCreator _corpus;
    private readonly int _trainPerCycle;
    private double _lastScore;

    public CorpusWarmCurriculum(int trainPerCycle = 128, PublicTextCorpusCreator? corpus = null)
    {
        _trainPerCycle = Math.Max(16, trainPerCycle);
        _corpus = corpus ?? Wikipedia;
    }

    public string Name => "warm:corpus";
    public int Difficulty => 1;                 // no levels: one job, warm the function-word signal from real text
    public bool IsMastered => _lastScore >= 0.85;

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        // Windowed next-chunk examples from real text. Training on them makes the substrate OBSERVE real co-occurrence;
        // the function-word signal warms from that. The substrate self-bounds under corpus-scale vocabulary via its own
        // relevance-decay discharge (DialecticalSpace.DischargeIrrelevant) — the rare long tail is released by low observation/
        // connectivity, so there is NO corpus-specific vocabulary filter here (that would be the overfit band-aid).
        try
        {
            var batch = _corpus.GenerateAsync(_trainPerCycle, difficulty: 1, forTraining: true).GetAwaiter().GetResult();
            return batch.IsDefaultOrEmpty ? Array.Empty<(string, string)>() : batch.ToArray();
        }
        catch { return Array.Empty<(string, string)>(); }
    }

    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>();  // graded by SelfAssess, not surface echo

    /// <summary>READINESS (property of the space): do the closed-class FUNCTION words read function-like (low neighbourhood
    /// clustering — they bridge) while common CONTENT words do not? Score = frac(func fn-like) − frac(content fn-like).</summary>
    public double? SelfAssess(GenesisEvalAppRuntime runtime)
    {
        if (runtime.State.Memory is not DialecticalSpace ds) return null;
        var f = Func.Count(ds.IsFunctionLike) / (double)Func.Length;
        var c = Content.Count(ds.IsFunctionLike) / (double)Content.Length;
        var score = Math.Max(0.0, f - c);
        _lastScore = score;
        return score;
    }

    public void RecordCycle(CycleGrade grade) { }

    // ── DIAGNOSTICS ──────────────────────────────────────────────────────────────────────────────────────────
    public IReadOnlyList<string> Glue => Func;
    public IReadOnlyList<string> SampleContent(int n) => Content.Take(n).ToList();
}
