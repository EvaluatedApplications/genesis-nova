using System.Collections.Immutable;
using System.Globalization;
using GenesisNova.Core;

namespace GenesisNova.Data.Creators;

/// <summary>
/// FOUNDATIONAL bootstrap lesson: number-word ↔ digit equivalence ("five" ≡ "5").
///
/// This emits TRAINING DATA only — bidirectional ("five"→"5" and "5"→"five") pairs. It does NOT
/// hardcode the model's answer or inject a lookup table into inference: the equivalence is still
/// LEARNED relationally by the platonic space (the input/output concepts get coupled into a relation
/// edge, which is the genuine carrier of "one ≡ 1"). The creator is the training REGIME that makes
/// the demonstrated capability reliably emerge; the demonstration test proves it CAN.
///
/// Difficulty widens the range: d0 = single digits 0–9 (the core), then teens, then the tens.
/// </summary>
public sealed class NumberWordCreator : IExampleCreator
{
    // The number → word vocabulary is the shared reference table (also used by the answer-equivalence
    // grader, so they can never drift). A word LIST is reference data, not a heuristic baked into the
    // model — the NN still learns the equivalence from pairs. See NumberWordVocabulary.
    private static readonly (int Value, string Word)[] Vocabulary = NumberWordVocabulary.Entries;

    // "corenova:" prefix marks a CORE tool-training lesson — it teaches the model to USE the platonic
    // space (here: the number-word↔digit equivalence relation), as opposed to an outcome-/answer-
    // producing creator. Lets the UI/planner and humans tell foundational tool-training apart.
    public string Name => "corenova:number-word-equiv";

    // Foundational — simplest tier, learned before arithmetic that depends on it.
    public int EstimatedComplexity => 8;

    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var vocab = SliceForLevel(Math.Max(0, difficulty));
        if (vocab.Length == 0 || count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (value, word) = vocab[(i / 2) % vocab.Length];
            var digit = value.ToString(CultureInfo.InvariantCulture);
            // Alternate direction so the relation is coupled symmetrically: word→digit and digit→word.
            // The bare-token form (no prose) keeps the lesson focused and the coupling clean.
            examples.Add((i % 2 == 0) ? (word, digit) : (digit, word));
        }

        return examples.ToImmutable();
    }

    private static (int Value, string Word)[] SliceForLevel(int level)
    {
        // d0 = 0–9 (10), d1 = 0–19 (20), d2+ = full table including the tens.
        var take = level switch
        {
            0 => 10,
            1 => 20,
            _ => Vocabulary.Length,
        };
        return Vocabulary.Take(Math.Min(take, Vocabulary.Length)).ToArray();
    }
}
