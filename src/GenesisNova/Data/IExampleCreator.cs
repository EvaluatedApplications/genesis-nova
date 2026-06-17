using System.Collections.Generic;
using System.Collections.Immutable;

namespace GenesisNova.Data;

/// <summary>
/// How a creator's answers are graded. The grader is always VALUE-AWARE (2 ≡ two) and contextual (the answer
/// only needs to OCCUR; filler/personality is free). These knobs cover what the grader can't infer:
///  • <see cref="RequirePlatonic"/> — correctness requires the platonic route (capability-mastery, not memorization).
///  • <see cref="AnswerVocabulary"/> — the domain's full set of answer-type words, so the grader can tell a
///    COMPETING wrong answer (e.g. "animal" for a fruit) from free filler in NON-numeric domains. Numeric domains
///    detect competing numbers automatically and need no vocabulary.
/// </summary>
public sealed record GradingPolicy(bool RequirePlatonic = true, IReadOnlyList<string>? AnswerVocabulary = null)
{
    public static readonly GradingPolicy Default = new();
}

/// <summary>
/// Produces training examples for a specific operation or domain.
///
/// Rules all creators must follow:
///  1. difficulty=0 MUST produce at least some minimal learnable examples.
///  2. If count exceeds the creator's unique example space, sample with replacement until count is reached.
///  3. Examples should be deterministic for the same (creator, difficulty) pair.
/// </summary>
public interface IExampleCreator
{
    /// <summary>Unique name (e.g. "arithmetic:add", "language:greet").</summary>
    string Name { get; }

    /// <summary>Rough estimate of learning complexity (lower = simpler).</summary>
    int EstimatedComplexity { get; }

    /// <summary>Generate <paramref name="count"/> examples at the given difficulty level.</summary>
    ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining);

    /// <summary>Training surface used by this creator.</summary>
    GenesisTrainingExampleKind TrainingKind { get; }

    /// <summary>Grading policy (require-platonic + answer vocabulary). Default: require-platonic, numeric-auto.</summary>
    GradingPolicy Grading => GradingPolicy.Default;

    /// <summary>ALL acceptable answers for an input — ambiguous/one-to-many domains return the full valid set
    /// (e.g. apple → {fruit, food}); numeric value-equivalence (2 ≡ two) is handled by the grader. Default: just
    /// the trained output.</summary>
    IReadOnlyList<string> AcceptableAnswers(string input, string trainedOutput, int difficulty) => new[] { trainedOutput };
}
