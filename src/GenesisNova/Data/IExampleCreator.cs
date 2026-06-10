using System.Collections.Immutable;

namespace GenesisNova.Data;

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
}
