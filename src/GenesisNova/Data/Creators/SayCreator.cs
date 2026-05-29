using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

/// <summary>
/// "say {phrase}" -> "{phrase}" exact echo creator.
/// Difficulty controls phrase length (difficulty + 1 words).
/// </summary>
public sealed class SayCreator : IExampleCreator
{
    public string Name => "say:exact";
    public int EstimatedComplexity => 10;

    private static readonly string[] WordPool =
    [
        "hello", "hi", "hey", "goodbye", "bye", "welcome",
        "good", "great", "well", "ok", "fine", "ready", "sure",
        "yes", "no", "now", "here", "go", "stop", "wait",
        "one", "two", "three", "four", "five", "six", "seven",
        "cat", "dog", "sun", "moon", "water", "fire", "light",
        "red", "blue", "green", "fast", "slow", "big", "small",
        "i", "you", "we", "it", "a", "the", "my", "your",
        "am", "are", "is", "do", "can", "know", "think", "feel"
    ];

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var wordCount = Math.Max(1, difficulty + 1);
        var rng = new Random(StableHash(nameof(SayCreator), difficulty));
        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);

        for (var i = 0; i < count; i++)
        {
            var words = new string[wordCount];
            for (var w = 0; w < wordCount; w++)
                words[w] = WordPool[rng.Next(WordPool.Length)];

            var phrase = string.Join(" ", words);
            examples.Add(($"say {phrase}", phrase));
        }

        return examples.ToImmutable();
    }

    private static int StableHash(string source, int extra)
    {
        uint h = 2166136261u;
        foreach (var c in source)
        {
            h ^= c;
            h *= 16777619u;
        }

        h ^= (uint)extra;
        h *= 16777619u;
        return (int)h;
    }
}
