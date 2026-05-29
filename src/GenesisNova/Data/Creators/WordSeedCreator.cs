using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class WordSeedCreator : IExampleCreator
{
    private const int StepSize = 20;

    public string Name => "language:words";
    public int EstimatedComplexity => 8;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var words = WordsForLevel(difficulty);
        if (words.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        return Enumerable.Range(0, count)
            .Select(i =>
            {
                var word = words[i % words.Length];
                return (word, word);
            })
            .ToImmutableArray();
    }

    private static string[] WordsForLevel(int level)
    {
        var pool = SayWordPool.Words;
        var rng = new Random(StableHash(nameof(WordSeedCreator), level));
        var words = new string[StepSize];
        for (var i = 0; i < StepSize; i++)
            words[i] = pool[rng.Next(pool.Length)];
        return words;
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
