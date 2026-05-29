using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class SayWordCreator : IExampleCreator
{
    private const int StepSize = 10;

    public string Name => "say:word";
    public int EstimatedComplexity => 10;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var words = WordsForLevel(difficulty);
        if (words.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        return Enumerable.Range(0, count)
            .Select(i =>
            {
                var word = words[i % words.Length];
                return ($"say {word}", word);
            })
            .ToImmutableArray();
    }

    private static string[] WordsForLevel(int level)
    {
        var pool = SayWordPool.Words;
        var rng = new Random(StableHash(nameof(SayWordCreator), level));
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
