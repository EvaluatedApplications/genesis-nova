using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class SayTwoWordsCreator : IExampleCreator
{
    private const int StepSize = 10;

    public string Name => "say:twowords";
    public int EstimatedComplexity => 12;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var pairs = PairsForLevel(difficulty);
        if (pairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        return Enumerable.Range(0, count)
            .Select(i =>
            {
                var (w1, w2) = pairs[i % pairs.Length];
                var phrase = $"{w1} {w2}";
                return ($"say {phrase}", phrase);
            })
            .ToImmutableArray();
    }

    private static (string w1, string w2)[] PairsForLevel(int level)
    {
        var pool = SayWordPool.Words;
        var rng = new Random(StableHash(nameof(SayTwoWordsCreator), level));
        var pairs = new (string, string)[StepSize];
        for (var i = 0; i < StepSize; i++)
            pairs[i] = (pool[rng.Next(pool.Length)], pool[rng.Next(pool.Length)]);
        return pairs;
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
