using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class SayThreeWordsCreator : IExampleCreator
{
    private const int StepSize = 10;

    public string Name => "say:threewords";
    public int EstimatedComplexity => 14;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var triples = TriplesForLevel(difficulty);
        if (triples.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        return Enumerable.Range(0, count)
            .Select(i =>
            {
                var (w1, w2, w3) = triples[i % triples.Length];
                var phrase = $"{w1} {w2} {w3}";
                return ($"say {phrase}", phrase);
            })
            .ToImmutableArray();
    }

    private static (string w1, string w2, string w3)[] TriplesForLevel(int level)
    {
        var pool = SayWordPool.Words;
        var rng = new Random(StableHash(nameof(SayThreeWordsCreator), level));
        var triples = new (string, string, string)[StepSize];
        for (var i = 0; i < StepSize; i++)
            triples[i] = (pool[rng.Next(pool.Length)], pool[rng.Next(pool.Length)], pool[rng.Next(pool.Length)]);
        return triples;
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
