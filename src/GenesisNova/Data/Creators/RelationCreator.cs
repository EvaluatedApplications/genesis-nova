using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class RelationCreator : IExampleCreator
{
    private const int StepSize = 12;

    private static readonly (string item, string category)[] Relations =
    [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"),
        ("dog", "animal"), ("cat", "animal"), ("wolf", "animal"), ("bear", "animal"),
        ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
        ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"), ("boat", "vehicle"),
        ("piano", "instrument"), ("drum", "instrument"), ("violin", "instrument"), ("flute", "instrument"),
        ("oak", "tree"), ("pine", "tree"), ("cedar", "tree"), ("maple", "tree")
    ];

    public string Name => "relation:category";
    public int EstimatedComplexity => 20;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var levelPairs = SliceForLevel(difficulty);
        if (levelPairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        var prompts = new Func<(string item, string category), (string input, string output)>[]
        {
            p => ($"{p.item} is a", p.category),
            p => ($"category of {p.item}", p.category),
            p => ($"what is {p.item}", p.category),
            p => ($"{p.item} belongs to", p.category)
        };

        return Enumerable.Range(0, count).Select(i =>
        {
            var pair = levelPairs[i % levelPairs.Length];
            return prompts[i % prompts.Length](pair);
        }).ToImmutableArray();
    }

    private static (string item, string category)[] SliceForLevel(int level)
    {
        var start = (Math.Max(0, level) * StepSize) % Relations.Length;
        var length = Math.Min(StepSize, Relations.Length);
        return Enumerable.Range(0, length)
            .Select(i => Relations[(start + i) % Relations.Length])
            .ToArray();
    }
}
