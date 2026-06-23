using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class RelationCreator : IExampleCreator
{
    private const int StepSize = 16;

    // Shared item→category reference table (see CreatorText.ItemCategories); RelationCreator applies its own
    // StepSize-rotation level slicing over it.
    private static readonly (string item, string category)[] Relations = CreatorText.ItemCategories;

    public string Name => "relation:category";
    public int EstimatedComplexity => 20;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

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
            p => ($"{p.item} belongs to", p.category),
            p => ($"classify {p.item}", p.category),
            p => ($"{p.item} type?", p.category)
        };

        return Enumerable.Range(0, count).Select(i =>
        {
            var pair = levelPairs[i % levelPairs.Length];
            return prompts[i % prompts.Length](pair);
        }).ToImmutableArray();
    }

    private static (string item, string category)[] SliceForLevel(int level)
    {
        var safeLevel = Math.Max(0, level);
        var length = Math.Min(Relations.Length, StepSize + (safeLevel * 4));
        var start = (safeLevel * StepSize) % Relations.Length;
        return Enumerable.Range(0, length)
            .Select(i => Relations[(start + i) % Relations.Length])
            .ToArray();
    }
}
