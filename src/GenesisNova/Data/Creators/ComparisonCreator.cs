using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

public sealed class ComparisonCreator : IExampleCreator
{
    private const int StepSize = 24;
    private const int RangeStep = 6;

    public string Name => "numeric:compare";
    public int EstimatedComplexity => 22;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var pairs = PairsForLevel(Math.Max(0, difficulty));
        var prompts = new[]
        {
            "compare {0} and {1}",
            "is {0} greater than {1}",
            "relation {0} {1}",
            "{0} ? {1}"
        };

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (left, right) = pairs[i % pairs.Length];
            var prompt = prompts[i % prompts.Length];
            var input = string.Format(CultureInfo.InvariantCulture, prompt, left, right);
            var output = Compare(left, right, i % prompts.Length);
            examples.Add((input, output));
        }

        return examples.ToImmutable();
    }

    private static string Compare(double left, double right, int promptIndex)
    {
        if (promptIndex == 1)
            return left > right ? "yes" : "no";
        if (promptIndex == 3)
            return left > right ? "greater" : left < right ? "less" : "equal";
        return left > right ? "greater" : left < right ? "less" : "equal";
    }

    private static (double Left, double Right)[] PairsForLevel(int difficulty)
    {
        var max = Math.Max(4, (difficulty + 1) * RangeStep);
        var rng = new Random(StableHash(nameof(ComparisonCreator), difficulty));
        var result = new (double Left, double Right)[StepSize];

        for (var i = 0; i < StepSize; i++)
        {
            var left = rng.Next(-max, max + 1);
            var right = rng.Next(-max, max + 1);
            if (i % 8 == 0)
                right = left; // include equal cases
            result[i] = (left, right);
        }

        return result;
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
