using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// PER-COMPONENT REGIMEN for the Compare block (PROJECT_GLIDER.md §6). Bare, focused form
/// "compare {a} {b}" → "greater" | "less" | "equal" — ONE prompt shape (bare beats diverse) so the
/// lesson is masterable. The answer is produced platonically: PlatonicGliderInterpreter runs the
/// hand-built glider Branch(Compare(&gt;), "greater", Branch(Compare(&lt;), "less", "equal")) on the
/// substrate (the difference-sign predicate), so RequirePlatonicForCorrect credits it. The creator is
/// the TRAINING regime that makes the routing to that block reliably emerge; the demonstration test
/// proves it CAN. Difficulty widens the operand range.
/// </summary>
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
        if (count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (left, right) = pairs[i % pairs.Length];
            var input = string.Format(CultureInfo.InvariantCulture, "compare {0} {1}", left, right);
            var output = left > right ? "greater" : left < right ? "less" : "equal";
            examples.Add((input, output));
        }

        return examples.ToImmutable();
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
