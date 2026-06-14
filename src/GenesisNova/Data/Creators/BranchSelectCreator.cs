using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// PER-COMPONENT REGIMEN for the Branch block (PROJECT_GLIDER.md §6). Bare, focused form
/// "larger {a} {b}" → the larger operand. The answer is produced platonically: the hand-built glider
/// Branch(Compare(&gt;, op0, op1), op0, op1) runs on the substrate — the SELECTION is driven by the
/// element-native Compare predicate (difference sign), and the chosen Operand is returned verbatim. So
/// this exercises Branch (control flow) on a platonic condition, credited via the platonic path. The
/// creator is the TRAINING regime; the demonstration test proves the routing CAN emerge. Difficulty
/// widens the operand range. Equal pairs are excluded so the larger is always well-defined.
/// </summary>
public sealed class BranchSelectCreator : IExampleCreator
{
    private const int StepSize = 24;
    private const int RangeStep = 6;

    public string Name => "numeric:larger";
    public int EstimatedComplexity => 23;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        if (count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var pairs = PairsForLevel(Math.Max(0, difficulty));
        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (left, right) = pairs[i % pairs.Length];
            var input = string.Format(CultureInfo.InvariantCulture, "larger {0} {1}", left, right);
            // The larger operand, formatted exactly as it appears in the input (Operand returns the token).
            var output = (left >= right ? left : right).ToString(CultureInfo.InvariantCulture);
            examples.Add((input, output));
        }

        return examples.ToImmutable();
    }

    private static (long Left, long Right)[] PairsForLevel(int difficulty)
    {
        var max = Math.Max(4, (difficulty + 1) * RangeStep);
        var rng = new Random(StableHash(nameof(BranchSelectCreator), difficulty));
        var result = new (long Left, long Right)[StepSize];

        for (var i = 0; i < StepSize; i++)
        {
            long left = rng.Next(-max, max + 1);
            long right = rng.Next(-max, max + 1);
            if (left == right)
                right = left + 1; // exclude equal: keep the larger well-defined
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
