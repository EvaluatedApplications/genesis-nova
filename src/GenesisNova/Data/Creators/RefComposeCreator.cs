using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// PER-COMPONENT REGIMEN for the Ref block (PROJECT_GLIDER.md §6.1) — the HIGHER-ORDER component: a
/// glider that invokes another glider (a "glider gun"). Bare form "twicelarger {a} {b}" → 2*max(a,b).
/// The answer is produced platonically by the hand-built glider Compute(Multiply, [Ref("larger"),
/// Const(2)]): Ref runs the named "larger" sub-glider (Branch+Compare) on the same operands, then the
/// result is doubled (Const+Compute) — composition by REFERENCE, not by inlining. So this exercises Ref
/// reusing an already-built block-glider, credited via the platonic path. The creator is the TRAINING
/// regime; the demonstration test proves the routing CAN emerge. Difficulty widens the operand range.
/// The larger operand is kept &gt;= 1 (the multiplicative/log face has no representation for 0).
/// </summary>
public sealed class RefComposeCreator : IExampleCreator
{
    private const int StepSize = 24;
    private const int RangeStep = 5;

    public string Name => "numeric:twice-larger";
    public int EstimatedComplexity => 26;
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
            var input = string.Format(CultureInfo.InvariantCulture, "twicelarger {0} {1}", left, right);
            var output = (2 * Math.Max(left, right)).ToString(CultureInfo.InvariantCulture);
            examples.Add((input, output));
        }

        return examples.ToImmutable();
    }

    private static (long Left, long Right)[] PairsForLevel(int difficulty)
    {
        var max = Math.Max(4, (difficulty + 1) * RangeStep);
        var rng = new Random(StableHash(nameof(RefComposeCreator), difficulty));
        var result = new (long Left, long Right)[StepSize];

        for (var i = 0; i < StepSize; i++)
        {
            // Left in [1, max] guarantees max(left, right) >= 1, so the doubled value is well-defined on
            // the log face (no 0). Right ranges over [-max, max] to keep both orderings represented.
            long left = rng.Next(1, max + 1);
            long right = rng.Next(-max, max + 1);
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
