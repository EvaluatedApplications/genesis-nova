using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// PER-COMPONENT REGIMEN for the Const block (PROJECT_GLIDER.md §6). Bare, focused forms
/// "double {x}" → x*2 and "triple {x}" → x*3. The answer is produced platonically: the hand-built
/// glider Compute(Multiply, [Operand(0), Const(k)]) runs on the substrate — the named operation carries
/// a numeric CONSTANT (k=2 or 3) that parameterises the element-native multiply (face homomorphism). So
/// this exercises Const feeding Compute, credited via the platonic path. The creator is the TRAINING
/// regime; the demonstration test proves the routing CAN emerge. Difficulty widens the operand range.
/// </summary>
public sealed class ConstScaleCreator : IExampleCreator
{
    private const int StepSize = 24;
    private const int RangeStep = 5;

    public string Name => "numeric:scale";
    public int EstimatedComplexity => 24;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        if (count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var operands = OperandsForLevel(Math.Max(0, difficulty));
        var scales = new (string Name, long Factor)[] { ("double", 2), ("triple", 3) };
        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var x = operands[(i / scales.Length) % operands.Length];
            var (name, factor) = scales[i % scales.Length];
            var input = string.Format(CultureInfo.InvariantCulture, "{0} {1}", name, x);
            var output = (x * factor).ToString(CultureInfo.InvariantCulture);
            examples.Add((input, output));
        }

        return examples.ToImmutable();
    }

    private static long[] OperandsForLevel(int difficulty)
    {
        var max = Math.Max(4, (difficulty + 1) * RangeStep);
        var rng = new Random(StableHash(nameof(ConstScaleCreator), difficulty));
        var result = new long[StepSize];
        // Operands start at 1: the multiplicative (log) face has no representation for 0 (log 0 is
        // undefined), so 0 is excluded from the scale regimen. Compare/larger use the additive face and
        // handle 0 fine; only this multiply-by-const lesson must avoid it.
        for (var i = 0; i < StepSize; i++)
            result[i] = rng.Next(1, max + 1);
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
