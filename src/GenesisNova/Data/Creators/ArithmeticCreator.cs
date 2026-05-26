using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// Arithmetic creator: trains add/sub/mul/div with diverse input surfaces.
/// 
/// Titrated curriculum — each difficulty level adds StepSize new operand pairs
/// in an expanding range (deterministic via seeded RNG).
/// </summary>
public sealed class ArithmeticCreator : IExampleCreator
{
    private readonly string _operation;

    public ArithmeticCreator(string operation) => _operation = operation;

    public string Name => $"arithmetic:{_operation}";

    public int EstimatedComplexity => _operation switch
    {
        "add" => 20,
        "sub" => 20,
        "mul" => 25,
        "div" => 30,
        _ => 20
    };

    private const int StepSize = 10;
    private const int RangeStep = 5;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var pairs = PairsForLevel(difficulty);
        if (pairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        var rng = new Random();
        var examples = new List<(string, string)>(count);

        while (examples.Count < count)
        {
            var (a, b, answer) = pairs[examples.Count % pairs.Length];
            int surfaceIdx = rng.Next(Surfaces(_operation).Length);

            examples.Add((
                BuildInput(a, b, surfaceIdx),
                F(answer)
            ));
        }

        return examples
            .OrderBy(_ => rng.Next())
            .Take(count)
            .ToImmutableArray();
    }

    // ── Deterministic pair generation ────────────────────────────────────

    private (double a, double b, double answer)[] PairsForLevel(int level)
    {
        var rng = new Random(StableHash(_operation, level));
        bool logOp = _operation is "mul" or "div";
        int curMax = (level + 1) * RangeStep;
        int prevMax = level * RangeStep;

        var pairs = new (double a, double b, double answer)[StepSize];
        for (int i = 0; i < StepSize; i++)
        {
            double a, b;
            if (level == 0)
            {
                a = logOp ? rng.Next(1, curMax + 1) : rng.Next(-curMax, curMax + 1);
                b = logOp ? rng.Next(1, curMax + 1) : rng.Next(-curMax, curMax + 1);
            }
            else
            {
                int mag = rng.Next(prevMax + 1, curMax + 1);
                a = logOp ? mag : (rng.Next(2) == 0 ? mag : -mag);
                b = logOp ? rng.Next(1, curMax + 1) : rng.Next(-curMax, curMax + 1);
            }

            if (_operation == "div" && b == 0) b = 1;
            pairs[i] = (a, b, Compute(a, b));
        }

        return pairs;
    }

    private double Compute(double a, double b) => _operation switch
    {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => b == 0 ? 0 : a / b,
        _ => throw new ArgumentException($"Unknown operation: {_operation}")
    };

    private string BuildInput(double a, double b, int surfaceIdx)
    {
        var surfaces = Surfaces(_operation);
        return string.Format(surfaces[surfaceIdx % surfaces.Length], F(a), F(b));
    }

    private static string[] Surfaces(string op) => op switch
    {
        "add" => [
            "add {0} {1}",
            "{0} + {1}",
            "what is {0} plus {1}?",
            "{0} plus {1}",
            "the sum of {0} and {1}",
        ],
        "sub" => [
            "sub {0} {1}",
            "{0} - {1}",
            "what is {0} minus {1}?",
            "{0} minus {1}",
            "the difference of {0} and {1}",
        ],
        "mul" => [
            "multiply {0} by {1}",
            "{0} * {1}",
            "what is {0} times {1}?",
            "{0} times {1}",
            "the product of {0} and {1}",
        ],
        "div" => [
            "divide {0} by {1}",
            "{0} / {1}",
            "what is {0} divided by {1}?",
            "{0} divided by {1}",
        ],
        _ => ["{0},{1}"]
    };

    private static string F(double v) =>
        Math.Abs(v - Math.Round(v)) < 1e-9
            ? ((long)Math.Round(v)).ToString(CultureInfo.InvariantCulture)
            : v.ToString("F2", CultureInfo.InvariantCulture);

    private static int StableHash(string s, int extra)
    {
        uint h = 2166136261u;
        foreach (char c in s) { h ^= (uint)c; h *= 16777619u; }
        h ^= (uint)extra; h *= 16777619u;
        return (int)h;
    }
}
