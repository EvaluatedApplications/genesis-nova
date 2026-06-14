using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// PROJECT GLIDER — the first glider TRAINING REGISTER (see PROJECT_GLIDER.md).
///
/// Emits framed-arithmetic → "the answer is Z" pairs that exercise the full answer-template glider:
/// a stored SCAFFOLD ("the answer is"), operand RESOLVE (operands appear as digits OR number-words),
/// face COMPUTE, and answer FORMAT (the result is rendered as a digit OR a word). The input SURFACE,
/// the operand forms, and the answer format all vary independently, so to fit the data the model must
/// generalise the STRUCTURE ("emit the scaffold, then resolve→compute→format the slot") rather than
/// memorise input→output products — exactly the glider the hand-built
/// <c>PlatonicGliderInterpreter</c> executes.
///
/// Difficulty widens the operator set and operand range: d0 add only (0-4), d1 +sub (0-8),
/// d2 +mul (0-12), d3+ +div and wider. Deterministic per (creator, difficulty) per IExampleCreator.
/// </summary>
public sealed class GliderAnswerCreator : IExampleCreator
{
    // "corenova:" marks the KIND (tool-training: teaches the model to USE the space to build a
    // structure), not the order. The complexity is HIGH because this glider COMPOSES equivalence
    // (corenova:number-word-equiv) and computation, so the focused curriculum trains it AFTER those.
    public string Name => "corenova:answer-template";
    public int EstimatedComplexity => 28;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    private const string Scaffold = "the answer is";
    private const int StepSize = 24;
    private const int RangeStep = 4;

    // Number-word vocabulary for rendering operands/answers as words (the equivalence the glider's
    // resolve/format steps rely on). Values outside this set render as digits (graceful fallback).
    private static readonly IReadOnlyDictionary<int, string> Words = new Dictionary<int, string>
    {
        [0] = "zero", [1] = "one", [2] = "two", [3] = "three", [4] = "four", [5] = "five",
        [6] = "six", [7] = "seven", [8] = "eight", [9] = "nine", [10] = "ten", [11] = "eleven",
        [12] = "twelve", [13] = "thirteen", [14] = "fourteen", [15] = "fifteen", [16] = "sixteen",
        [17] = "seventeen", [18] = "eighteen", [19] = "nineteen", [20] = "twenty",
        [30] = "thirty", [40] = "forty",
    };

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        if (count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var level = Math.Max(0, difficulty);
        var ops = OpsForLevel(level);
        var pairs = PairsForLevel(level, ops);
        if (pairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (op, a, b, result) = pairs[i % pairs.Length];

            // Independently vary operand surface and answer format so structure — not a memorised
            // (X,Y)→phrase product — is what fits the data.
            var aWord = i % 2 == 0;
            var bWord = (i / 2) % 2 == 0;
            var answerAsWord = i % 3 == 0;

            var surfaces = Surfaces(op);
            var surfaceIdx = (i + level) % surfaces.Length;
            var input = string.Format(
                CultureInfo.InvariantCulture,
                surfaces[surfaceIdx],
                Render(a, aWord),
                Render(b, bWord));

            var output = $"{Scaffold} {Render(result, answerAsWord)}";
            examples.Add((input, output));
        }

        return examples.ToImmutable();
    }

    private static string[] OpsForLevel(int level) => level switch
    {
        0 => ["add"],
        1 => ["add", "sub"],
        2 => ["add", "sub", "mul"],
        _ => ["add", "sub", "mul", "div"],
    };

    private (string Op, int A, int B, int Result)[] PairsForLevel(int level, string[] ops)
    {
        var rng = new Random(StableHash(Name, level));
        var curMax = (level + 1) * RangeStep; // 4, 8, 12, 16, 20 ...
        var pairs = new (string, int, int, int)[StepSize];
        for (var i = 0; i < StepSize; i++)
        {
            var op = ops[i % ops.Length];
            int a, b, result;
            switch (op)
            {
                case "div":
                    b = rng.Next(1, curMax + 1);
                    var quotient = rng.Next(0, curMax + 1);
                    a = quotient * b;
                    result = a / b;
                    break;
                case "mul":
                    a = rng.Next(0, curMax + 1);
                    b = rng.Next(0, curMax + 1);
                    result = a * b;
                    break;
                case "sub":
                    a = rng.Next(0, curMax + 1);
                    b = rng.Next(0, curMax + 1);
                    if (b > a) (a, b) = (b, a); // keep results non-negative (word-renderable)
                    result = a - b;
                    break;
                default: // add
                    a = rng.Next(0, curMax + 1);
                    b = rng.Next(0, curMax + 1);
                    result = a + b;
                    break;
            }

            pairs[i] = (op, a, b, result);
        }

        return pairs;
    }

    private static string[] Surfaces(string op) => op switch
    {
        "add" =>
        [
            "what is {0} plus {1}", "{0} + {1}", "the sum of {0} and {1}",
            "add {0} and {1}", "what is {0} + {1}?",
        ],
        "sub" =>
        [
            "what is {0} minus {1}", "{0} - {1}", "the difference of {0} and {1}",
            "subtract {1} from {0}",
        ],
        "mul" =>
        [
            "what is {0} times {1}", "{0} * {1}", "the product of {0} and {1}",
        ],
        "div" =>
        [
            "what is {0} divided by {1}", "{0} / {1}", "the quotient of {0} and {1}",
        ],
        _ => ["{0} {1}"],
    };

    // Render a value as a number-word (when requested AND in the word vocabulary) or as a digit.
    private static string Render(int value, bool asWord)
        => asWord && Words.TryGetValue(value, out var word)
            ? word
            : value.ToString(CultureInfo.InvariantCulture);

    private static int StableHash(string s, int extra)
    {
        uint h = 2166136261u;
        foreach (var c in s) { h ^= c; h *= 16777619u; }
        h ^= (uint)extra; h *= 16777619u;
        return (int)h;
    }
}
