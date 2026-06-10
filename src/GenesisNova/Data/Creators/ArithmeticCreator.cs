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

    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public int EstimatedComplexity => _operation switch
    {
        "add" => 20,
        "sub" => 20,
        "mul" => 25,
        "div" => 30,
        _ => 20
    };

    private const int StepSize = 24;
    private const int RangeStep = 4;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var pairs = PairsForLevel(difficulty);
        if (pairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        var level = Math.Max(0, difficulty);
        var surfaces = Surfaces(_operation, level);
        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (a, b, answer) = pairs[i % pairs.Length];
            var surfaceIdx = (i + level) % surfaces.Length;

            // Guarantee compact symbolic forms (e.g., 1+1, 1-1) appear in generated
            // training data for add/sub creators.
            if ((_operation is "add" or "sub") && i % 3 == 0)
                surfaceIdx = 1; // compact form index in Surfaces(op)

            examples.Add((
                BuildInput(a, b, surfaceIdx, level),
                F(answer)
            ));
        }

        return examples.ToImmutable();
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
            if (_operation == "div")
            {
                // Keep division targets learnable: generate mostly exact divisions.
                var denomMax = Math.Max(2, curMax);
                b = rng.Next(1, denomMax + 1);
                var quotient = rng.Next(0, denomMax + 1);
                if (level > 0 && rng.NextDouble() < 0.35)
                {
                    quotient = rng.Next(-denomMax, denomMax + 1);
                }

                a = quotient * b;
            }
            else
            {
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

    private string BuildInput(double a, double b, int surfaceIdx, int difficulty)
    {
        var surfaces = Surfaces(_operation, difficulty);
        return string.Format(surfaces[surfaceIdx % surfaces.Length], F(a), F(b));
    }

    private static string[] Surfaces(string op, int difficulty) => op switch
    {
        "add" => difficulty switch
        {
            0 => ["add {0} {1}", "{0}+{1}", "{0} + {1}", "{0} plus {1}"],
            1 => ExpandSynonyms(
                ["add {0} {1}", "{0}+{1}", "{0} + {1}", "{0} {plusWord} {1}", "{askLead} {0} {plusWord} {1}?", "the {sumWord} of {0} and {1}"],
                new Dictionary<string, string[]>
                {
                    ["{plusWord}"] = ["plus", "added to"],
                    ["{askLead}"] = ["what is", "what's"],
                    ["{sumWord}"] = ["sum", "total"]
                }),
            _ => ExpandSynonyms(
                ["add {0} {1}", "{0}+{1}", "{0} + {1}", "{0} {plusWord} {1}", "{askLead} {0} {plusWord} {1}?", "the {sumWord} of {0} and {1}", "{calcVerb} {0} + {1}", "{computeVerb} the {sumWord}: {0} and {1}", "if you {addVerb} {0} and {1}, what do you get?"],
                new Dictionary<string, string[]>
                {
                    ["{plusWord}"] = ["plus", "added to"],
                    ["{askLead}"] = ["what is", "what's", "tell me"],
                    ["{sumWord}"] = ["sum", "total"],
                    ["{calcVerb}"] = ["calculate", "work out"],
                    ["{computeVerb}"] = ["compute", "find"],
                    ["{addVerb}"] = ["add", "combine"]
                })
        },
        "sub" => difficulty switch
        {
            0 => ["sub {0} {1}", "{0}-{1}", "{0} - {1}", "{0} minus {1}"],
            1 => ExpandSynonyms(
                ["sub {0} {1}", "{0}-{1}", "{0} - {1}", "{0} {minusWord} {1}", "{askLead} {0} {minusWord} {1}?", "the {diffWord} of {0} and {1}"],
                new Dictionary<string, string[]>
                {
                    ["{minusWord}"] = ["minus", "take away"],
                    ["{askLead}"] = ["what is", "what's"],
                    ["{diffWord}"] = ["difference", "delta"]
                }),
            _ => ExpandSynonyms(
                ["sub {0} {1}", "{0}-{1}", "{0} - {1}", "{0} {minusWord} {1}", "{askLead} {0} {minusWord} {1}?", "the {diffWord} of {0} and {1}", "{calcVerb} {0} - {1}", "{computeVerb} the {diffWord}: {0} and {1}", "if you {subVerb} {1} from {0}, what remains?"],
                new Dictionary<string, string[]>
                {
                    ["{minusWord}"] = ["minus", "take away"],
                    ["{askLead}"] = ["what is", "what's", "tell me"],
                    ["{diffWord}"] = ["difference", "delta"],
                    ["{calcVerb}"] = ["calculate", "work out"],
                    ["{computeVerb}"] = ["compute", "find"],
                    ["{subVerb}"] = ["subtract", "take"]
                })
        },
        "mul" => difficulty switch
        {
            0 => ["multiply {0} by {1}", "{0}*{1}", "{0} * {1}", "{0} times {1}"],
            1 => ExpandSynonyms(
                ["multiply {0} by {1}", "{0}*{1}", "{0} * {1}", "{0} {timesWord} {1}", "{askLead} {0} {timesWord} {1}?", "the {prodWord} of {0} and {1}"],
                new Dictionary<string, string[]>
                {
                    ["{timesWord}"] = ["times", "multiplied by"],
                    ["{askLead}"] = ["what is", "what's"],
                    ["{prodWord}"] = ["product", "result"]
                }),
            _ => ExpandSynonyms(
                ["multiply {0} by {1}", "{0}*{1}", "{0} * {1}", "{0} {timesWord} {1}", "{askLead} {0} {timesWord} {1}?", "the {prodWord} of {0} and {1}", "{calcVerb} {0} * {1}", "{computeVerb} the {prodWord}: {0} and {1}", "if {0} groups each contain {1}, what is the {totalWord}?"],
                new Dictionary<string, string[]>
                {
                    ["{timesWord}"] = ["times", "multiplied by"],
                    ["{askLead}"] = ["what is", "what's", "tell me"],
                    ["{prodWord}"] = ["product", "result"],
                    ["{calcVerb}"] = ["calculate", "work out"],
                    ["{computeVerb}"] = ["compute", "find"],
                    ["{totalWord}"] = ["total", "sum"]
                })
        },
        "div" => difficulty switch
        {
            0 => ["divide {0} by {1}", "{0}/{1}", "{0} / {1}", "{0} divided by {1}"],
            1 => ExpandSynonyms(
                ["divide {0} by {1}", "{0}/{1}", "{0} / {1}", "{0} divided by {1}", "{askLead} {0} divided by {1}?", "the {quotWord} of {0} and {1}"],
                new Dictionary<string, string[]>
                {
                    ["{askLead}"] = ["what is", "what's"],
                    ["{quotWord}"] = ["quotient", "ratio"]
                }),
            _ => ExpandSynonyms(
                ["divide {0} by {1}", "{0}/{1}", "{0} / {1}", "{0} divided by {1}", "{askLead} {0} divided by {1}?", "the {quotWord} of {0} and {1}", "{calcVerb} {0} / {1}", "{computeVerb} the {quotWord}: {0} and {1}", "if you {splitVerb} {0} into {1} equal parts, what is each part?"],
                new Dictionary<string, string[]>
                {
                    ["{askLead}"] = ["what is", "what's", "tell me"],
                    ["{quotWord}"] = ["quotient", "ratio"],
                    ["{calcVerb}"] = ["calculate", "work out"],
                    ["{computeVerb}"] = ["compute", "find"],
                    ["{splitVerb}"] = ["split", "divide"]
                })
        },
        _ => ["{0},{1}"]
    };

    private static string[] ExpandSynonyms(string[] templates, IReadOnlyDictionary<string, string[]> replacements)
    {
        var expanded = templates.ToList();
        foreach (var (token, values) in replacements)
        {
            if (values.Length == 0)
                continue;

            var next = new List<string>(expanded.Count * values.Length);
            foreach (var template in expanded)
            {
                if (!template.Contains(token, StringComparison.Ordinal))
                {
                    next.Add(template);
                    continue;
                }

                foreach (var value in values)
                    next.Add(template.Replace(token, value, StringComparison.Ordinal));
            }
            expanded = next;
        }

        return expanded.Distinct(StringComparer.Ordinal).ToArray();
    }

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
