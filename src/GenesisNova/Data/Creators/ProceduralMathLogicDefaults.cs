using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

/// <summary>
/// Deterministic math and logic QA creators with multi-face prompts.
/// The banks are intentionally explicit and answerable so they reinforce truthy supervision.
/// </summary>
public static class ProceduralMathLogicDefaults
{
    public static readonly LanguageCreator Fractions = new("math:fractions", BuildFractions());
    public static readonly LanguageCreator Percent = new("math:percent", BuildPercent());
    public static readonly LanguageCreator Ratio = new("math:ratio", BuildRatio());
    public static readonly LanguageCreator AlgebraSolve = new("math:algebra-solve", BuildAlgebraSolve());
    public static readonly LanguageCreator Geometry = new("math:geometry", BuildGeometry());

    public static readonly LanguageCreator Boolean = new("logic:boolean", BuildBoolean());
    public static readonly LanguageCreator Implication = new("logic:implication", BuildImplication());
    public static readonly LanguageCreator Quantifiers = new("logic:quantifiers", BuildQuantifiers());
    public static readonly LanguageCreator Ordering = new("logic:ordering", BuildOrdering());
    public static readonly LanguageCreator Syllogism = new("logic:syllogism", BuildSyllogism());

    private static (string, string)[] BuildFractions()
    {
        var pairs = new List<(string, string)>();
        for (var denominator = 2; denominator <= 12; denominator++)
        {
            for (var numerator = 1; numerator < denominator; numerator++)
            {
                var simplified = SimplifyFraction(numerator, denominator);
                pairs.Add(($"simplify {numerator}/{denominator}", simplified));
                pairs.Add(($"reduce {numerator}/{denominator}", simplified));

                if (numerator + 1 < denominator)
                {
                    var sum = SimplifyFraction(numerator + 1, denominator);
                    pairs.Add(($"what is {numerator}/{denominator} + 1/{denominator}", sum));
                }
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildPercent()
    {
        var pairs = new List<(string, string)>();
        int[] rates = [5, 10, 12, 20, 25, 40, 50, 75, 80];
        int[] wholes = [20, 40, 60, 80, 100, 120, 160];
        foreach (var rate in rates)
        {
            foreach (var whole in wholes)
            {
                var value = whole * rate / 100;
                if (whole * rate % 100 != 0)
                    continue;

                pairs.Add(($"what is {rate}% of {whole}", value.ToString(CultureInfo.InvariantCulture)));
                pairs.Add(($"compute {rate}% of {whole}", value.ToString(CultureInfo.InvariantCulture)));
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildRatio()
    {
        var pairs = new List<(string, string)>();
        for (var a = 1; a <= 6; a++)
        {
            for (var b = 1; b <= 6; b++)
            {
                for (var scale = 2; scale <= 4; scale++)
                {
                    pairs.Add(($"simplify ratio {a * scale}:{b * scale}", $"{a}:{b}"));
                    pairs.Add(($"what is the ratio of {a * scale} to {b * scale}", $"{a}:{b}"));
                }
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildAlgebraSolve()
    {
        var pairs = new List<(string, string)>();
        for (var x = 1; x <= 12; x++)
        {
            for (var b = 1; b <= 8; b++)
            {
                pairs.Add(($"solve 2x + {b} = {2 * x + b}", x.ToString(CultureInfo.InvariantCulture)));
                pairs.Add(($"solve 3x - {b} = {3 * x - b}", x.ToString(CultureInfo.InvariantCulture)));
                pairs.Add(($"solve x + {b} = {x + b}", x.ToString(CultureInfo.InvariantCulture)));
                pairs.Add(($"solve x - {b} = {x - b}", x.ToString(CultureInfo.InvariantCulture)));

                if (x % 2 == 0)
                    pairs.Add(($"solve x / 2 = {x / 2}", x.ToString(CultureInfo.InvariantCulture)));
                if (x % 3 == 0)
                    pairs.Add(($"solve x / 3 = {x / 3}", x.ToString(CultureInfo.InvariantCulture)));
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildGeometry()
    {
        var pairs = new List<(string, string)>();
        for (var width = 2; width <= 12; width++)
        {
            for (var height = 2; height <= 12; height++)
            {
                pairs.Add(($"rectangle area {width} {height}", (width * height).ToString(CultureInfo.InvariantCulture)));
                pairs.Add(($"rectangle perimeter {width} {height}", (2 * (width + height)).ToString(CultureInfo.InvariantCulture)));

                if (width == height)
                {
                    pairs.Add(($"square area {width}", (width * width).ToString(CultureInfo.InvariantCulture)));
                    pairs.Add(($"square perimeter {width}", (4 * width).ToString(CultureInfo.InvariantCulture)));
                }

                if ((width * height) % 2 == 0)
                    pairs.Add(($"triangle area base {width} height {height}", ((width * height) / 2).ToString(CultureInfo.InvariantCulture)));
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildBoolean()
    {
        var pairs = new List<(string, string)>();
        (string Text, bool Value)[] values = [("true", true), ("false", false)];
        foreach (var (leftText, leftValue) in values)
        {
            foreach (var (rightText, rightValue) in values)
            {
                pairs.Add(($"{leftText} and {rightText}", (leftValue && rightValue) ? "true" : "false"));
                pairs.Add(($"{leftText} or {rightText}", (leftValue || rightValue) ? "true" : "false"));
                pairs.Add(($"not {leftText}", (!leftValue) ? "true" : "false"));
                pairs.Add(($" {leftText} xor {rightText} ".Trim(), (leftValue ^ rightValue) ? "true" : "false"));
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildImplication()
    {
        var pairs = new List<(string, string)>();
        foreach (var premise in new[] { "true", "false" })
        {
            foreach (var conclusion in new[] { "true", "false" })
            {
                var answer = premise == "true" && conclusion == "false" ? "no" : "yes";
                pairs.Add(($"if premise is {premise} and conclusion is {conclusion}, is the implication true", answer));
                pairs.Add(($"does {premise} imply {conclusion}", answer));
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildQuantifiers()
    {
        var pairs = new List<(string, string)>();
        var facts = new[]
        {
            ("all cats are mammals", "yes"),
            ("all birds are mammals", "no"),
            ("some birds can fly", "yes"),
            ("some rocks can fly", "no"),
            ("no squares are circles", "yes"),
            ("no dogs are animals", "no"),
            ("all apples are fruit", "yes"),
            ("some apples are fruit", "yes"),
            ("all fish are mammals", "no"),
            ("some fish live in water", "yes"),
            ("all triangles have three sides", "yes"),
            ("some triangles have four sides", "no"),
            ("all even numbers are divisible by two", "yes"),
            ("some odd numbers are divisible by two", "no")
        };

        foreach (var (statement, answer) in facts)
        {
            pairs.Add(($"is it true that {statement}", answer));
            pairs.Add(($"quantifier check: {statement}", answer));
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildOrdering()
    {
        var pairs = new List<(string, string)>();
        for (var a = 1; a <= 8; a++)
        {
            for (var b = 1; b <= 8; b++)
            {
                for (var c = 1; c <= 8; c++)
                {
                    var sorted = new[] { a, b, c }.OrderBy(x => x).Select(x => x.ToString(CultureInfo.InvariantCulture));
                    pairs.Add(($"sort {a} {b} {c}", string.Join(' ', sorted)));
                    pairs.Add(($"max of {a} {b} {c}", Math.Max(a, Math.Max(b, c)).ToString(CultureInfo.InvariantCulture)));
                }
            }
        }

        return pairs.Distinct().ToArray();
    }

    private static (string, string)[] BuildSyllogism()
    {
        var pairs = new List<(string, string)>();
        var subjects = new[]
        {
            ("cat", "animal"),
            ("dog", "animal"),
            ("sparrow", "bird"),
            ("rose", "plant"),
            ("car", "vehicle"),
            ("piano", "instrument")
        };

        foreach (var (subject, category) in subjects)
        {
            pairs.Add(($"all {subject}s are {category}s; is a {subject} a {category}", "yes"));
            pairs.Add(($"all {category}s are living things; is a {subject} a living thing", "yes"));
            pairs.Add(($"all {subject}s are {category}s; is a {category} a {subject}", "no"));
        }

        return pairs.Distinct().ToArray();
    }

    private static string SimplifyFraction(int numerator, int denominator)
    {
        var gcd = Gcd(numerator, denominator);
        numerator /= gcd;
        denominator /= gcd;
        return denominator == 1
            ? numerator.ToString(CultureInfo.InvariantCulture)
            : $"{numerator}/{denominator}";
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return Math.Abs(a);
    }
}
