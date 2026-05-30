using GenesisNova.Data.Creators;

namespace GenesisNova.Data;

/// <summary>
/// Central registry of all available example creators.
/// Sorted by EstimatedComplexity (simplest first).
/// </summary>
public static class ExampleCreatorRegistry
{
    public static readonly IReadOnlyList<IExampleCreator> All = new List<IExampleCreator>
    {
        // ── Arithmetic ────────────────────────────────────────────────────
        new ArithmeticCreator("add"),    // arithmetic:add
        new ArithmeticCreator("sub"),    // arithmetic:sub
        new ArithmeticCreator("mul"),    // arithmetic:mul
        new ArithmeticCreator("div"),    // arithmetic:div

        // ── Numeric reasoning ──────────────────────────────────────────────
        new ComparisonCreator(),         // numeric:compare
        new SequenceCreator(),           // sequence:next

        // ── Relational creators ────────────────────────────────────────────
        new RelationCreator(),               // relation:category

        // ── Language ──────────────────────────────────────────────────────
        LanguageDefaults.Greet,          // language:greet
        LanguageDefaults.Acknowledge,    // language:acknowledge
        LanguageDefaults.Facts,          // language:facts
        LanguageDefaults.Commands,       // language:commands
    };

    /// <summary>Randomly select N creators without replacement (Fisher-Yates).</summary>
    public static IReadOnlyList<IExampleCreator> Pick(int n, Random rng)
    {
        if (n >= All.Count)
        {
            if (n == All.Count)
            {
                var copy = All.ToList();
                for (int i = copy.Count - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (copy[i], copy[j]) = (copy[j], copy[i]);
                }
                return copy;
            }

            var result = new List<IExampleCreator>(n);
            for (int i = 0; i < n; i++)
                result.Add(All[rng.Next(All.Count)]);
            return result;
        }

        var pool = All.ToList();
        var selected = new List<IExampleCreator>(n);
        for (int i = 0; i < n; i++)
        {
            int j = rng.Next(i, pool.Count);
            (pool[i], pool[j]) = (pool[j], pool[i]);
            selected.Add(pool[i]);
        }
        return selected;
    }
}
