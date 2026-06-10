using GenesisNova.Data.Creators;

namespace GenesisNova.Data;

/// <summary>
/// Central registry of all available example creators.
/// Sorted by EstimatedComplexity (simplest first).
/// </summary>
public static class ExampleCreatorRegistry
{
    /// <summary>
    /// Legacy creators kept for optional/manual reintegration.
    /// They are intentionally hidden from default UI and autonomous planning.
    /// </summary>
    public static readonly IReadOnlyList<IExampleCreator> Legacy = new List<IExampleCreator>
    {
        new ArithmeticCreator("add"),    // arithmetic:add
        new ArithmeticCreator("sub"),    // arithmetic:sub
        new ArithmeticCreator("mul"),    // arithmetic:mul
        new ArithmeticCreator("div"),    // arithmetic:div
        new ComparisonCreator(),         // numeric:compare
        new SequenceCreator(),           // sequence:next
        new RelationCreator(),           // relation:category
    };

    public static readonly IReadOnlyList<IExampleCreator> All = new List<IExampleCreator>
    {
        // ── Public corpora (text windows + prompt-answer) ────────────
        PublicTextCorpusDefaults.FineWebEdu,   // public:fineweb-edu
        PublicTextCorpusDefaults.SlimPajama,   // public:slimpajama
        PublicTextCorpusDefaults.Gutenberg,    // public:gutenberg
        PublicTextCorpusDefaults.OpenWebMath,  // public:openwebmath
        PublicTextCorpusDefaults.GSM8K,        // public:gsm8k
        PublicTextCorpusDefaults.WikidataTriples, // public:wikidata-triples
        ProceduralMathLogicDefaults.Fractions, // math:fractions
        ProceduralMathLogicDefaults.Percent,   // math:percent
        ProceduralMathLogicDefaults.Ratio,     // math:ratio
        ProceduralMathLogicDefaults.AlgebraSolve, // math:algebra-solve
        ProceduralMathLogicDefaults.Geometry,  // math:geometry
        ProceduralMathLogicDefaults.Boolean,   // logic:boolean
        ProceduralMathLogicDefaults.Implication, // logic:implication
        ProceduralMathLogicDefaults.Quantifiers, // logic:quantifiers
        ProceduralMathLogicDefaults.Ordering,   // logic:ordering
        ProceduralMathLogicDefaults.Syllogism,  // logic:syllogism
        
        // ── Arithmetic precision (curriculum: add → sub → mul → div) ────────────
        new ArithmeticCreator("add"),          // arithmetic:add
        new ArithmeticCreator("sub"),          // arithmetic:sub
        new ArithmeticCreator("mul"),          // arithmetic:mul (for HuggingFace training)
        new ArithmeticCreator("div"),          // arithmetic:div (for HuggingFace training)
        
        // ── Language facts (factual grounding) ────────────
        LanguageDefaults.Facts,                // language:facts
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
