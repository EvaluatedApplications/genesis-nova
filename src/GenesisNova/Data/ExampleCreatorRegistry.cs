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
        new SequenceCreator(),           // sequence:next
        new RelationCreator(),           // relation:category
    };

    public static readonly IReadOnlyList<IExampleCreator> All = new List<IExampleCreator>
    {
        // ── Active autonomous-run curriculum: PLATONIC-MASTERABLE creators only ───────────────────────
        // Pruned 2026-06-13 to the set a focused-curriculum run can actually CONVERGE on — each is
        // answerable VIA THE PLATONIC PATH (a learned relation, or the exact face homomorphism), so it
        // can reach capability-mastery instead of burning FocusBudget on a skill the platonic interface
        // cannot yet perform. Complexity order trains the corenova primitives FIRST, then arithmetic.
        new NumberWordCreator(),               // corenova:number-word-equiv (8) — equivalence relation (proven)
        new CategoryRetrievalCreator(),        // corenova:retrieval-category (10) — single-answer retrieval (proven)
        new ArithmeticCreator("add"),          // arithmetic:add (20) — face homomorphism via the GRU query path
        new ArithmeticCreator("sub"),          // arithmetic:sub (20)
        new ArithmeticCreator("mul"),          // arithmetic:mul (25)
        new ArithmeticCreator("div"),          // arithmetic:div (30)

        // ── DEFERRED (intentionally NOT in the active curriculum yet) ─────────────────────────────────
        // NB the hardcoded glider-capability creators (compare/larger/scale/twice-larger) + the templated
        // answer creator were REMOVED 2026-06-14: hand-wired block compositions lock tokens to fixed
        // meanings and the templated answer is overfitting. Capability should EMERGE from the GRU composing
        // the substrate (faces, relations-as-elements, learned-function transforms) — not premade tables.
        //   • public:* corpora (fineweb-edu, slimpajama, gutenberg, openwebmath, gsm8k, wikidata-triples)
        //     — remote hydration + windowed-text never reach platonic-path mastery, so they only burn
        //     FocusBudget. Re-add for broad LM-style training once the core is solid.
        //   • math:* / logic:* (ProceduralMathLogicDefaults) — require computation the faces do not
        //     perform (fraction simplification, boolean logic, sorting); answered neurally, never
        //     platonic-master. Need dedicated face/glider support first.
        //   • language:facts — prose-wrapped, multi-word answers; retrieval breadth to revisit once the
        //     bare retrieval primitive generalises to framed prompts.
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
