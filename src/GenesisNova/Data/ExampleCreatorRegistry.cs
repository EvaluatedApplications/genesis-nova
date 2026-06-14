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

        // ── Per-component glider-block regimens (2026-06-14, PROJECT_GLIDER.md §6) ──────────────────────
        // Each trains ONE reusable block via the platonic path: the input is answered by running a small
        // hand-built glider (a block composition) on the substrate (PlatonicGliderInterpreter
        // .TryResolveCapability), so RequirePlatonicForCorrect credits it and the router learns to send
        // the capability to platonic-direct. Components-first: gliders compose from already-trained
        // blocks (we deliberately do NOT train a full-glider creator — see the answer-template deferral).
        new ComparisonCreator(),               // numeric:compare      (22) — Compare block (difference-sign predicate)
        new BranchSelectCreator(),             // numeric:larger       (23) — Branch block (select on platonic compare)
        new ConstScaleCreator(),               // numeric:scale        (24) — Const block (parameterises Compute)
        new RefComposeCreator(),               // numeric:twice-larger (26) — Ref block (higher-order: glider invokes glider)

        // ── DEFERRED (intentionally NOT in the active curriculum yet) ─────────────────────────────────
        // Removed so the run converges; re-add each when its platonic path is ready:
        //   • corenova:answer-template (GliderAnswerCreator) — needs the GRU glider PLAN-heads
        //     (see PROJECT_GLIDER.md). Until then it cannot CONSTRUCT the glider via the platonic path,
        //     so it would stall the curriculum at its high complexity.
        //   • Seq + Literal blocks have NO dedicated regimen by design (PROJECT_GLIDER.md §6): Seq is the
        //     output boundary, already exercised by every multi-token creator; Literal (emit a stored
        //     chunk) is covered by the Hop retrieval primitive. A standalone creator for either would be
        //     contrived, not beneficial.
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
