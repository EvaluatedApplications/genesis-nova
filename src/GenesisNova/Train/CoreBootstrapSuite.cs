using GenesisNova.Data;
using GenesisNova.Data.Creators;

namespace GenesisNova.Train;

/// <summary>
/// One lesson in the focused CORE BOOTSTRAP curriculum: a creator at a fixed difficulty, trained
/// focused-first to establish a primitive tool-use behaviour BEFORE broad/noisy learning. Grounded
/// in the empirical finding that the model learns best from focused curricula and is diluted by broad
/// ones — so the primitives are scaffolded first, then general learning builds on them.
/// </summary>
/// <param name="TargetAccuracy">
/// Optional per-lesson convergence bar. Null → use the regime's default (strict, for EXACT capabilities
/// like relational lookup). A lower value declares a MAJORITY-MASTERY capability — one that is genuinely
/// LEARNED and therefore stochastic (e.g. arithmetic, now that the operation is classified by the GRU op
/// head from context rather than read off an exact symbol parser). The held-stability-window still proves
/// convergence-not-oscillation; the bar just reflects what the learned capability reliably achieves.
/// </param>
public sealed record CoreBootstrapLesson(
    IExampleCreator Creator, int Difficulty, string Demonstrates, double? TargetAccuracy = null);

/// <summary>
/// The core bootstrap suite, built TEST-FIRST: the demonstration that a behaviour CAN emerge lives in
/// <c>Tests/CoreBootstrapTests.cs</c>; this is the training REGIME that makes it reliably DO so. A
/// lesson is added here ONLY once its demonstration passes — the suite is the accumulation of
/// empirically-proven foundational regimes, ordered simplest → compositional.
/// </summary>
public static class CoreBootstrapSuite
{
    public static readonly IReadOnlyList<CoreBootstrapLesson> Lessons = new[]
    {
        // Proven by NumberWordEquivalence_CanEmerge_FromFocusedBootstrap (carrier 10/10).
        new CoreBootstrapLesson(new NumberWordCreator(), Difficulty: 0,
            Demonstrates: "number-word <-> digit equivalence (one == 1), via the learned relation edge"),

        // Proven by Retrieval_CanEmerge_AndProbesDecodeTermination (carrier 16/16, first-token 16/16).
        new CoreBootstrapLesson(new CategoryRetrievalCreator(), Difficulty: 0,
            Demonstrates: "single-answer retrieval (item -> category) via bare pairs, committed first-token"),

        // Proven by Computation_GeneralizesFromBareBootstrap_PlatonicCompression (held-out 10/10):
        // bare focused arithmetic compresses into the face homomorphism and GENERALISES to operands
        // never trained — a reusable computation, not memorised pairs.
        // Arithmetic is a LEARNED capability now (the op is classified by the GRU op head from context,
        // not read off the old exact symbol parser — removed 2026-06-14), so its convergence is a
        // MAJORITY-MASTERY bar (0.85 held), not the exact 0.95 the parser used to satisfy. This is the
        // honest "demonstrate-can-emerge" level for a stochastic pipeline (op-classify · route · decode),
        // consistent with the majority bars used elsewhere (e.g. GruQueryConstructionTests).
        new CoreBootstrapLesson(new ArithmeticCreator("add"), Difficulty: 0, TargetAccuracy: 0.85,
            Demonstrates: "compute: addition via the poly-face homomorphism, generalises to unseen operands"),
        new CoreBootstrapLesson(new ArithmeticCreator("sub"), Difficulty: 0, TargetAccuracy: 0.85,
            Demonstrates: "compute: subtraction via the poly-face homomorphism, generalises to unseen operands"),
    };
}
