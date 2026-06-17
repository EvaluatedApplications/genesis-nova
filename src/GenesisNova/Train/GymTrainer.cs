using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// Procedural "gym" curriculum that trains the GRU's core skills on SYNTHETIC data. GENERAL (no app/domain
/// specifics, no runtime, no file I/O) so BOTH the desktop app and the headless CLI can host it; the host drives
/// train/probe against its <c>GenesisEvalAppRuntime</c> and persists metrics/level/throttle.
///
/// Two lesson families, each exercising distinct GRU "muscles" / training feedback loops:
///  • GENERALIZING (they generalize → FRESH instances test generalization): arithmetic add/sub/mul
///    (face homomorphism), framed "what is a plus b" (query head), "compare a b" → greater/less/equal
///    (plan/predicate head), "sum x1..xk" (plan/fold head). Magnitude + fold-arity grow with <see cref="Level"/>.
///  • ASSOCIATIVE (arbitrary → test BUILD + RETENTION on TRAINED pairs): a STABLE pool of nonce synonym
///    (equivalence, bidirectional) + nonce item→class (category) pairs; CARDINALITY grows with the level.
/// The always-on edit / route / perception heads ride along on every training example.
///
/// Difficulty is an UNBOUNDED, MASTERY-GATED <see cref="Level"/>: hold the mixed accuracy at/above
/// <see cref="MasteryBar"/> for <see cref="StableCyclesToAdvance"/> cycles → the level climbs (to 100+ and beyond).
/// </summary>
public sealed class GymTrainer : ITrainingCurriculum
{
    private readonly Random _rng;
    private readonly List<(string Cue, string Ans, bool Relate)> _pool = new();
    // Learned UNARY FUNCTIONS: nonce names + an affine/multiplicative transform, taught via example clusters so
    // the TransformAccumulator learns T(fn) (open-vocab, selected by the learned-function route — NOT hardcoded).
    private readonly List<(string Name, bool Mul, int K)> _functions = new();
    private int _streak;

    public int Level { get; private set; }
    public double MasteryBar { get; init; } = 0.80;
    public int StableCyclesToAdvance { get; init; } = 3;
    public int TrainPerCycle { get; init; } = 64;
    public int ProbeCount { get; init; } = 24;

    public GymTrainer(int startLevel = 1, int? seed = null)
    {
        _rng = seed is { } s ? new Random(s) : new Random();
        Level = Math.Max(1, startLevel);
        BuildPool();
    }

    /// <summary>The current STABLE associative pool (grows in cardinality with the level).</summary>
    public IReadOnlyList<(string Cue, string Ans, bool Relate)> Pool => _pool;

    private void BuildPool()
    {
        _pool.Clear();
        var nAssoc = 6 + Level * 2;
        var nClasses = Math.Max(2, 2 + Level / 3);
        string Nonce(string pre, int i) => pre + i.ToString();
        for (var i = 0; i < nAssoc; i++)
        {
            _pool.Add((Nonce("eqa", i), Nonce("eqb", i), true));             // equivalence (synonym), both ways
            _pool.Add((Nonce("itm", i), Nonce("cls", i % nClasses), false)); // category item → class
        }

        // Number↔WORD equivalence — the LEGIT n≡word association (e.g. 5≡"five"), bidirectional. This is NOT a
        // number↔number edge (those pollute arithmetic — see nova-retention-diagnosis); it is exactly what the
        // number-word creator teaches. It provides the digit→word edge the arith→word route's Hop needs, plus
        // word-equivalence retrieval. Coverage grows with the level.
        var wordCap = Math.Min(20, 9 + Level);
        foreach (var (value, word) in GenesisNova.Core.NumberWordVocabulary.Entries)
            if (value <= wordCap)
                _pool.Add((value.ToString(), word, true));

        // Learned UNARY FUNCTIONS — nonce names taught via example CLUSTERS so the TransformAccumulator learns
        // T(fn) (the learned-function route then selects it open-vocab; NOT a hardcoded name). Affine/
        // multiplicative so they GENERALIZE in face space (the only class that does). Count grows with level.
        _functions.Clear();
        var nFns = Math.Min(4, 1 + Level / 2);
        for (var i = 0; i < nFns; i++)
        {
            var mul = i % 2 == 0;
            var k = mul ? 2 + (i % 3) : 2 + (i % 5);  // ×{2..4} or +{2..6}, fixed per function
            _functions.Add((Nonce("fn", i), mul, k));
        }
    }

    // One held-out application of a learned function (fresh operand → generalization test of T(fn)).
    private (string Input, string Output) NextFunction((string Name, bool Mul, int K) f)
    {
        var x = _rng.Next(1, 13);
        long y = f.Mul ? (long)x * f.K : x + f.K;
        return ($"{f.Name} {x}", y.ToString());
    }

    /// <summary>One GENERALIZING instance at the current level (structure skills; fresh = a generalization test).</summary>
    public (string Input, string Output) NextGeneralizing()
    {
        var maxN = 9 + Level * 5; // grows unbounded with level
        // NO LOCKED CUE TOKENS (see [[nova-no-token-locking]]): the skill is NEVER selected by an invented
        // control word ("sum"/"compare"/"twicelarger"). The operator symbols (+, -, x) carry no fixed meaning
        // either — the op head classifies them from CONTEXT (the SAME symbols recur across skills); here "x"
        // means multiply only because the surrounding tokens are operands. The shape is DERIVED from the
        // example's own structure/values (the resolver), so the GRU must reason about structure, not memorise
        // a word. Chaining emerges from MULTI-OPERAND expressions, not a magic cue.
        switch (_rng.Next(10))
        {
            case 0: { int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1); return ($"{x} + {y}", (x + y).ToString()); }
            case 1: { int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, x + 1);     return ($"{x} - {y}", (x - y).ToString()); }
            case 2: { int x = _rng.Next(1, maxN + 1), y = _rng.Next(1, maxN + 1); return ($"{x} x {y}", (x * y).ToString()); } // 'x' = multiply by CONTEXT (log face)
            // SEQ shape (plan-kind 7) — a Concatenate-composition: a scaffold chunk bound to a computed value.
            // The output "the answer is N" is MINED into the chunk store and the Seq route retrieves+binds it.
            // Exercises the plan/seq head + chunk mining. Derived from structure (scaffold words + sum), no cue.
            case 3: { int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1); return ($"{x} + {y}", $"the answer is {x + y}"); }
            // ARITH→WORD shape (plan-kind 4) — compute the value, then Hop the digit to its NUMBER-WORD (uses
            // the digit↔word edges from the pool). Sum capped ≤ 20 so the result has a word. Exercises the
            // format conditional + the Hop(Compute, Word) route + word-equivalence grading (5≡"five").
            case 4:
            {
                int x = _rng.Next(0, 11), y = _rng.Next(0, 20 - x + 1);
                var sum = x + y;
                var word = GenesisNova.Core.NumberWordVocabulary.Entries.FirstOrDefault(e => e.Value == sum).Word;
                return ($"{x} + {y}", word ?? sum.ToString());
            }
            // CHAINING by context — a multi-operand expression of one operator: the substrate chains N compute
            // elements (one R2 compose). ≥3 operands whose sum/product equals the output → the fold shape,
            // DERIVED from the values. No "sum" word; the chain is the surface.
            case 5: { var k = Math.Max(3, 3 + Level / 8); var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(0, maxN + 1)).ToArray(); return (string.Join(" + ", xs), xs.Sum().ToString()); }
            case 6: { var k = Math.Max(3, 3 + Level / 8); var cap = Math.Max(2, maxN / 3); var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(1, cap + 1)).ToArray(); return (string.Join(" x ", xs), xs.Aggregate(1L, (s, v) => s * v).ToString()); }
            // MULTI-OPERATOR EXPRESSION — the complex chain: MIXED operators evaluated with precedence by
            // chaining compute-elements (mul then add). Each operator is disambiguated by context at inference
            // (no magic word); the shape is derived from the expression's value. Small operands, integer result.
            case 7:
            {
                int a = _rng.Next(1, Math.Max(2, maxN / 2) + 1), b = _rng.Next(1, Math.Max(2, maxN / 3) + 1);
                var prod = a * b;
                switch (_rng.Next(3))
                {
                    case 0: { int c = _rng.Next(0, maxN + 1); return ($"{a} x {b} + {c}", (prod + c).ToString()); }
                    case 1: { int c = _rng.Next(0, maxN + 1); return ($"{c} + {a} x {b}", (c + prod).ToString()); }
                    default: { int c = _rng.Next(0, prod + 1); return ($"{a} x {b} - {c}", (prod - c).ToString()); }
                }
            }
            // Predicate — derived from the greater/less/equal OUTPUT, not an invented code. Real, VARIED
            // natural phrasings (no single token triggers it) so the GRU learns the shape from the relational
            // context + output structure.
            case 8:
            {
                int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1);
                var q = _rng.Next(3) switch
                {
                    0 => $"{x} compared to {y}",
                    1 => $"how does {x} compare to {y}",
                    _ => $"is {x} larger or smaller than {y}",
                };
                return (q, x > y ? "greater" : x < y ? "less" : "equal");
            }
            // Worded arithmetic — real language (the query head learns to ignore framing and read the operands);
            // varied framings so no single word is the trigger.
            default:
            {
                int x = _rng.Next(1, maxN + 1), y = _rng.Next(1, maxN + 1);
                var q = _rng.Next(2) == 0 ? $"what is {x} plus {y}" : $"add {x} and {y}";
                return (q, (x + y).ToString());
            }
        }
    }

    /// <summary>This cycle's training batch: half associative (build+retain), half generalizing. Bidirectional
    /// equivalence pairs emit both directions.</summary>
    public List<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string, string)>();
        for (var i = 0; i < TrainPerCycle / 2; i++)
        {
            var p = _pool[_rng.Next(_pool.Count)];
            batch.Add((p.Cue, p.Ans));
            if (p.Relate) batch.Add((p.Ans, p.Cue));
        }
        for (var i = 0; i < TrainPerCycle / 2; i++)
            batch.Add(NextGeneralizing());
        // Learned-function CLUSTERS: several examples of the SAME function so T(fn) accumulates within the
        // batch (a single example can't fit a transform). Exercises transform discovery + the learned-function
        // route + reliability-weighted routing.
        foreach (var f in _functions)
            for (var j = 0; j < 4; j++)
                batch.Add(NextFunction(f));
        return batch;
    }

    /// <summary>This cycle's probe set: associative (TRAINED pairs → retrieval/retention) + generalizing
    /// (FRESH → generalization).</summary>
    public List<(string Query, string Expected)> NextProbeBatch()
    {
        var probes = new List<(string, string)>();
        for (var i = 0; i < ProbeCount / 2; i++) { var p = _pool[_rng.Next(_pool.Count)]; probes.Add((p.Cue, p.Ans)); }
        for (var i = 0; i < ProbeCount / 2; i++) probes.Add(NextGeneralizing());
        // Held-out function applications — proves T(fn) GENERALIZED (fresh operands), not memorised.
        foreach (var f in _functions) probes.Add(NextFunction(f));
        return probes;
    }

    /// <summary>Record a cycle's mixed accuracy; advance the LEVEL when mastered (held the bar for
    /// <see cref="StableCyclesToAdvance"/> cycles). Returns true if it leveled up this call.</summary>
    public bool RecordCycle(double accuracy)
    {
        if (accuracy >= MasteryBar)
        {
            if (++_streak >= StableCyclesToAdvance) { Level++; _streak = 0; BuildPool(); return true; }
        }
        else _streak = 0;
        return false;
    }

    // ── ITrainingCurriculum (lets the modular orchestrator drive the gym like any other curriculum) ──────────
    public string Name => "gym";
    int ITrainingCurriculum.Difficulty => Level;
    IReadOnlyList<(string Input, string Output)> ITrainingCurriculum.NextTrainBatch() => NextTrainBatch();
    IReadOnlyList<TrainingProbe> ITrainingCurriculum.NextProbes() =>
        NextProbeBatch().Select(p => new TrainingProbe(p.Query, new[] { p.Expected }, 1)).ToList();
    void ITrainingCurriculum.RecordCycle(CycleGrade grade) => RecordCycle(grade.Accuracy);
}
