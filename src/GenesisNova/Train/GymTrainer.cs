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
///  • ASSOCIATIVE (REAL facts → test BUILD + RETENTION): synonym groups (a synonym of a real word — MANY correct
///    answers accepted), item→category membership, and digit↔word — all TRUE/transferable (not nonce noise, so
///    what the model learns stays correct); coverage grows with the level. See <see cref="GymLanguageFacts"/>.
/// The always-on edit / route / perception heads ride along on every training example.
///
/// Difficulty is an UNBOUNDED, MASTERY-GATED <see cref="Level"/>: hold the mixed accuracy at/above
/// <see cref="MasteryBar"/> for <see cref="StableCyclesToAdvance"/> cycles → the level climbs (to 100+ and beyond).
/// </summary>
public sealed class GymTrainer : ITrainingCurriculum
{
    private readonly Random _rng;
    // Active REAL-FACT pools this level (coverage grows with the level; STABLE — a fact never changes, so no
    // cross-level contradiction). Synonym groups, item→category facts, digit↔word pairs. Real words, so the
    // prompt is self-explanatory and what the model learns is TRUE/transferable (not nonce noise).
    private readonly List<string[]> _synGroups = new();
    private string[] _synWords = Array.Empty<string>();              // all active synonym words — competing-answer vocab
    private readonly List<(string Item, string Cat)> _cats = new();
    private string[] _catVocab = Array.Empty<string>();             // active category labels — competing-answer vocab
    private readonly List<(string Digit, string Word)> _numWords = new();
    private enum AssocKind { Synonym, Category, NumberWord }

    // ── Natural-language FRAMING ───────────────────────────────────────────────────────────────────────────────
    // The language skills use REAL words (see GymLanguageFacts), so the prompt is self-explanatory: "a synonym for
    // big" is a genuine question a person would answer — "a synonym for eqa5" never could. TWO clean phrasings per
    // skill (minimal, not a grab-bag, but enough that the skill isn't one locked cue token). Framing/function words
    // are registered as OPERATION TOKENS (FramingTokens) so they form NO relation edges and aren't the retrieval
    // anchor — only the real CONTENT word (big / apple / 7) anchors; otherwise "what/is/for/in/words" would couple
    // to every answer and hub (see [[nova-find-hub-collapse-fix]]). The format cue ("in words" / "as a number")
    // distinguishes a format request from a value question, and the grader honours it (SurfaceStrict). Public so
    // GenesisInspect's gymprobe reuses the SAME frames.
    public static readonly string[] SynonymFrames =
        { "a synonym for {0}", "another word for {0}" };
    public static readonly string[] CategoryFrames =
        { "what kind of thing is {0}", "{0} is a kind of" };
    public static readonly string[] DigitWordFrames =
        { "{0} in words", "spell out {0}" };
    public static readonly string[] WordDigitFrames =
        { "{0} as a number", "the number {0}" };
    public static readonly string[] ArithWordFrames =
        { "{0} plus {1} in words", "{0} + {1} in words" };
    public static readonly string[] PredicateVocab = { "greater", "less", "equal" }; // competing-answer set for predicate
    // Pure framing/function words (NEVER answers or operands) → op-tokens so they form no relation edges. REAL
    // content words (big/large/apple/fruit/one...) and operands are NOT here — those ARE the associations.
    public static readonly string[] FramingTokens =
    {
        "a", "an", "the", "what", "is", "of", "to", "in", "as", "for", "and", "or",
        "synonym", "another", "word", "words", "kind", "thing", "spell", "out", "number", "answer",
        "plus", "add", "compared", "compare", "how", "does", "larger", "smaller", "than", "fn",
    };
    private int _streak;
    private int _stuck; // consecutive sub-mastery cycles → back off a level when it gets stuck

    public int Level { get; private set; }
    public double MasteryBar { get; init; } = 0.80;
    public int StableCyclesToAdvance { get; init; } = 3;
    public int StuckCyclesToDrop { get; init; } = 3; // sub-mastery cycles before DROPPING a level (overshot difficulty)
    public int TrainPerCycle { get; init; } = 64;
    public int ProbeCount { get; init; } = 24;

    public GymTrainer(int startLevel = 1, int? seed = null)
    {
        _rng = seed is { } s ? new Random(s) : new Random();
        Level = Math.Max(1, startLevel);
        BuildPool();
    }

    private void BuildPool()
    {
        // SYNONYM groups (real facts) — coverage grows with the level; STABLE (a fact never changes → no
        // cross-level contradiction). Query any word → ANY OTHER member is correct (multiple-correct answers).
        _synGroups.Clear();
        var nSyn = Math.Min(GymLanguageFacts.SynonymGroups.Length, 4 + Level);
        for (var i = 0; i < nSyn; i++) _synGroups.Add(GymLanguageFacts.SynonymGroups[i]);
        _synWords = _synGroups.SelectMany(g => g).Distinct().ToArray();

        // CATEGORY membership (real facts) — item → its true category. Many items per category (a real many-to-one).
        _cats.Clear();
        var nCat = Math.Min(GymLanguageFacts.Categories.Length, 8 + Level * 2);
        for (var i = 0; i < nCat; i++) _cats.Add(GymLanguageFacts.Categories[i]);
        _catVocab = _cats.Select(c => c.Cat).Distinct().ToArray();

        // Number↔WORD equivalence — the LEGIT n≡word association (e.g. 5≡"five"), bidirectional. NOT a number↔
        // number edge (those pollute arithmetic — see nova-retention-diagnosis). Coverage grows with the level.
        _numWords.Clear();
        var wordCap = Math.Min(20, 9 + Level);
        foreach (var (value, word) in GenesisNova.Core.NumberWordVocabulary.Entries)
            if (value <= wordCap)
                _numWords.Add((value.ToString(), word));
    }

    // SELF-CONTAINED few-shot FUNCTION INDUCTION: each prompt DEFINES a function with 3 example pairs, then asks
    // for a FRESH operand — so the rule can only come from the in-prompt examples, NOT a memorised T(fn) (the
    // transform VARIES every call). "fn" is a neutral marker (registered as an op-token → forms no relation edges).
    // This makes the query sensible on its own (the demos justify the answer) and tests genuine in-context
    // induction + the arithmetic homomorphism — instead of an opaque nonce "fn0 10".
    private (string Input, string Output) NextFunction()
    {
        var mul = _rng.Next(2) == 0;
        var k = mul ? _rng.Next(2, 5) : _rng.Next(2, 7);   // ×{2..4} or +{2..6}, fixed within THIS prompt
        long F(long v) => mul ? v * k : v + k;
        var ops = new HashSet<int>();
        while (ops.Count < 4) ops.Add(_rng.Next(1, 13));   // 3 distinct demos + 1 fresh query
        var a = ops.ToArray();
        var demos = string.Join(" ", a.Take(3).Select(o => $"fn {o} is {F(o)}"));
        return ($"{demos} fn {a[3]} is", F(a[3]).ToString());
    }

    // Wrap an associative cue in a VARIED natural-language frame for its skill (no single locked cue token).
    private string FrameAssoc(string from, AssocKind kind, bool forward)
    {
        var frames = kind switch
        {
            AssocKind.Synonym => SynonymFrames,
            AssocKind.Category => CategoryFrames,
            AssocKind.NumberWord => forward ? DigitWordFrames : WordDigitFrames,
            _ => SynonymFrames,
        };
        return string.Format(frames[_rng.Next(frames.Length)], from);
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
        switch (_rng.Next(9))
        {
            case 0: { int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1); return ($"{x} + {y}", (x + y).ToString()); }
            case 1: { int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, x + 1);     return ($"{x} - {y}", (x - y).ToString()); }
            case 2: { int x = _rng.Next(1, maxN + 1), y = _rng.Next(1, maxN + 1); return ($"{x} x {y}", (x * y).ToString()); } // 'x' = multiply by CONTEXT (log face)
            // ARITH→WORD shape — compute the value, then Hop the digit to its NUMBER-WORD. Sum capped ≤ 20 so the
            // result has a word. Framed "... in words" → DISTINCT from bare add; graded surface-strict (word, not
            // digit). (The old SEQ skill was REMOVED: its prompt "x + y the answer is" already ended with "the
            // answer is", so the natural correct completion is just "N" — but it demanded the redundant "the
            // answer is N", marking a correct sum wrong. Any sensible version is either redundant with add or keeps
            // penalising the right value, so it earned its deletion.)
            case 3:
            {
                int x = _rng.Next(0, 11), y = _rng.Next(0, 20 - x + 1);
                var sum = x + y;
                var word = GenesisNova.Core.NumberWordVocabulary.Entries.FirstOrDefault(e => e.Value == sum).Word;
                var q = string.Format(ArithWordFrames[_rng.Next(ArithWordFrames.Length)], x, y);
                return (q, word ?? sum.ToString());
            }
            // CHAINING by context — a multi-operand expression of one operator: the substrate chains N compute
            // elements (one R2 compose). ≥3 operands whose sum/product equals the output → the fold shape,
            // DERIVED from the values. No "sum" word; the chain is the surface.
            case 4: { var k = Math.Max(3, 3 + Level / 8); var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(0, maxN + 1)).ToArray(); return (string.Join(" + ", xs), xs.Sum().ToString()); }
            case 5: { var k = Math.Max(3, 3 + Level / 8); var cap = Math.Max(2, maxN / 3); var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(1, cap + 1)).ToArray(); return (string.Join(" x ", xs), xs.Aggregate(1L, (s, v) => s * v).ToString()); }
            // MULTI-OPERATOR EXPRESSION — the complex chain: MIXED operators evaluated with precedence by
            // chaining compute-elements (mul then add). Each operator is disambiguated by context at inference
            // (no magic word); the shape is derived from the expression's value. Small operands, integer result.
            case 6:
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
            case 7:
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
            switch (_rng.Next(3))
            {
                case 0: // synonym: cue → a random OTHER member of its group (any synonym is a valid target)
                {
                    var g = _synGroups[_rng.Next(_synGroups.Count)];
                    var a = _rng.Next(g.Length);
                    int b; do { b = _rng.Next(g.Length); } while (b == a);
                    batch.Add((FrameAssoc(g[a], AssocKind.Synonym, true), g[b]));
                    break;
                }
                case 1: // category: item → its true category
                {
                    var c = _cats[_rng.Next(_cats.Count)];
                    batch.Add((FrameAssoc(c.Item, AssocKind.Category, true), c.Cat));
                    break;
                }
                default: // number↔word: either direction
                {
                    var nw = _numWords[_rng.Next(_numWords.Count)];
                    if (_rng.Next(2) == 0) batch.Add((FrameAssoc(nw.Digit, AssocKind.NumberWord, true), nw.Word));
                    else batch.Add((FrameAssoc(nw.Word, AssocKind.NumberWord, false), nw.Digit));
                    break;
                }
            }
        }
        for (var i = 0; i < TrainPerCycle / 2; i++)
            batch.Add(NextGeneralizing());
        // SELF-CONTAINED few-shot function-induction examples (each defines its own function in-prompt).
        for (var i = 0; i < 10; i++)
            batch.Add(NextFunction());
        return batch;
    }

    /// <summary>This cycle's probe set as GRADED probes (multi-answer, value-aware, degrees-of-correctness via
    /// <see cref="GenesisGrader"/>): associative (TRAINED pairs, framed → retrieval/retention) + generalizing
    /// (FRESH → generalization) + held-out function applications.</summary>
    public List<TrainingProbe> NextProbes()
    {
        var probes = new List<TrainingProbe>();
        for (var i = 0; i < ProbeCount / 2; i++)
        {
            switch (_rng.Next(3))
            {
                case 0: // synonym — ANY other group member is correct (multiple correct answers; need any one)
                {
                    var g = _synGroups[_rng.Next(_synGroups.Count)];
                    var cue = g[_rng.Next(g.Length)];
                    var allowed = g.Where(w => w != cue).ToArray();
                    var competing = _synWords.Where(w => !g.Contains(w)).ToArray(); // a synonym from ANOTHER group = wrong
                    probes.Add(new TrainingProbe(FrameAssoc(cue, AssocKind.Synonym, true), allowed, 1, competing));
                    break;
                }
                case 1: // category — the one true category; a competing category is penalized
                {
                    var c = _cats[_rng.Next(_cats.Count)];
                    probes.Add(new TrainingProbe(FrameAssoc(c.Item, AssocKind.Category, true), new[] { c.Cat }, 1, _catVocab));
                    break;
                }
                default: // number↔word — a FORMAT conversion, graded surface-strict (word ≠ digit)
                {
                    var nw = _numWords[_rng.Next(_numWords.Count)];
                    if (_rng.Next(2) == 0)
                        probes.Add(new TrainingProbe(FrameAssoc(nw.Digit, AssocKind.NumberWord, true), new[] { nw.Word }, 1, null, SurfaceStrict: true));
                    else
                        probes.Add(new TrainingProbe(FrameAssoc(nw.Word, AssocKind.NumberWord, false), new[] { nw.Digit }, 1, null, SurfaceStrict: true));
                    break;
                }
            }
        }
        for (var i = 0; i < ProbeCount / 2; i++)
        {
            var (q, ans) = NextGeneralizing();
            IReadOnlyList<string>? vocab = ans is "greater" or "less" or "equal" ? PredicateVocab : null;
            // arith→word ("... in words") is a FORMAT request — the WORD is the answer, not the digit value.
            var wantsWord = q.Contains("in words");
            probes.Add(new TrainingProbe(q, new[] { ans }, 1, vocab, SurfaceStrict: wantsWord));
        }
        // Held-out few-shot function induction — fresh function + fresh query each time (value-graded).
        for (var i = 0; i < 8; i++) { var (q, ans) = NextFunction(); probes.Add(new TrainingProbe(q, new[] { ans }, 1)); }
        return probes;
    }

    /// <summary>Record a cycle's mixed accuracy; ADVANCE the level when mastered (held the bar for
    /// <see cref="StableCyclesToAdvance"/> cycles), and DROP a level when STUCK (sub-mastery for
    /// <see cref="StuckCyclesToDrop"/> cycles) so an overshot difficulty backs off until it can re-master
    /// — instead of churning forever at a level it can't hold. Never drops below 1. The advance and drop
    /// counters are mutually exclusive (one resets the other), so a level near the bar oscillates gently
    /// rather than thrashing. Returns true if the level CHANGED (up or down) this call.</summary>
    public bool RecordCycle(double accuracy)
    {
        if (accuracy >= MasteryBar)
        {
            _stuck = 0;
            if (++_streak >= StableCyclesToAdvance) { Level++; _streak = 0; BuildPool(); return true; }
        }
        else
        {
            _streak = 0;
            if (++_stuck >= StuckCyclesToDrop && Level > 1) { Level--; _stuck = 0; BuildPool(); return true; }
        }
        return false;
    }

    // ── ITrainingCurriculum (lets the modular orchestrator drive the gym like any other curriculum) ──────────
    public string Name => "gym";
    int ITrainingCurriculum.Difficulty => Level;
    IReadOnlyList<(string Input, string Output)> ITrainingCurriculum.NextTrainBatch() => NextTrainBatch();
    IReadOnlyList<TrainingProbe> ITrainingCurriculum.NextProbes() => NextProbes();
    // Framing/function words are route-context only, NEVER relation participants — excluded from edge formation
    // and the retrieval anchor so they can't become hubs (see [[nova-find-hub-collapse-fix]]).
    IReadOnlyList<string> ITrainingCurriculum.OperationTokens => FramingTokens;
    void ITrainingCurriculum.RecordCycle(CycleGrade grade) => RecordCycle(grade.Accuracy);
}
