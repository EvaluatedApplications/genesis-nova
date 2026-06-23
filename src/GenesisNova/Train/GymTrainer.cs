using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>The individually-trainable gym "muscles". Each is a self-contained example generator; the host can
/// enable any subset (one checkbox each) so the model can FOCUS on a single problem at a time instead of the
/// whole mix. <see cref="GymTrainer"/> generates + probes only the enabled skills.</summary>
public enum GymSkill
{
    Synonym,           // "a synonym for big" → large/huge/giant (REAL facts; many correct)
    Category,          // "what kind of thing is apple" → fruit
    NumberWord,        // "5 in words" → five  /  "five as a number" → 5  (surface-strict)
    Add, Subtract, Multiply,
    FoldAdd,           // a + b + c            (chaining)
    FoldMultiply,      // a x b x c
    Expression,        // a x b + c            (mixed operators)
    Predicate,         // "x compared to y" → greater/less/equal
    WordedAdd,         // "what is x plus y" → sum
    ArithToWord,       // "x plus y in words" → sum-as-word (surface-strict)
    FunctionInduction, // "fn 2 is 4 fn 5 is 10 fn 3 is" → 6  (few-shot in-context rule)
}

/// <summary>
/// Procedural "gym" curriculum that trains the GRU's core skills on SYNTHETIC data. GENERAL (no app/domain
/// specifics, no runtime, no file I/O) so BOTH the desktop app and the headless CLI can host it; the host drives
/// train/probe against its <c>GenesisEvalAppRuntime</c> and persists metrics/level/throttle.
///
/// Each <see cref="GymSkill"/> is a distinct generator exercising a distinct GRU "muscle". Construct with a SUBSET
/// of skills to focus training on just those (the UI maps one checkbox per skill); the default is all of them.
/// Two lesson FAMILIES: ASSOCIATIVE (REAL facts — synonym groups / item→category / digit↔word — test BUILD +
/// RETENTION; see <see cref="GymLanguageFacts"/>) and GENERALIZING (arithmetic / fold / expression / predicate /
/// few-shot function — FRESH instances test generalization). The always-on edit/route/perception heads ride along.
///
/// Difficulty is an UNBOUNDED, MASTERY-GATED <see cref="Level"/>: hold accuracy at/above <see cref="MasteryBar"/>
/// for <see cref="StableCyclesToAdvance"/> cycles → the level climbs. A focused (single-skill) gym levels that one
/// muscle independently.
/// </summary>
public sealed class GymTrainer : ITrainingCurriculum
{
    private readonly Random _rng;
    private readonly GymSkill[] _skills;   // the ENABLED muscles (default: all)
    // Active REAL-FACT pools this level (coverage grows with the level; STABLE — a fact never changes, so no
    // cross-level contradiction). Synonym groups, item→category facts, digit↔word pairs. Real words, so the
    // prompt is self-explanatory and what the model learns is TRUE/transferable (not nonce noise).
    private readonly List<string[]> _synGroups = new();
    private string[] _synWords = Array.Empty<string>();              // all active synonym words — competing-answer vocab
    private readonly List<(string Item, string Cat)> _cats = new();
    private string[] _catVocab = Array.Empty<string>();             // active category labels — competing-answer vocab
    private enum AssocKind { Synonym, Category, NumberWord }

    // ── Natural-language FRAMING ───────────────────────────────────────────────────────────────────────────────
    // The language skills use REAL words (see GymLanguageFacts), so the prompt is self-explanatory: "a synonym for
    // big" is a genuine question a person would answer — "a synonym for eqa5" never could. TWO clean phrasings per
    // skill (minimal, not a grab-bag, but enough that the skill isn't one locked cue token). The gym declares NO
    // op-tokens: framing words ("what/is/of/plus") are kept out of the relation graph DATA-DRIVENLY by the engine's
    // discriminative coupling (it couples each answer to its lowest-degree cue, so high-degree framing words are
    // never chosen) — no hardcoded stopword list, which doesn't scale across curricula (see
    // [[nova-find-hub-collapse-fix]]). The format cue ("in words" / "as a number") distinguishes a format request
    // from a value question, and the grader honours it (SurfaceStrict). Public so gymprobe reuses the SAME frames.
    // MANY varied phrasings per skill (was two each) that ROTATE the framing vocabulary so NO single framing word
    // ("kind"/"synonym"/"word") sits in every example — a constant framing word co-occurs with every cue and
    // becomes a retrieval-winning HUB (apple→"kind", three→"rapid"); spreading the framing across distinct wordings
    // drops each framing word's degree so the engine's discriminative coupling latches the real cue instead. Frames
    // read like genuine prompts a person would type (see [[nova-gym-prompt-naturalness]]). gymprobe reuses these.
    public static readonly string[] SynonymFrames =
    {
        "a synonym for {0}", "another word for {0}", "what means the same as {0}",
        "{0} is similar in meaning to", "give me a word close in meaning to {0}",
        "something that means about the same as {0}", "{0} is much like which other word",
        "a near match in meaning for {0}", "what else can you call {0}",
    };
    public static readonly string[] CategoryFrames =
    {
        "what kind of thing is {0}", "{0} is a kind of", "what type of thing is {0}",
        "{0} belongs to which group", "which category does {0} fall into", "what sort of thing is {0}",
        "{0} is an example of a", "{0} is classified as a", "what group does {0} belong to",
    };
    public static readonly string[] DigitWordFrames =
    {
        "{0} in words", "spell out {0}", "write {0} in words", "how do you say {0} in words", "{0} written out",
    };
    public static readonly string[] WordDigitFrames =
    {
        "{0} as a number", "the number {0}", "write {0} as a numeral", "what number is {0}", "{0} in digits",
    };
    public static readonly string[] ArithWordFrames =
    {
        "{0} plus {1} in words", "{0} + {1} in words", "write {0} plus {1} in words", "the sum of {0} and {1} in words",
    };
    public static readonly string[] PredicateVocab = { "greater", "less", "equal" }; // competing-answer set for predicate

    // OPTIONAL conversational lead-ins — surface "cruft" that ROTATES (and is often empty) so it adds VARIETY without
    // any one filler token becoming a constant correlate (the bare-beats-CONSISTENT-filler lesson, not bare-beats-
    // VARIED). Applied only to natural-language retrieval/worded frames — NEVER to bare arithmetic, which works and
    // has no framing hub. The empties weight toward "no lead-in" so most prompts stay clean.
    private static readonly string[] LeadIns =
        { "", "", "", "", "hey, ", "quick one, ", "tell me, ", "i wonder, ", "ok so, ", "remind me, ", "just curious, " };
    private string Cruft(string frame) => LeadIns[_rng.Next(LeadIns.Length)] + frame;
    private int _streak;
    private int _stuck; // consecutive sub-mastery cycles → back off a level when it gets stuck

    public int Level { get; private set; }
    public double MasteryBar { get; init; } = 0.80;
    public int StableCyclesToAdvance { get; init; } = 3;
    public int StuckCyclesToDrop { get; init; } = 3; // sub-mastery cycles before DROPPING a level (overshot difficulty)
    public int TrainPerCycle { get; init; } = 64;
    public int ProbeCount { get; init; } = 24;

    private static readonly GymSkill[] AllSkills = Enum.GetValues<GymSkill>();

    public GymTrainer(int startLevel = 1, int? seed = null, IReadOnlyList<GymSkill>? skills = null)
    {
        _rng = seed is { } s ? new Random(s) : new Random();
        _skills = skills is { Count: > 0 } ? skills.Distinct().ToArray() : AllSkills;
        Level = Math.Max(1, startLevel);
        BuildPool();
    }

    /// <summary>The enabled muscles (for telemetry / level-file keying).</summary>
    public IReadOnlyList<GymSkill> Skills => _skills;

    private void BuildPool()
    {
        // DIFFICULTY = a GROWING POOL. The active fact pool grows DETERMINISTICALLY with the level (more groups /
        // more items to keep distinct = harder retrieval), capped at the curated content. Add more facts to
        // GymLanguageFacts to raise the ceiling. STABLE — a fact never changes → no cross-level contradiction.
        // (Number-word and function-induction scale UNBOUNDED — generated, not pooled — see Generate().)
        // WIDER level-1 BASE (was 4+Level / 8+Level*2): the active pool was so small at level 1 — where every muscle
        // is stuck — that each category had too few members for attraction to pull a clean meaning-CLUSTER, forcing
        // retrieval back onto framing-word edges (the hub). More distinct members per category from the start gives
        // the geometry something to cluster. Still grows with level, still capped at the curated content.
        _synGroups.Clear();
        var nSyn = Math.Min(GymLanguageFacts.SynonymGroups.Length, 10 + Level); // +1 group per level
        for (var i = 0; i < nSyn; i++) _synGroups.Add(GymLanguageFacts.SynonymGroups[i]);
        _synWords = _synGroups.SelectMany(g => g).Distinct().ToArray();

        _cats.Clear();
        var nCat = Math.Min(GymLanguageFacts.Categories.Length, 20 + Level * 2); // +2 items per level
        for (var i = 0; i < nCat; i++) _cats.Add(GymLanguageFacts.Categories[i]);
        _catVocab = _cats.Select(c => c.Cat).Distinct().ToArray();
    }

    // One generated example for a skill, carrying BOTH the training target and the probe's grading config so the
    // train batch and the probe set stay in lock-step (same generator, no drift).
    private readonly record struct GymExample(
        string Input, string TrainOutput, IReadOnlyList<string> Allowed,
        IReadOnlyList<string>? Vocab = null, bool SurfaceStrict = false);

    private GymExample Num(string input, long value)
    {
        var s = value.ToString();
        return new GymExample(input, s, new[] { s });
    }

    // SELF-CONTAINED few-shot FUNCTION INDUCTION: each prompt DEFINES a function with 3 example pairs, then asks
    // for a FRESH operand — so the rule can only come from the in-prompt examples, NOT a memorised T(fn) (the
    // transform VARIES every call). Tests genuine in-context induction + the arithmetic homomorphism.
    private GymExample FunctionExample()
    {
        var mul = _rng.Next(2) == 0;
        var k = mul ? _rng.Next(2, 5) : _rng.Next(2, 7);   // ×{2..4} or +{2..6}, fixed within THIS prompt
        long F(long v) => mul ? v * k : v + k;
        var opMax = 8 + Level * 2;                         // operand range grows with the level (unbounded)
        var ops = new HashSet<int>();
        while (ops.Count < 4) ops.Add(_rng.Next(1, opMax + 1)); // 3 distinct demos + 1 fresh query
        var a = ops.ToArray();
        var demos = string.Join(" ", a.Take(3).Select(o => $"fn {o} is {F(o)}"));
        return Num($"{demos} fn {a[3]} is", F(a[3]));
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
        return Cruft(string.Format(frames[_rng.Next(frames.Length)], from));
    }

    /// <summary>Generate ONE fresh example for the given muscle (train target + probe grading config). The shape is
    /// DERIVED from the example's own structure/values — NO locked cue tokens (see [[nova-no-token-locking]]); the
    /// op head classifies the operator from CONTEXT, chaining emerges from multi-operand expressions.</summary>
    private GymExample Generate(GymSkill skill)
    {
        var maxN = 9 + Level * 5; // grows unbounded with level
        switch (skill)
        {
            case GymSkill.Synonym:
            {
                var g = _synGroups[_rng.Next(_synGroups.Count)];
                var cue = g[_rng.Next(g.Length)];
                var allowed = g.Where(w => w != cue).ToArray();           // ANY other group member is correct
                var trainOut = allowed[_rng.Next(allowed.Length)];        // train toward one partner
                var competing = _synWords.Where(w => !g.Contains(w)).ToArray(); // a synonym from ANOTHER group = wrong
                return new GymExample(FrameAssoc(cue, AssocKind.Synonym, true), trainOut, allowed, competing);
            }
            case GymSkill.Category:
            {
                var c = _cats[_rng.Next(_cats.Count)];
                return new GymExample(FrameAssoc(c.Item, AssocKind.Category, true), c.Cat, new[] { c.Cat }, _catVocab);
            }
            case GymSkill.NumberWord:
            {
                // GENERATIVE → UNBOUNDED pool: the number range grows with the level and the word phrase is composed
                // for ANY value (47 → "forty seven", 312 → "three hundred twelve"), so difficulty scales past the 28
                // listed single-word values. digit→word: the WORD is the answer (surface-strict, multi-token);
                // word→digit: the DIGIT is the answer (value-graded, so "147" never satisfies "47").
                var numCap = Math.Min(9999, 9 + Level * 11);
                var n = _rng.Next(0, numCap + 1);
                var word = GenesisNova.Core.NumberWordVocabulary.ToWords(n);
                return _rng.Next(2) == 0
                    ? new GymExample(FrameAssoc(n.ToString(), AssocKind.NumberWord, true), word, new[] { word }, null, SurfaceStrict: true)
                    : new GymExample(FrameAssoc(word, AssocKind.NumberWord, false), n.ToString(), new[] { n.ToString() }, null, SurfaceStrict: false);
            }
            case GymSkill.Add:
            {
                int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1);
                return Num($"{x} + {y}", x + y);
            }
            case GymSkill.Subtract:
            {
                int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, x + 1);
                return Num($"{x} - {y}", x - y);
            }
            case GymSkill.Multiply:
            {
                int x = _rng.Next(1, maxN + 1), y = _rng.Next(1, maxN + 1); // 'x' = multiply by CONTEXT (log face)
                return Num($"{x} x {y}", (long)x * y);
            }
            case GymSkill.FoldAdd:
            {
                var k = Math.Max(3, 3 + Level / 8);
                var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(0, maxN + 1)).ToArray();
                return Num(string.Join(" + ", xs), xs.Sum());
            }
            case GymSkill.FoldMultiply:
            {
                var k = Math.Max(3, 3 + Level / 8); var cap = Math.Max(2, maxN / 3);
                var xs = Enumerable.Range(0, k).Select(_ => _rng.Next(1, cap + 1)).ToArray();
                return Num(string.Join(" x ", xs), xs.Aggregate(1L, (s, v) => s * v));
            }
            case GymSkill.Expression:
            {
                int a = _rng.Next(1, Math.Max(2, maxN / 2) + 1), b = _rng.Next(1, Math.Max(2, maxN / 3) + 1);
                var prod = a * b;
                switch (_rng.Next(3)) // MIXED operators evaluated with precedence (mul then add/sub)
                {
                    case 0: { int c = _rng.Next(0, maxN + 1); return Num($"{a} x {b} + {c}", prod + c); }
                    case 1: { int c = _rng.Next(0, maxN + 1); return Num($"{c} + {a} x {b}", c + prod); }
                    default: { int c = _rng.Next(0, prod + 1); return Num($"{a} x {b} - {c}", prod - c); }
                }
            }
            case GymSkill.Predicate:
            {
                int x = _rng.Next(0, maxN + 1), y = _rng.Next(0, maxN + 1);
                var q = _rng.Next(6) switch
                {
                    0 => $"{x} compared to {y}",
                    1 => $"how does {x} compare to {y}",
                    2 => $"is {x} larger or smaller than {y}",
                    3 => $"which is bigger, {x} or {y}",
                    4 => $"put {x} next to {y}",
                    _ => $"compare {x} with {y}",
                };
                var ans = x > y ? "greater" : x < y ? "less" : "equal";
                return new GymExample(Cruft(q), ans, new[] { ans }, PredicateVocab);
            }
            case GymSkill.WordedAdd:
            {
                int x = _rng.Next(1, maxN + 1), y = _rng.Next(1, maxN + 1);
                var q = _rng.Next(5) switch
                {
                    0 => $"what is {x} plus {y}",
                    1 => $"add {x} and {y}",
                    2 => $"what do you get adding {x} and {y}",
                    3 => $"the total of {x} and {y}",
                    _ => $"{x} added to {y}",
                };
                return Num(Cruft(q), x + y);
            }
            case GymSkill.ArithToWord:
            {
                var cap = Math.Min(4999, 9 + Level * 6); // operands scale with level; ToWords composes ANY sum → word
                int x = _rng.Next(0, cap + 1), y = _rng.Next(0, cap + 1);
                var word = GenesisNova.Core.NumberWordVocabulary.ToWords(x + y);
                var q = Cruft(string.Format(ArithWordFrames[_rng.Next(ArithWordFrames.Length)], x, y));
                return new GymExample(q, word, new[] { word }, null, SurfaceStrict: true); // the WORD is the answer, not the digit
            }
            default: // FunctionInduction
                return FunctionExample();
        }
    }

    /// <summary>This cycle's training batch — fresh examples uniformly across the ENABLED muscles.</summary>
    public List<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string, string)>(TrainPerCycle);
        for (var i = 0; i < TrainPerCycle; i++)
        {
            var e = Generate(_skills[_rng.Next(_skills.Length)]);
            batch.Add((e.Input, e.TrainOutput));
        }
        return batch;
    }

    /// <summary>This cycle's GRADED probes (multi-answer, value-aware, surface-aware via <see cref="GenesisGrader"/>)
    /// — an equal share for each enabled muscle, so a focused gym probes only its one skill.</summary>
    public List<TrainingProbe> NextProbes()
    {
        var probes = new List<TrainingProbe>();
        var perSkill = Math.Max(2, ProbeCount / _skills.Length);
        foreach (var skill in _skills)
            for (var j = 0; j < perSkill; j++)
            {
                var e = Generate(skill);
                probes.Add(new TrainingProbe(e.Input, e.Allowed, 1, e.Vocab, SurfaceStrict: e.SurfaceStrict));
            }
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
    // Name reflects the enabled set so the host can key per-focus level files: "gym" (all), "gym-multiply" (one),
    // or "gym-mix" (an arbitrary subset).
    public string Name =>
        _skills.Length == AllSkills.Length ? "gym"
        : _skills.Length == 1 ? "gym-" + _skills[0].ToString().ToLowerInvariant()
        : "gym-mix";
    int ITrainingCurriculum.Difficulty => Level;
    IReadOnlyList<(string Input, string Output)> ITrainingCurriculum.NextTrainBatch() => NextTrainBatch();
    IReadOnlyList<TrainingProbe> ITrainingCurriculum.NextProbes() => NextProbes();
    // The gym declares NO op-tokens — framing words are kept out of the relation graph DATA-DRIVENLY by the
    // engine's discriminative coupling (lowest-degree cue wins), not a hardcoded stopword list. (The op-token
    // mechanism remains for genuine ROUTE-TRIGGER verbs declared by other curricula/languages.)
    void ITrainingCurriculum.RecordCycle(CycleGrade grade) => RecordCycle(grade.Accuracy);
}
