using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// OP-CUE curriculum — teaches the WORDED names of the four operations ("the product of x and y" → multiply) as LEARNED
/// cue→operation relations, NOT a hardcoded synonym list. Each example is a CLEAN worded arithmetic frame (no operator
/// SYMBOL, correct operand ORDER) so the trainer's <c>LearnArithmeticCue</c> can infer the op from the answer and relate
/// the op-word to that operation's "∘" anchor (proven sound in isolation: 3 clean examples per op resolve every
/// synonym; see OpCueLearningDirectTest). The arithmetic itself still rides the homomorphism — only the op WORD is
/// learned here. This is the op-cue analogue of <see cref="GrammarCurriculum"/> (which feeds the role head): a focused,
/// reliable source of clean examples, instead of diluting the arithmetic skills with worded frames (where Cruft +
/// operand-order reversals + 5-way frame split starved each synonym below the learning threshold).
///
/// Why a separate curriculum rather than worded frames inside Add/Subtract/Multiply: the arithmetic skills' accuracy is
/// dominated by symbol forms (always correct via the homomorphism), so a buried worded weakness never surfaces to the
/// weakest-first scheduler — and the symbol "-"/"x" tokens make those examples EXPLICIT-operator, which LearnArithmeticCue
/// deliberately skips. Here every example is cue-only, so accuracy honestly tracks op-cue mastery.
///
/// The op-WORDS live in the DATA (this curriculum), not in any code list (the engine has no TryOpCue). They are the
/// common English names a person actually uses; the field generalises the relation, and untaught words abstain.
/// </summary>
public sealed class OpCueCurriculum : ITrainingCurriculum
{
    // Clean frames per operation — op(operand0, operand1) == answer for EVERY frame (no "subtract y from x" reversal,
    // no operator symbol). Many synonyms so each op-word ("sum"/"difference"/"product"/"quotient" + the everyday ones)
    // is taught; shared framing words ("the"/"of"/"and") appear across ALL ops and correctly abstain (competing-op).
    private static readonly string[] AddFrames = { "the sum of {0} and {1}", "the total of {0} and {1}", "{0} plus {1}", "add {0} and {1}", "{0} added to {1}", "adding {0} and {1}" };
    private static readonly string[] SubFrames = { "the difference of {0} and {1}", "{0} minus {1}", "{0} less {1}", "{0} take away {1}", "{0} decreased by {1}" };
    private static readonly string[] MulFrames = { "the product of {0} and {1}", "{0} times {1}", "multiply {0} and {1}", "{0} multiplied by {1}" };
    private static readonly string[] DivFrames = { "the quotient of {0} and {1}", "{0} divided by {1}", "divide {0} by {1}", "{0} over {1}" };

    private readonly Random _rng = new();
    private readonly int _trainPerCycle;
    private readonly int _probeCount;

    public OpCueCurriculum(int trainPerCycle = 64, int probeCount = 24)
    {
        _trainPerCycle = Math.Max(16, trainPerCycle);
        _probeCount = Math.Max(8, probeCount);
    }

    public string Name => "op-cues";
    public int Difficulty => Level;
    public int Level { get; private set; } = 1;
    public double MasteryBar { get; init; } = 0.85;
    public int StableCyclesToAdvance { get; init; } = 3;
    private const int MaxLevel = 3;
    public int MasteryDepth => MaxLevel;
    private int _streak;
    private bool _mastered;
    public bool IsMastered => _mastered;

    // Operands scale with level; division stays CLEAN (x = y*q) so the answer is integral.
    private int Cap => 9 + Level * 6;

    private (string Input, string Output) One()
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        switch (_rng.Next(4))
        {
            case 0: { int x = _rng.Next(0, Cap + 1), y = _rng.Next(0, Cap + 1); return (string.Format(AddFrames[_rng.Next(AddFrames.Length)], x, y), (x + y).ToString(inv)); }
            case 1: { int x = _rng.Next(0, Cap + 1), y = _rng.Next(0, x + 1); return (string.Format(SubFrames[_rng.Next(SubFrames.Length)], x, y), (x - y).ToString(inv)); }
            case 2: { int x = _rng.Next(1, Cap + 1), y = _rng.Next(1, Cap + 1); return (string.Format(MulFrames[_rng.Next(MulFrames.Length)], x, y), ((long)x * y).ToString(inv)); }
            default: { int y = _rng.Next(1, Math.Max(2, Cap / 2) + 1), q = _rng.Next(1, Math.Max(2, Cap / 2) + 1); int x = y * q; return (string.Format(DivFrames[_rng.Next(DivFrames.Length)], x, y), q.ToString(inv)); }
        }
    }

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++) batch.Add(One());
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        // Fresh operands each probe — once the cue→op relation is learned the homomorphism computes ANY operands, so the
        // probe genuinely tests "does the worded op resolve", not memorisation. Value-graded.
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount; i++) { var (q, a) = One(); probes.Add(new TrainingProbe(q, new[] { a }, RequiredDepth: 1, RequirePlatonic: false)); }
        return probes;
    }

    public void RecordCycle(CycleGrade grade)
    {
        if (grade.Accuracy >= MasteryBar)
        {
            if (++_streak >= StableCyclesToAdvance) { _streak = 0; if (Level < MaxLevel) Level++; else _mastered = true; }
        }
        else { _streak = 0; if (grade.Accuracy < MasteryBar - 0.15) _mastered = false; }
    }
}
