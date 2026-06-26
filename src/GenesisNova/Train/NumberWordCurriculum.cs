using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// NUMBER-WORD curriculum — clean digit↔word pairs so the LEARNED number-word lexicon (de-hardcoding #5) reliably
/// learns its ATOMS, the way <see cref="OpCueCurriculum"/> teaches op-cues and <see cref="GrammarCurriculum"/> teaches
/// roles. The lexicon learns from the DIGIT→WORD direction (a single source VALUE in, the number-word out — see
/// <c>GenesisInferenceEngine.LearnNumberWord</c>); the WORD→DIGIT direction can't be learned from the gym's existing
/// NumberWord skill because its Cruft lead-in ("quick one,") injects the number-word "one". So this curriculum emits
/// CLEAN frames (no cruft), DIGIT→WORD dominant, covering small atoms FIRST (so scales like "hundred"/"thousand" can be
/// solved compositionally once the small atoms are known) and scaling the range with the level. Words are the codec's
/// GROUND TRUTH (reference data, allowed); the ENGINE answers from the learned lexicon. Also feeds the to-word/to-digit
/// intent cues (#4) and the decoder.
/// </summary>
public sealed class NumberWordCurriculum : ITrainingCurriculum
{
    private static readonly string[] ToWordFrames = { "{0} in words", "spell out {0}", "write {0} in words", "{0} written out" };
    private static readonly string[] ToDigitFrames = { "{0} as a number", "the number {0}", "write {0} as a numeral", "{0} in digits" };

    private readonly Random _rng;
    private readonly int _trainPerCycle;
    private readonly int _probeCount;

    public NumberWordCurriculum(int trainPerCycle = 64, int probeCount = 24, int? seed = null)
    {
        _rng = seed is int s ? new Random(s) : new Random();
        _trainPerCycle = Math.Max(16, trainPerCycle);
        _probeCount = Math.Max(8, probeCount);
    }

    public string Name => "number-words";
    public int Difficulty => Level;
    public int Level { get; private set; } = 1;
    public double MasteryBar { get; init; } = 0.85;
    public int StableCyclesToAdvance { get; init; } = 3;
    private const int MaxLevel = 6;
    public int MasteryDepth => MaxLevel;
    private int _streak;
    private bool _mastered;
    public bool IsMastered => _mastered;

    // Range grows with the level: L1 atoms (0-19), L2 tens, L3 hundreds (teaches "hundred"), L4+ thousands. Small values
    // stay in the mix at every level so the atoms a composition needs are always being reinforced.
    private long Cap => Level switch { 1 => 19, 2 => 99, 3 => 999, 4 => 9999, 5 => 99_999, _ => 999_999 };

    private string GT(long v) => GenesisNova.Core.NumberWordVocabulary.ToWords(v);

    private long Sample()
    {
        // ~60% small (<=99) so atoms are always reinforced; the rest spread across the current range.
        if (_rng.Next(10) < 6) return _rng.Next(0, 100);
        return _rng.Next(0, (int)Math.Min(int.MaxValue, Cap) + 1);
    }

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++)
        {
            var v = Sample();
            if (i % 4 != 0) // 75% DIGIT->WORD (the direction the lexicon learns from)
                batch.Add((string.Format(ToWordFrames[_rng.Next(ToWordFrames.Length)], v), GT(v)));
            else            // 25% WORD->DIGIT (trains the decoder + the to-digit intent cue)
                batch.Add((string.Format(ToDigitFrames[_rng.Next(ToDigitFrames.Length)], GT(v)), v.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount; i++)
        {
            var v = Sample();
            if (i % 2 == 0) probes.Add(new TrainingProbe(string.Format(ToWordFrames[_rng.Next(ToWordFrames.Length)], v), new[] { GT(v) }, RequiredDepth: 1, RequirePlatonic: false, SurfaceStrict: true));
            else probes.Add(new TrainingProbe(string.Format(ToDigitFrames[_rng.Next(ToDigitFrames.Length)], GT(v)), new[] { v.ToString(System.Globalization.CultureInfo.InvariantCulture) }, RequiredDepth: 1, RequirePlatonic: false));
        }
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
