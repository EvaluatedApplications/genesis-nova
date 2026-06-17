using System.Collections.Generic;
using System.Linq;
using GenesisNova.Data;

namespace GenesisNova.Train;

/// <summary>
/// One skill CREATOR as an independently-gated training unit (number-word, category, arithmetic add/sub/mul/div).
/// Generates train/probe examples at its OWN difficulty and DRIVES TO DEPTH — holding the bar ramps difficulty up
/// to <c>MaxDifficulty</c>. Wrapped by <see cref="FocusedCurriculum"/> so the focus stays on it until it masters
/// at depth, then advances to the next creator (the autonomous trainer's skill-ladder walk).
/// </summary>
public sealed class CreatorUnit : ITrainingCurriculum
{
    private readonly IExampleCreator _creator;
    private readonly int _trainCount, _probeCount, _maxDifficulty, _stable;
    private readonly double _bar;
    private int _difficulty, _streak;

    public CreatorUnit(IExampleCreator creator, int trainCount = 48, int probeCount = 12, int maxDifficulty = 3, double bar = 0.80, int stable = 3)
    {
        _creator = creator;
        _trainCount = trainCount;
        _probeCount = probeCount;
        _maxDifficulty = maxDifficulty;
        _bar = bar;
        _stable = stable;
    }

    public string Name => _creator.Name;
    public int Difficulty => _difficulty;
    public int MasteryDepth => _maxDifficulty;   // drive each creator to its max difficulty before mastering

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
        => _creator.Generate(_trainCount, _difficulty, forTraining: true);

    public IReadOnlyList<TrainingProbe> NextProbes()
        => _creator.Generate(_probeCount, _difficulty, forTraining: false)   // held-out split → tests generalization at depth
                   .Select(e => new TrainingProbe(
                       e.Input,
                       _creator.AcceptableAnswers(e.Input, e.Output, _difficulty), // ambiguous/one-to-many → full valid set
                       RequiredDepth: 1,                                            // occurrence of ANY valid answer
                       AnswerVocabulary: _creator.Grading.AnswerVocabulary,         // type-aware over-gen for words
                       RequirePlatonic: _creator.Grading.RequirePlatonic))
                   .ToList();

    public void RecordCycle(CycleGrade grade)
    {
        // DRIVE-TO-DEPTH: hold the bar → climb difficulty (to MaxDifficulty). The wrapping FocusUnit only marks
        // the creator MASTERED once it holds the bar AT the drive-to-depth difficulty, so the focus climbs first.
        if (grade.Accuracy >= _bar) { if (++_streak >= _stable && _difficulty < _maxDifficulty) { _difficulty++; _streak = 0; } }
        else _streak = 0;
    }

    /// <summary>The platonic-masterable skill creators (complexity-ordered) as individual focusable units.</summary>
    public static IReadOnlyList<ITrainingCurriculum> SkillLadder(int trainCount = 48, int probeCount = 12, int maxDifficulty = 3)
        => ExampleCreatorRegistry.All.Select(c => (ITrainingCurriculum)new CreatorUnit(c, trainCount, probeCount, maxDifficulty)).ToList();
}
