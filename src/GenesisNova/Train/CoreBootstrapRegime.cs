using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Tokenization;

namespace GenesisNova.Train;

/// <summary>Knobs for <see cref="CoreBootstrapRegime"/>.</summary>
public sealed record BootstrapRegimeOptions(
    double TargetAccuracy = 0.95,
    int MaxEpochsPerLesson = 80,
    int StabilityWindow = 3,          // consecutive epochs at/above target → "converged" (held, not a fluke)
    bool ReplayPriorLessons = true,   // replay mastered lessons each epoch (anti-forgetting)
    int MaxAnswerTokens = 4,
    int Seed = 1234,
    // Capability-mastery (autonomous mode): count an example correct ONLY when answered VIA THE
    // PLATONIC PATH (used-platonic, not neural fallback) — the mission is "learn to USE the tools",
    // and this sidesteps the loss≠structure trap where the neural path memorises the answer. Default
    // false preserves the plain-accuracy behaviour the bootstrap demos assert.
    bool RequirePlatonicPath = false,
    // Bounded replay for registry-scale catalogs: when >0, sample at most this many mastered examples
    // per epoch instead of full replay (full replay is too costly once dozens of units are mastered;
    // the runtime walker's re-verify/re-open covers the retention the sampling might miss). 0 = full.
    int MaxReplayPerEpoch = 0);

/// <summary>Per-lesson result of a bootstrap run.</summary>
public sealed record BootstrapLessonOutcome(
    string Creator,
    int Difficulty,
    double FinalAccuracy,
    int EpochsRun,
    bool Converged,
    double FinalLearningRate,
    double RetainedPriorAccuracy); // mean accuracy of all PRIOR lessons after this one (forgetting check)

/// <summary>
/// The bootstrapping TRAINING REGIME over <see cref="CoreBootstrapSuite"/>. The lessons (creators)
/// say WHAT to learn; this says HOW to train them so they CONVERGE instead of reaching ~90% and
/// oscillating.
///
/// Three first-principles mechanisms, each targeting a distinct oscillation cause:
///  1. MASTERY GATING — train ONE lesson to convergence before the next, so lessons don't fight over
///     the shared encoder mid-flight (interference oscillation).
///  2. LR ANNEALING — shrink the SGD step as a lesson's accuracy rises. A fixed step overshoots the
///     optimum once most examples are right, so the last hard ones flip-flop forever (~90% plateau);
///     smaller steps near the top let the model settle into it. This is the primary fix.
///  3. REHEARSAL — interleave a fraction of already-mastered lessons each epoch so advancing doesn't
///     erode what's done (catastrophic forgetting, which also reads as oscillation across lessons).
///
/// Convergence is DETECTED (target accuracy held for StabilityWindow epochs), not a fixed budget —
/// the epoch cap is just the give-up bound.
/// </summary>
public sealed class CoreBootstrapRegime
{
    private readonly GenesisTrainer _trainer;
    private readonly GenesisInferenceEngine _inference;
    private readonly BootstrapRegimeOptions _options;

    public CoreBootstrapRegime(
        GenesisTrainer trainer,
        GenesisInferenceEngine inference,
        BootstrapRegimeOptions? options = null)
    {
        _trainer = trainer;
        _inference = inference;
        _options = options ?? new BootstrapRegimeOptions();
    }

    // Anneal the step size by how close the lesson is to mastery — the anti-oscillation curve.
    // CRITICAL: keep FULL steps until accuracy is genuinely HIGH. Annealing too early starves the
    // learning rate while the lesson is still climbing and freezes it on a plateau (an 86%-stuck
    // lesson is NOT oscillating-near-the-optimum — it needs MORE step, not less). Only shrink once
    // we're at the top, where the user's real "90% then oscillate" overshoot actually happens.
    // Anneal RELATIVE to the lesson's own target so the curve works for both the strict (0.95) and the
    // majority-mastery (0.85, learned/stochastic) bars: full steps until just below target, shrink in the
    // approach band, smallest at/above. (For target 0.95 this reproduces the original 0.92/0.97 cuts.)
    private static double AnnealFactor(double accuracy, double target) => MasteryAnneal.Factor(accuracy, target);

    public IReadOnlyList<BootstrapLessonOutcome> Run(
        IReadOnlyList<CoreBootstrapLesson>? lessons = null,
        Action<string>? log = null)
    {
        lessons ??= CoreBootstrapSuite.Lessons;
        var baseLr = _trainer.LearningRate;
        var rng = new Random(_options.Seed);
        var outcomes = new List<BootstrapLessonOutcome>();
        var masteredPriorExamples = new List<GenesisExample>();
        var priorLessonSets = new List<List<GenesisExample>>();

        try
        {
            foreach (var lesson in lessons)
            {
                var examples = UniqueExamples(lesson);
                if (examples.Count == 0)
                    continue;

                _trainer.LearningRate = baseLr; // reset the anneal per lesson

                // Per-lesson bar: exact capabilities use the strict default; learned/stochastic ones
                // (arithmetic) declare a majority-mastery target on the lesson itself.
                var targetAccuracy = lesson.TargetAccuracy ?? _options.TargetAccuracy;

                var aboveTargetStreak = 0;
                var converged = false;
                var accuracy = 0.0;
                var epochsRun = 0;

                for (var e = 0; e < _options.MaxEpochsPerLesson; e++)
                {
                    epochsRun = e + 1;

                    // Epoch = this lesson's examples + a FULL replay of every mastered prior example.
                    // Fraction-sampling left some priors unrehearsed for many epochs so they drifted
                    // and were forgotten (100% → ~73%); full replay guarantees no prior is neglected.
                    // The bootstrap sets are tiny, so replaying all of them each epoch is cheap.
                    var epoch = new List<GenesisExample>(examples);
                    if (_options.ReplayPriorLessons && masteredPriorExamples.Count > 0)
                    {
                        if (_options.MaxReplayPerEpoch <= 0 || masteredPriorExamples.Count <= _options.MaxReplayPerEpoch)
                            epoch.AddRange(masteredPriorExamples); // full replay (small catalog)
                        else
                            for (var k = 0; k < _options.MaxReplayPerEpoch; k++) // bounded sample (scale)
                                epoch.Add(masteredPriorExamples[rng.Next(masteredPriorExamples.Count)]);
                    }
                    Shuffle(epoch, rng);

                    foreach (var ex in epoch)
                        _trainer.TrainStep(ex);

                    accuracy = Accuracy(examples);

                    // Anneal BEFORE the next epoch based on where we are now.
                    _trainer.LearningRate = baseLr * AnnealFactor(accuracy, targetAccuracy);

                    log?.Invoke($"[bootstrap] {lesson.Creator.Name} epoch {epochsRun} " +
                        $"acc={accuracy:P0} lr={_trainer.LearningRate:F4}");

                    if (accuracy >= targetAccuracy)
                    {
                        if (++aboveTargetStreak >= _options.StabilityWindow)
                        {
                            converged = true;
                            break;
                        }
                    }
                    else
                    {
                        aboveTargetStreak = 0;
                    }
                }

                var retained = priorLessonSets.Count == 0
                    ? 1.0
                    : priorLessonSets.Average(Accuracy);

                outcomes.Add(new BootstrapLessonOutcome(
                    lesson.Creator.Name, lesson.Difficulty, accuracy, epochsRun, converged,
                    _trainer.LearningRate, retained));

                masteredPriorExamples.AddRange(examples);
                priorLessonSets.Add(examples);
            }
        }
        finally
        {
            _trainer.LearningRate = baseLr; // never leave the trainer in an annealed state
        }

        return outcomes;
    }

    private List<GenesisExample> UniqueExamples(CoreBootstrapLesson lesson)
    {
        // Pull a generous count and dedupe on (Input, Output) to get the lesson's full unique set
        // deterministically (creators sample with replacement beyond their unique space).
        return lesson.Creator.Generate(512, lesson.Difficulty, forTraining: true)
            .GroupBy(p => (p.Input, p.Output))
            .Select(g => new GenesisExample(g.Key.Input, g.Key.Output, SourceCreatorName: lesson.Creator.Name))
            .ToList();
    }

    private double Accuracy(IReadOnlyList<GenesisExample> examples)
    {
        if (examples.Count == 0)
            return 1.0;
        var correct = 0;
        foreach (var ex in examples)
        {
            var g = _inference.Generate(new GenerationRequest(ex.Input, _options.MaxAnswerTokens));
            // Face-aware: a digit and its number-word both count (see AnswerEquivalence).
            var rightAnswer = GenesisNova.Core.AnswerEquivalence.Equivalent(g.Output, ex.Output);
            // Capability-mastery: optionally require the answer came from the PLATONIC path, not a
            // neural fallback that merely memorised it.
            var viaPlatonic = !_options.RequirePlatonicPath || (g.UsedPlatonicQuery && !g.UsedNeuralFallback);
            if (rightAnswer && viaPlatonic)
                correct++;
        }
        return correct / (double)examples.Count;
    }

    private static void Shuffle(List<GenesisExample> data, Random rng)
    {
        for (var i = data.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (data[i], data[j]) = (data[j], data[i]);
        }
    }
}
