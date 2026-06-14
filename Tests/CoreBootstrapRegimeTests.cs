using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE BOOTSTRAPPING REGIME demonstration. The reported failure: running the bootstrap lessons,
/// accuracy reaches ~90% then OSCILLATES (fixed-LR overshoot near the optimum + lesson interference).
/// CoreBootstrapRegime adds mastery gating + LR annealing + rehearsal so each lesson CONVERGES (hits
/// target and HOLDS it for a stability window — the proof it stopped oscillating) and earlier lessons
/// are RETAINED. One end-to-end training test (own class → runs in parallel); capability/stability
/// demonstration, not exact-certainty.
/// </summary>
public sealed class CoreBootstrapRegimeTests
{
    private readonly ITestOutputHelper _out;
    public CoreBootstrapRegimeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Regime_ConvergesEachLesson_PastTheOscillationPlateau_AndRetainsPriorLessons()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null,
            trainer.FoldPathDiscovery, trainer.TransformAccumulator,
            enableDiagnosticFaceArithmeticShortcut: true);
        trainer.SetInferencePolicy(inference);

        var regime = new CoreBootstrapRegime(trainer, inference,
            new BootstrapRegimeOptions(TargetAccuracy: 0.95, MaxEpochsPerLesson: 80, StabilityWindow: 3));

        var outcomes = regime.Run(log: _out.WriteLine);

        _out.WriteLine("");
        foreach (var o in outcomes)
            _out.WriteLine($"{o.Creator,-20} acc={o.FinalAccuracy:P0} epochs={o.EpochsRun} " +
                $"converged={o.Converged} finalLr={o.FinalLearningRate:F4} retainedPrior={o.RetainedPriorAccuracy:P0}");

        Assert.NotEmpty(outcomes);

        // Each lesson CONVERGED — reached the target and HELD it for the stability window. Holding is
        // the direct refutation of "reaches 90% then oscillates": an oscillating run never strings
        // together StabilityWindow consecutive at-target epochs.
        foreach (var o in outcomes)
        {
            Assert.True(o.Converged,
                $"lesson '{o.Creator}' did not converge — reached {o.FinalAccuracy:P0} in {o.EpochsRun} epochs " +
                $"without holding target for the stability window (the oscillation signature).");
            // Annealing must actually have kicked in (final LR below the base 0.05).
            Assert.True(o.FinalLearningRate < 0.05,
                $"lesson '{o.Creator}' final LR {o.FinalLearningRate:F4} — annealing did not engage.");
        }

        // RETENTION: advancing to later lessons must not erode earlier ones (rehearsal working).
        Assert.True(outcomes[^1].RetainedPriorAccuracy >= 0.80,
            $"prior lessons eroded to {outcomes[^1].RetainedPriorAccuracy:P0} after the final lesson — forgetting.");
    }
}
