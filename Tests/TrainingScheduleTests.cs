using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Fast, deterministic guard on the autonomous loop's LR-anneal curve (GenesisTrainingOrchestrator).
/// The curve is the anti-oscillation fix: full steps while CLIMBING, shrink only NEAR mastery. A
/// regression here (e.g. annealing too early) starves a sub-target plateau and freezes it at ~90% —
/// the exact failure this curve exists to prevent — so it's worth pinning without any training.
/// </summary>
public sealed class TrainingScheduleTests
{
    [Theory]
    [InlineData(0.00, 1.00)]
    [InlineData(0.50, 1.00)]
    [InlineData(0.86, 1.00)]  // the plateau zone MUST keep full steps (not be starved)
    [InlineData(0.91, 1.00)]
    [InlineData(0.92, 0.30)]  // near the top → shrink to settle
    [InlineData(0.96, 0.30)]
    [InlineData(0.97, 0.10)]  // at mastery → small steps to hold
    [InlineData(1.00, 0.10)]
    public void AnnealCurve_KeepsFullStepsWhileClimbing_ShrinksOnlyNearMastery(double success, double expectedFactor)
        => Assert.Equal(expectedFactor, GenesisTrainingOrchestrator.AnnealedLearningRateFactor(success));
}
