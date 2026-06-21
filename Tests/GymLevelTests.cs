using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// The gym's MASTERY-GATED level control: it ADVANCES when it holds the bar, and now DROPS a level when it gets
/// STUCK (sub-mastery for a window) so an overshot difficulty backs off instead of churning forever — and never
/// drops below level 1.
/// </summary>
public sealed class GymLevelTests
{
    [Fact] // Sub-mastery for StuckCyclesToDrop cycles backs the level off by one.
    public void DropsLevel_WhenStuck()
    {
        var gym = new GymTrainer(startLevel: 10, seed: 1) { StuckCyclesToDrop = 5, StableCyclesToAdvance = 3 };
        Assert.Equal(10, gym.Level);
        var changed = false;
        for (var i = 0; i < 5; i++) changed = gym.RecordCycle(0.2); // clearly below the 0.80 bar
        Assert.True(changed, "the level should have changed (dropped) after the stuck window");
        Assert.Equal(9, gym.Level);
    }

    [Fact] // Holding the bar for StableCyclesToAdvance cycles advances the level.
    public void AdvancesLevel_WhenMastered()
    {
        var gym = new GymTrainer(startLevel: 1, seed: 1) { StableCyclesToAdvance = 3 };
        for (var i = 0; i < 3; i++) gym.RecordCycle(0.95);
        Assert.Equal(2, gym.Level);
    }

    [Fact] // Stuck at level 1 never drops to 0.
    public void NeverDropsBelowOne()
    {
        var gym = new GymTrainer(startLevel: 1, seed: 1) { StuckCyclesToDrop = 3 };
        for (var i = 0; i < 12; i++) gym.RecordCycle(0.0);
        Assert.Equal(1, gym.Level);
    }

    [Fact] // A single good cycle resets the stuck counter — near-bar oscillation doesn't thrash the level down.
    public void GoodCycle_ResetsStuckCounter()
    {
        var gym = new GymTrainer(startLevel: 5, seed: 1) { StuckCyclesToDrop = 4, StableCyclesToAdvance = 3 };
        gym.RecordCycle(0.2); gym.RecordCycle(0.2); gym.RecordCycle(0.2); // 3 sub-bar (one short of dropping)
        gym.RecordCycle(0.95);                                            // a good cycle resets the stuck count
        gym.RecordCycle(0.2); gym.RecordCycle(0.2); gym.RecordCycle(0.2); // 3 more — still short
        Assert.Equal(5, gym.Level);                                       // no drop yet
    }
}
