using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Fast scheduling tests for <see cref="FocusedCurriculum"/> (no training; pure rotation logic). The key
/// invariant: for UNBOUNDED units (gym muscles that never "master"), the focus must KEEP ROTATING one-at-a-time
/// forever — it must NEVER latch every unit "done" and collapse into a heavy, uncapped all-units mix (the
/// "mixed oscillates" regime the focused curriculum exists to avoid). A bounded unit that genuinely masters drops
/// out of the rotation but still rehearses as a capped rider.
/// </summary>
public sealed class FocusedCurriculumRotationTests
{
    // A controllable leaf curriculum: emits a fixed-size, name-tagged batch so a FULL-depth focus contribution
    // ("Name-0".."Name-19") is distinguishable from a CAPPED rider (≤ replayCap). Difficulty 1 == MasteryDepth 1,
    // so mastery is decided purely by the graded accuracy we feed in.
    private sealed class FakeUnit : ITrainingCurriculum
    {
        private const int BatchSize = 20;
        public FakeUnit(string name) => Name = name;
        public string Name { get; }
        public int Difficulty => 1;
        public int MasteryDepth => 1;
        public IReadOnlyList<(string Input, string Output)> NextTrainBatch() =>
            Enumerable.Range(0, BatchSize).Select(i => ($"{Name}-{i}", "o")).ToList();
        public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>();
        public void RecordCycle(CycleGrade grade) { }
    }

    // One orchestrator-style cycle: build the batch, then grade every active unit with acc(name).
    private static IReadOnlyList<(string Input, string Output)> Step(FocusedCurriculum c, Func<string, double> acc)
    {
        var batch = c.NextTrainBatch();
        foreach (var u in c.Units) u.RecordCycle(new CycleGrade(acc(u.Name), 1.0, 1.0));
        return batch;
    }

    private static Dictionary<string, int> CountByUnit(IReadOnlyList<(string Input, string Output)> batch) =>
        batch.GroupBy(e => e.Input.Split('-')[0]).ToDictionary(g => g.Key, g => g.Count());

    [Fact]
    public void UnboundedUnits_RotateFocusForever_NeverCollapseIntoUncappedMix()
    {
        const int replayCap = 5;
        var c = new FocusedCurriculum(
            new ITrainingCurriculum[] { new FakeUnit("a"), new FakeUnit("b"), new FakeUnit("c") },
            focusBudget: 4, replayCap: replayCap);

        var focusSeq = new List<string>();
        for (var i = 0; i < 40; i++)
        {
            var counts = CountByUnit(Step(c, _ => 0.0)); // nobody ever masters → all unbounded
            var full = counts.Where(kv => kv.Value > replayCap).ToList();
            Assert.Single(full);                                                     // EXACTLY one focus at depth — never the uncapped mix
            Assert.All(counts.Where(kv => kv.Key != full[0].Key),
                kv => Assert.True(kv.Value <= replayCap, $"rider {kv.Key} not capped: {kv.Value}")); // riders stay light
            focusSeq.Add(full[0].Key);
        }

        Assert.Equal(new[] { "a", "b", "c" }, focusSeq.Distinct().OrderBy(x => x).ToArray()); // every muscle got focus
        Assert.True(focusSeq.Count(x => x == "a") >= 2, "focus must wrap back around, not one-and-done"); // rotation sustains
    }

    [Fact]
    public void ManyUnboundedUnits_FocusStaysMajority_RehearsalIsBounded()
    {
        const int replayCap = 5, ridersPerCycle = 3;
        var units = Enumerable.Range(0, 10).Select(i => (ITrainingCurriculum)new FakeUnit("u" + i)).ToArray();
        var c = new FocusedCurriculum(units, focusBudget: 4, replayCap: replayCap, rehearsalRidersPerCycle: ridersPerCycle);

        // Warm up so every muscle has been introduced (rotation has swept all 10 → all are eligible riders).
        for (var i = 0; i < 60; i++) Step(c, _ => 0.0);

        for (var i = 0; i < 20; i++)
        {
            var counts = CountByUnit(Step(c, _ => 0.0));
            var full = counts.Where(kv => kv.Value > replayCap).ToList();
            Assert.Single(full);                                                        // one focus at depth
            var focusSize = full[0].Value;
            var replaySize = counts.Where(kv => kv.Key != full[0].Key).Sum(kv => kv.Value);
            Assert.True(focusSize > replaySize, $"focus {focusSize} must exceed total replay {replaySize}"); // clear majority
            Assert.True(counts.Count - 1 <= ridersPerCycle, $"too many riders this cycle: {counts.Count - 1}"); // bounded
        }
    }

    [Fact]
    public void MasteredBoundedUnit_DropsOutOfRotation_StillRehearsesCapped()
    {
        const int replayCap = 5;
        var c = new FocusedCurriculum(
            new ITrainingCurriculum[] { new FakeUnit("a"), new FakeUnit("b") },
            masteryBar: 0.80, stabilityWindow: 2, focusBudget: 100, replayCap: replayCap);

        // 'a' is the first focus; feed it the bar so it masters (streak ≥ window, depth reached), 'b' stays weak.
        for (var i = 0; i < 6; i++) Step(c, name => name == "a" ? 1.0 : 0.0);

        var counts = CountByUnit(Step(c, name => name == "a" ? 1.0 : 0.0));
        Assert.True(counts.TryGetValue("b", out var bn) && bn > replayCap, "b should hold focus once a masters");
        Assert.True(counts.TryGetValue("a", out var an) && an <= replayCap, "mastered a should ride as capped rehearsal, not refocus");
    }
}
