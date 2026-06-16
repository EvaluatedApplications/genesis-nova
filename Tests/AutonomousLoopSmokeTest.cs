using System;
using System.IO;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// END-TO-END smoke of the REAL autonomous loop (<see cref="GenesisEvalAppRuntime.TrainAutonomousAsync"/>):
/// focused planner → candidate-pool generation → batched training round → history, over a few rounds on
/// the pruned masterable creators. Confirms a multi-hour unattended run will actually PROGRESS and not
/// crash — the unit tests cover the planner/creators in isolation, this covers the whole runtime wiring.
/// Heavy (full runtime + GPU); isolated to a temp state dir so it never touches real checkpoints.
/// </summary>
public sealed class AutonomousLoopSmokeTest
{
    private readonly ITestOutputHelper _out;
    public AutonomousLoopSmokeTest(ITestOutputHelper output) => _out = output;

    [SlowFact]
    public async Task AutonomousLoop_RunsSeveralRounds_WithoutCrashing()
    {
        var tempState = Path.Combine(Path.GetTempPath(), "glider_auto_smoke_" + Guid.NewGuid().ToString("N"));
        var config = new GenesisNovaConfig(
            AutoResume: false,
            AutoPersist: false,
            LocalStateDirectory: tempState);

        var runtime = new GenesisEvalAppRuntime(config);
        // Focus ARITHMETIC: this exercises the GRU query heads + the edit-head's separate REINFORCE
        // backward in the batch path from round 1 — the exact combination that crashed a long run at the
        // arithmetic phase ("backward through the graph a second time"). Confirms it now runs (and, if a
        // transient autograd error occurs, the orchestrator recovers and the loop continues).
        var request = new GenesisAutonomousTrainingRequest(
            MaxRounds: 4,
            InitialSampleCount: 6,
            MinSampleCount: 4,
            MaxSampleCount: 16,
            InitialTrainCount: 4,
            MaxTrainCount: 10,
            RoundTrainBudget: 12,
            InitialDifficulty: 1,
            EnabledCreators: new[] { "arithmetic:add" });

        try
        {
            var run = await runtime.TrainAutonomousAsync(request, uiLogger: s => _out.WriteLine(s));
            Assert.NotNull(run);
            Assert.NotEmpty(run.Rounds); // the loop completed real rounds end-to-end
            _out.WriteLine($"completed rounds={run.Rounds.Count}");
        }
        finally
        {
            try { if (Directory.Exists(tempState)) Directory.Delete(tempState, recursive: true); } catch { }
        }
    }
}
