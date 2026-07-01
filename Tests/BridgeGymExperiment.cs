using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// BRIDGE A/B on the REAL gym loop, restricted to the property-shaped REASONING creators (Category + Synonym).
/// Runs the actual GenesisModularTrainingOrchestrator (as MainWindow does) bridge-OFF then bridge-ON for a bounded
/// budget and compares graded accuracy, and probes for the `field-bridge` decision path to confirm the bridge fires
/// on real curriculum queries. [SlowFact]; env BRIDGE_MINUTES (default 1.5 each side). Honest caveat: two separate
/// short runs are stochastic — read the delta as directional, and the field-bridge fire-rate as the mechanism check.
/// </summary>
public sealed class BridgeGymExperiment
{
    private readonly ITestOutputHelper _out;
    public BridgeGymExperiment(ITestOutputHelper o) => _out = o;

    private async Task<(int cycles, double best, double last, int bridgeFires)> RunGym(bool bridge, double minutes)
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-bridge-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var config = new GenesisNovaConfig(
            Backend: ComputeBackend.Cpu, HiddenSize: 256, FaceDimensionOverride: 256,
            AutoResume: false, AutoPersist: false, LocalStateDirectory: dir)
            .WithProductionMechanisms() with { BridgeReasoning = bridge };

        var runtime = new GenesisEvalAppRuntime(config);
        // REASONING creators only: item→category and word→synonym (the property-shaped gym muscles the bridge lifts).
        var skills = new[] { GymSkill.Synonym, GymSkill.Category };
        var children = skills.Select(s => (ITrainingCurriculum)new GymTrainer(startLevel: 1, skills: new[] { s })
            { MasteryBar = 0.8, TrainPerCycle = 64 }).ToList();
        var curriculum = new FocusedCurriculum(children, masteryBar: 0.8, focusBudget: 8);
        var options = new GenesisModularTrainingOrchestrator.Options
        {
            MasteryBar = 0.8, RequirePlatonic = true, WorkDir = dir, TrainOnFailureOnly = true, ThrottlePercent = () => 0,
        };

        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(minutes));
        int cycles = 0; double best = 0, last = 0;
        try
        {
            await new GenesisModularTrainingOrchestrator().RunAsync(runtime, curriculum, options, m =>
            { cycles++; best = Math.Max(best, m.Accuracy); last = m.Accuracy; }, cts.Token);
        }
        catch (OperationCanceledException) { }

        // Probe held-out-style reasoning queries; count how many route through the bridge (mechanism check).
        var bridgeFires = 0;
        foreach (var q in new[] { "what kind of thing is apple", "a synonym for big", "what kind of thing is dog",
                                  "another word for small", "what type of thing is rose" })
        {
            var res = (await runtime.PredictAsync(q, 8)).Result;
            if (res?.DecisionPath == "field-bridge") bridgeFires++;
            _out.WriteLine($"  [{(bridge ? "ON " : "OFF")}] {q,-32} → {res?.Output,-14} [{res?.DecisionPath}]");
        }
        try { Directory.Delete(dir, true); } catch { }
        return (cycles, best, last, bridgeFires);
    }

    [SlowFact]
    public async Task Bridge_GymExperiment_OnReasoningCreators_OffVsOn()
    {
        var minutes = double.TryParse(Environment.GetEnvironmentVariable("BRIDGE_MINUTES"), out var mm) ? mm : 1.5;
        _out.WriteLine($"=== BRIDGE A/B on reasoning creators (Category+Synonym) — {minutes} min/side ===");

        var off = await RunGym(false, minutes);
        var on = await RunGym(true, minutes);

        _out.WriteLine($"\nOFF : {off.cycles,3} cycles   best {off.best,5:P0}   last {off.last,5:P0}   bridge-fires {off.bridgeFires}/5");
        _out.WriteLine($"ON  : {on.cycles,3} cycles   best {on.best,5:P0}   last {on.last,5:P0}   bridge-fires {on.bridgeFires}/5");
        _out.WriteLine($"Δ best accuracy (on − off) = {(on.best - off.best):P0}   |   bridge fired on {on.bridgeFires}/5 probes when ON");

        Assert.True(off.cycles > 0 && on.cycles > 0, "both sides ran");
        Assert.Equal(0, off.bridgeFires);   // OFF must never route through the bridge (gate honoured)
    }
}
