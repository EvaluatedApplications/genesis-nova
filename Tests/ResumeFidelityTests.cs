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
/// RESUME FIDELITY: reproduce "resumed training and accuracy was 0% when it was 90%+ before". Train via the gym,
/// SAVE, then build a FRESH runtime that AUTO-RESUMES from the same dir (exactly what relaunching the app does) and
/// re-measure. Inspect the space on both sides to pinpoint what is lost on the save→reload round-trip.
/// [SlowFact]; a short real gym run. Env GYM_GPU=1 to match production backend.
/// </summary>
public sealed class ResumeFidelityTests
{
    private readonly ITestOutputHelper _out;
    public ResumeFidelityTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task ResumeFromCheckpoint_PreservesAccuracy()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var hidden = int.TryParse(Environment.GetEnvironmentVariable("GYM_HIDDEN"), out var hh) ? hh : 256;
        var dir = Path.Combine(Path.GetTempPath(), "gn-resume-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(dir);
        GenesisNovaConfig Cfg() => new GenesisNovaConfig(
            Backend: backend, HiddenSize: hidden, FaceDimensionOverride: Math.Min(hidden, 512),
            AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();

        var probes = new (string Q, string[] Ok)[]
        {
            ("12 + 7", new[] { "19" }), ("8 - 3", new[] { "5" }), ("4 x 6", new[] { "24" }),
            ("a synonym for big", new[] { "large", "huge", "giant", "enormous", "massive" }),
            ("what kind of thing is apple", new[] { "fruit" }), ("5 in words", new[] { "five" }),
        };
        async Task<int> Score(GenesisEvalAppRuntime rt)
        {
            var hits = 0;
            foreach (var (q, ok) in probes)
            {
                var r = (await rt.PredictAsync(q, 8)).Result;
                var got = r?.Output?.Trim() ?? "";
                var hit = ok.Any(a => got.Contains(a, StringComparison.OrdinalIgnoreCase));
                if (hit) hits++;
                _out.WriteLine($"    {q,-30} → {got,-16} [{r?.DecisionPath}] {(hit ? "OK" : "")}");
            }
            return hits;
        }

        // ── TRAIN (fresh) ──
        var rt1 = new GenesisEvalAppRuntime(Cfg());
        var skills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord };
        var children = skills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.8, TrainPerCycle = 48 }).ToList();
        var curriculum = new FocusedCurriculum(children, masteryBar: 0.8, focusBudget: 6);
        // Autosave ON (like the real gym) — this is the path that, before fix A, made the next probe RELOAD the
        // just-written checkpoint mid-run and degrade the model. With the fix, ReloadCount must stay 0 during training.
        var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.8, WorkDir = dir, AutosaveSeconds = 15, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
        var seconds = double.TryParse(Environment.GetEnvironmentVariable("RESUME_SECONDS"), out var ss) ? ss : 90.0;
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
        {
            try
            {
                await new GenesisModularTrainingOrchestrator().RunAsync(rt1, curriculum, opt, m =>
                {
                    if (m.Cycle % 10 == 0) _out.WriteLine($"    train cycle {m.Cycle} acc {m.Accuracy:P0}");
                }, cts.Token);
            }
            catch (OperationCanceledException) { }
        }
        var d1 = rt1.Diagnose();
        _out.WriteLine($"── TRAINED (rt1) ──  concepts={d1.NodeCount} relations={d1.RelationCount} chunks={d1.ChunkCount}  self-reloads-during-training={rt1.ReloadCount}");
        Assert.Equal(0, rt1.ReloadCount); // fix A: the gym's own autosave must NOT trigger a mid-training self-reload
        var before = await Score(rt1);

        // ── SAVE to the autosave path (what the gym writes) ──
        await rt1.SaveAsync(rt1.AutoCheckpointPath);
        _out.WriteLine($"saved → {rt1.AutoCheckpointPath}");

        // ── RESUME: a brand-new runtime auto-loads from the same dir (relaunching the app) ──
        var rt2 = new GenesisEvalAppRuntime(Cfg());
        var d2 = rt2.Diagnose();
        _out.WriteLine($"── RESUMED (rt2) ──  concepts={d2.NodeCount} relations={d2.RelationCount} chunks={d2.ChunkCount}  reloads={rt2.ReloadCount}");
        var after = await Score(rt2);

        _out.WriteLine($"\nACCURACY  before {before}/{probes.Length}   after-resume {after}/{probes.Length}");
        Assert.True(before >= probes.Length - 2, $"the model trained before saving; {before}/{probes.Length}");
        Assert.True(after >= before - 1, $"RESUME must preserve accuracy; was {before}/{probes.Length}, now {after}/{probes.Length}");
    }
}
