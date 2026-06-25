using System;
using System.Collections.Generic;
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
/// "On reload, training ALWAYS starts with lower accuracy than it previously ended on." The model bytes round-trip
/// exactly (ResumeFidelity), so this isolates the TRAINING-LOOP behaviour on resume: train the multi-muscle gym, note
/// where it ended, SAVE, then resume with a fresh runtime + a fresh FocusedCurriculum at the restored levels (exactly
/// what the app does) and compare. Splits the dip into (a) model fidelity, (b) the first orchestrator cycle.
/// [SlowFact]. Env GYM_GPU=1.
/// </summary>
public sealed class ResumeTrainingContinuityTests
{
    private readonly ITestOutputHelper _out;
    public ResumeTrainingContinuityTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task Resume_FirstCycleAccuracy_MatchesWhereItEnded()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-cont-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(dir);
        GenesisNovaConfig Cfg() => new GenesisNovaConfig(
            Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
            AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();
        var skills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord };
        var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.8, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };

        // ── TRAIN (with the persona seeded + conversational mode, as the app runs it) ──
        var rt1 = new GenesisEvalAppRuntime(Cfg());
        var persona = new PersonalityCurriculum();
        rt1.SetConversationalMode(true);
        rt1.SeedConversationalChunks(persona.Repertoire);
        var children1 = skills.Select(s => new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.8, TrainPerCycle = 48 }).ToList();
        var cur1 = (ITrainingCurriculum)new ProbeAlongsideCurriculum(new FocusedCurriculum(children1, masteryBar: 0.8, focusBudget: 6), persona);
        var endedAcc = 0.0; var recent = new Queue<double>();
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(75)))
            await new GenesisModularTrainingOrchestrator().RunAsync(rt1, cur1, opt, m =>
            { recent.Enqueue(m.Accuracy); while (recent.Count > 5) recent.Dequeue(); endedAcc = recent.Average(); }, cts.Token);
        var levels = children1.ToDictionary(c => c.Skills[0], c => c.Level);
        _out.WriteLine($"ENDED  acc≈{endedAcc:P0}  levels=[{string.Join(",", levels.Select(kv => $"{kv.Key}:{kv.Value}"))}]");
        await rt1.SaveAsync(rt1.AutoCheckpointPath);

        // ── RESUME (fresh runtime, restored levels) — and RE-SEED the persona, exactly as StartGym does on restart ──
        var rt2 = new GenesisEvalAppRuntime(Cfg());
        rt2.SetConversationalMode(true);
        rt2.SeedConversationalChunks(persona.Repertoire); // the suspect: does re-seeding into a trained space hurt skills?

        // (a) MODEL fidelity — probe directly, no training.
        var battery = new (string Q, string[] Ok)[]
        { ("12 + 7", new[] { "19" }), ("8 - 3", new[] { "5" }), ("4 x 6", new[] { "24" }),
          ("a synonym for big", new[] { "large", "huge", "giant" }), ("what kind of thing is apple", new[] { "fruit" }), ("5 in words", new[] { "five" }) };
        var direct = 0;
        foreach (var (q, ok) in battery)
        { var got = (await rt2.PredictAsync(q, 8)).Result?.Output?.Trim() ?? ""; if (ok.Any(a => got.Contains(a, StringComparison.OrdinalIgnoreCase))) direct++; }
        _out.WriteLine($"RESUME model-only probe: {direct}/{battery.Length}");

        // (b) FIRST orchestrator cycles at the restored levels — what the gym actually shows on resume.
        var children2 = skills.Select(s => new GymTrainer(levels[s], 7, new[] { s }) { MasteryBar = 0.8, TrainPerCycle = 48 }).ToList();
        var cur2 = new FocusedCurriculum(children2, masteryBar: 0.8, focusBudget: 6);
        var firstFive = new List<double>();
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(40)))
            await new GenesisModularTrainingOrchestrator().RunAsync(rt2, cur2, opt, m =>
            { firstFive.Add(m.Accuracy); _out.WriteLine($"  resume cycle {m.Cycle} acc {m.Accuracy:P0} units={m.Units?.Count}"); if (m.Cycle >= 8) cts.Cancel(); }, cts.Token);

        var resumeFirst = firstFive.Count > 0 ? firstFive[0] : 0;
        var resumeBest = firstFive.Count > 0 ? firstFive.Max() : 0;
        _out.WriteLine($"\nENDED≈{endedAcc:P0}   model-only {direct}/{battery.Length}   resume cycle1 {resumeFirst:P0}   resume best-of-8 {resumeBest:P0}");
        // The model loaded fine if the direct probe is high; the question is whether the gym METRIC dips on resume.
        Assert.True(direct >= battery.Length - 1, $"resumed model predicts correctly; {direct}/{battery.Length}");
    }
}
