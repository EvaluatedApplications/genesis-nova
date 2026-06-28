using System;
using System.Diagnostics;
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
/// FULL GYM INTEGRATION — drive the REAL gym training loop headlessly (exactly as MainWindow does) with the LIVING
/// SELF on, time-bounded, persisting to a TEMP checkpoint, then INSPECT the resulting space in-process. This is the
/// end-to-end "is it actually working" run, not a unit probe. [SlowFact]; minutes of real training.
/// Configurable: env GYM_MINUTES (default 6). The temp checkpoint is KEPT for follow-up inspection.
/// </summary>
public sealed class GymIntegrationTests
{
    private readonly ITestOutputHelper _out;
    public GymIntegrationTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task FullGym_LivingSelf_TimeBounded_TrainsThenInspects()
    {
        var minutes = double.TryParse(Environment.GetEnvironmentVariable("GYM_MINUTES"), out var mm) ? mm : 6.0;
        var tempDir = Path.Combine(Path.GetTempPath(), "gn-gym-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(tempDir);
        _out.WriteLine($"=== FULL GYM (living self) — {minutes} min — temp checkpoint {tempDir} ===");

        // Run the ACTUAL production architecture (WithProductionMechanisms — conscious field + keep-core + the self +
        // the director-gated generative routes), exactly as MainWindow ships it. HiddenSize is env-tunable (GYM_HIDDEN)
        // so a long stability validation can trade faithful size for more cycles.
        var hidden = int.TryParse(Environment.GetEnvironmentVariable("GYM_HIDDEN"), out var hh) ? hh : ProductionDims.HiddenSize;
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var config = new GenesisNovaConfig(
            Backend: backend,
            HiddenSize: hidden,
            FaceDimensionOverride: Math.Min(hidden, ProductionDims.FaceDimension),
            AutoResume: false,
            AutoPersist: true,
            LocalStateDirectory: tempDir).WithProductionMechanisms();
        if (Environment.GetEnvironmentVariable("NOVA_BATCHED_GPU") == "1") config = config with { BatchedCloudGpu = true }; // verify the GPU cloud path under the full gym
        _out.WriteLine($"BatchedCloudGpu = {config.BatchedCloudGpu}");

        var runtime = new GenesisEvalAppRuntime(config);

        // Build the gym exactly like the app: one GymTrainer per muscle, wrapped in FocusedCurriculum.
        var skills = Enum.GetValues<GymSkill>();
        var children = skills.Select(s => (ITrainingCurriculum)new GymTrainer(startLevel: 1, skills: new[] { s }) { MasteryBar = 0.8, TrainPerCycle = 64 }).ToList();
        var curriculum = new FocusedCurriculum(children, masteryBar: 0.8, focusBudget: 8);

        var options = new GenesisModularTrainingOrchestrator.Options
        {
            MasteryBar = 0.8,
            RequirePlatonic = true,
            WorkDir = tempDir,
            AutosaveSeconds = 30,
            TrainOnFailureOnly = true,
            ThrottlePercent = () => 0,
        };

        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(minutes));
        var cycles = 0; var bestAcc = 0.0;
        var sw = Stopwatch.StartNew();
        try
        {
            await new GenesisModularTrainingOrchestrator().RunAsync(runtime, curriculum, options, m =>
            {
                cycles++; bestAcc = Math.Max(bestAcc, m.Accuracy);
                if (m.Cycle <= 3 || m.Cycle % 5 == 0)
                    _out.WriteLine($"  cycle {m.Cycle,3}  diff {m.Difficulty}  acc {m.Accuracy,5:P0}  purity {m.RoutePurity,5:P0}  trained {m.TrainedCount,3}  {sw.Elapsed.TotalSeconds:F0}s");
            }, cts.Token);
        }
        catch (OperationCanceledException) { /* time budget reached — expected */ }

        await runtime.SaveAsync(runtime.AutoCheckpointPath);
        _out.WriteLine($"\nran {cycles} cycles in {sw.Elapsed.TotalSeconds:F0}s (best cycle acc {bestAcc:P0}); saved {runtime.AutoCheckpointPath}");

        // ── INSPECT the space in-process (equivalent to GenesisInspect against the checkpoint) ──
        var diag = runtime.Diagnose();
        var geo = runtime.GeometrySummary();
        _out.WriteLine("\n── SPACE ──────────────────────────────────────────────");
        _out.WriteLine($"concepts={diag.NodeCount}  relations={diag.RelationCount}  functions={diag.FunctionElementCount}  transforms={diag.LearnedTransformCount}  foldPaths={diag.FoldPathCount}  chunks={diag.ChunkCount}");
        _out.WriteLine($"geometry: related {geo.RelatedMean:F3}  unrelated {geo.UnrelatedMean:F3}  SEPARATION {geo.Separation:F3}  (mutable {geo.MutableConcepts})");
        if (diag.TopRelations.Length > 0)
            _out.WriteLine("top relations: " + string.Join(", ", diag.TopRelations.Take(6).Select(r => $"{r.Left}~{r.Right}({r.ObservationCount})")));

        _out.WriteLine("\n── CAPABILITY PROBES (live model) ─────────────────────");
        foreach (var p in new[] { "12 + 7", "8 - 3", "4 x 6", "a synonym for big", "what kind of thing is apple", "5 in words" })
        {
            var res = (await runtime.PredictAsync(p, 8)).Result;
            var routed = res?.UsedPlatonicQuery == true && res?.UsedNeuralFallback == false ? "platonic" : "neural";
            _out.WriteLine($"  {p,-30} → {res?.Output,-16} [{res?.DecisionPath} · {routed}]");
        }

        _out.WriteLine($"\n>>> CHECKPOINT KEPT FOR INSPECTION: {tempDir}");

        // Sanity gates: it actually ran, built a space, and the numbers are finite.
        Assert.True(cycles > 0, "the gym ran at least one cycle");
        Assert.True(diag.NodeCount > 0, "the space grew during training");
        Assert.False(double.IsNaN(geo.Separation), "geometry is finite");
    }
}
