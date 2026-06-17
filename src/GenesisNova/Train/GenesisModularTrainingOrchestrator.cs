using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Data;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// The unified, MODULAR training orchestrator — consolidates every training feature we built. It owns ONE loop
/// and composes pluggable parts:
///  • CURRICULUM (<see cref="ITrainingCurriculum"/>) decides what to train + owns difficulty/mastery (gym,
///    creators, fact sets).
///  • The ENGINE supplies per-example mechanics (route/query/plan/edit/perception/reliability heads,
///    numbers-never-relation-edges, mastered-rehearsal) inside <see cref="GenesisEvalAppRuntime.TrainAsync"/>,
///    plus batch sizing + OOM/autograd-graph recovery.
///  • <see cref="GenesisGrader"/> grades fuzzy full-list + require-platonic.
///  • A cycle-level LR anneal (<see cref="AnnealFactor"/>) and a live THROTTLE hook (0..500% backoff).
///
/// Cycle: anneal LR → curriculum.NextTrainBatch → batched TrainAsync → probe + grade → curriculum.RecordCycle →
/// emit metrics → throttle. The host (desktop app, headless runner) just provides a runtime + curriculum +
/// a metrics callback.
/// </summary>
public sealed class GenesisModularTrainingOrchestrator
{
    public sealed class Options
    {
        public double MasteryBar { get; init; } = 0.80;       // also the LR-anneal target
        public bool RequirePlatonic { get; init; } = true;    // neural-fallback correct answers score 0
        public int ProbeGateWaitMs { get; init; } = 1000;     // patience acquiring a model slot for a probe
        public int ProbeRetries { get; init; } = 5;
        public double MinRestMs { get; init; } = 150;         // floor keeps the UI responsive + the model gate free for recalls
        public Func<int>? ThrottlePercent { get; init; }      // live 0..500 — rest this % of the cycle's own time
        public int SampleEvery { get; init; } = 5;            // emit a few example Q→A diagnostics every N cycles
        public int SampleCount { get; init; } = 6;
        public string WorkDir { get; init; } = Path.Combine(Path.GetTempPath(), "genesis-nova-train");
    }

    public async Task RunAsync(GenesisEvalAppRuntime runtime, ITrainingCurriculum curriculum, Options opt, Action<CycleMetrics>? onCycle, CancellationToken ct)
    {
        Directory.CreateDirectory(opt.WorkDir);
        var chunkPath = Path.Combine(opt.WorkDir, "modular-chunk.txt");
        foreach (var op in curriculum.OperationTokens) { try { runtime.RegisterOperationToken(op); } catch { } } // route triggers (find/contains/calls)
        double baseLr = 0; try { baseLr = runtime.LearningRate; } catch { /* runtime owns LR */ }
        double lastAcc = 0; var cycle = 0;

        while (!ct.IsCancellationRequested)
        {
            var sw = Stopwatch.StartNew();

            // Cycle-level LR anneal: full step until near the bar, then shrink (only near-the-top oscillation).
            if (baseLr > 0) { try { runtime.LearningRate = baseLr * AnnealFactor(lastAcc, opt.MasteryBar); } catch { } }

            var batch = curriculum.NextTrainBatch();                       // top-level decides WHAT trains (mix vs focus)
            if (batch.Count == 0) { try { await Task.Delay(500, ct); } catch { break; } continue; }
            try { File.WriteAllLines(chunkPath, batch.Select(e => $"{e.Input} => {e.Output}")); } catch { }

            double loss;
            try { loss = (await runtime.TrainAsync(chunkPath, epochs: 1)).AverageLoss.TotalLoss; }
            catch (OperationCanceledException) { break; }
            catch (Exception) { try { await Task.Delay(2000, ct); } catch { break; } continue; } // OOM/autograd handled inside TrainAsync; this is a backstop
            cycle++;

            // PER-UNIT probe + grade (read AFTER NextTrainBatch so a focused curriculum has set its active set):
            // each curriculum advances its OWN difficulty/mastery on its OWN score.
            var units = curriculum.Units;
            var capture = cycle % Math.Max(1, opt.SampleEvery) == 0;
            var samples = new List<ProbeSample>();
            double aggQuality = 0, aggConf = 0; int aggN = 0, aggPlatonic = 0;
            foreach (var unit in units)
            {
                double uQ = 0, uConf = 0; int uN = 0, uPlat = 0;
                foreach (var probe in unit.NextProbes())
                {
                    if (ct.IsCancellationRequested) break;
                    GenesisPredictTaskData? res = null;
                    for (var i = 0; i < opt.ProbeRetries && res is null; i++)
                        res = await runtime.TryPredictAsync(probe.Query, gateWaitMilliseconds: opt.ProbeGateWaitMs);
                    if (res?.Result is null) continue;
                    uN++;
                    var neural = res.Result.UsedNeuralFallback;
                    var output = res.Result.Output ?? string.Empty;
                    var pq = GenesisGrader.Quality(output, probe.Allowed, probe.RequiredDepth, neural, probe.RequirePlatonic, probe.AnswerVocabulary);
                    uQ += pq; if (!neural) uPlat++; uConf += res.Result.PlatonicConfidence;
                    if (capture && samples.Count < opt.SampleCount) samples.Add(new ProbeSample(probe.Query, output, pq > 0, !neural));
                }
                unit.RecordCycle(new CycleGrade(uN > 0 ? uQ / uN : 0.0, uN > 0 ? (double)uPlat / uN : 0.0, uN > 0 ? uConf / uN : 0.0));
                aggQuality += uQ; aggConf += uConf; aggN += uN; aggPlatonic += uPlat;
            }
            var acc = aggN > 0 ? aggQuality / aggN : 0.0;
            var purity = aggN > 0 ? (double)aggPlatonic / aggN : 0.0;
            var conf = aggN > 0 ? aggConf / aggN : 0.0;
            lastAcc = acc;
            onCycle?.Invoke(new CycleMetrics(cycle, curriculum.Difficulty, loss, acc, purity, conf, sw.Elapsed.TotalSeconds, samples));

            // Throttle: rest this % of the cycle's own time (floor keeps the UI + model gate responsive).
            var pct = opt.ThrottlePercent?.Invoke() ?? 0;
            var restMs = Math.Min(Math.Max(opt.MinRestMs, sw.Elapsed.TotalMilliseconds * (pct / 100.0)), 600000);
            try { await Task.Delay(TimeSpan.FromMilliseconds(restMs), ct); } catch { break; }
        }

        if (baseLr > 0) { try { runtime.LearningRate = baseLr; } catch { } }
    }

    // Anti-oscillation LR curve (consolidated from the bootstrap regime + daemon): full LR until just below the
    // bar (a sub-target plateau needs MORE step, not less), then damp as it nears/holds the top.
    private static double AnnealFactor(double acc, double target) => acc < target - 0.03 ? 1.0 : acc < target + 0.02 ? 0.30 : 0.10;
}
