using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Cognition;
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
        public int AutosaveSeconds { get; init; } = 120;      // checkpoint at least this often DURING training (0 = off);
                                                              // train cycles otherwise never persist (TrainAsync saves nothing)
        public bool TrainOnFailureOnly { get; init; }         // predict each example; train ONLY the currently-wrong ones
                                                              // (skip already-correct — re-checked next cycle, so regressions re-admit)
    }

    public async Task RunAsync(GenesisEvalAppRuntime runtime, ITrainingCurriculum curriculum, Options opt, Action<CycleMetrics>? onCycle, CancellationToken ct)
    {
        Directory.CreateDirectory(opt.WorkDir);
        var chunkPath = Path.Combine(opt.WorkDir, "modular-chunk.txt");
        foreach (var op in curriculum.OperationTokens) { try { runtime.RegisterOperationToken(op); } catch { } } // route triggers (find/contains/calls)
        double baseLr = 0; try { baseLr = runtime.LearningRate; } catch { /* runtime owns LR */ }
        double lastAcc = 0, lastLoss = 0; var cycle = 0;
        var sinceSave = Stopwatch.StartNew();

        // LEARNING MODULES (Open/Closed): the loop iterates these registered mechanisms instead of hardcoding each
        // one. PRE-TRAIN batch filter (correctness gate) + per-graded-probe reactors (credit assignment → Rung-1
        // disruption → Rung-2 gradient), in this registry order. Add a mechanism = add a module here.
        var batchGate = new CorrectnessGateModule(runtime, opt.TrainOnFailureOnly, opt.ProbeRetries, opt.ProbeGateWaitMs);
        var probeModules = new IProbeOutcomeModule[]
        {
            new CreditAssignmentModule(),
            new DisruptionModule(),       // Rung 1
            new FunctionGradientModule(), // Rung 2
            new CueSelfHealModule(),      // self-heal a CUE MISROUTE (learn from a wrong route — unlearn a bad cue→route edge)
        };

        while (!ct.IsCancellationRequested)
        {
            var sw = Stopwatch.StartNew();

            // Cycle-level LR anneal: full step until near the bar, then shrink (only near-the-top oscillation).
            if (baseLr > 0) { try { runtime.LearningRate = baseLr * AnnealFactor(lastAcc, opt.MasteryBar); } catch { } }

            var generated = curriculum.NextTrainBatch();                   // top-level decides WHAT trains (mix vs focus)
            if (generated.Count == 0) { try { await Task.Delay(500, ct); } catch { break; } continue; }

            // CORRECTNESS-GATED TRAINING (the anti-erosion gate, opt-in): keep only the currently-WRONG examples so a
            // mastered skill trains nothing and gradient pours into the failing ones. Pass-through when disabled.
            var batch = await batchGate.FilterAsync(generated, ct);

            // Train ONLY the wrong set. When everything's correct, skip the gradient step entirely (carry the last
            // loss) but STILL fall through to the probe/grade below — that re-check is what catches a regression.
            var loss = lastLoss;
            if (batch.Count > 0)
            {
                try { File.WriteAllLines(chunkPath, batch.Select(e => $"{e.Input} => {e.Output}")); } catch { }
                try { loss = (await runtime.TrainAsync(chunkPath, epochs: 1)).AverageLoss.TotalLoss; }
                catch (OperationCanceledException) { break; }
                catch (Exception) { try { await Task.Delay(2000, ct); } catch { break; } continue; } // OOM/autograd handled inside TrainAsync; backstop
                lastLoss = loss;
            }
            cycle++;

            // PER-CYCLE SPACE MAINTENANCE — the LIVE eviction pass (relevance-decay discharge + the hard active-concept
            // cap). The legacy orchestrator runs this per epoch; the modular gym trains via runtime.TrainAsync (no held
            // trainer) so it never ran here, letting corpus vocab grow the space unbounded. Drive it through the runtime
            // each cycle so the space stays bounded. Best-effort: a maintenance hiccup must never stop training.
            try { runtime.MaintainPlatonicSpace(); } catch { }

            // PER-UNIT probe + grade (read AFTER NextTrainBatch so a focused curriculum has set its active set):
            // each curriculum advances its OWN difficulty/mastery on its OWN score.
            var units = curriculum.Units;
            var capture = cycle % Math.Max(1, opt.SampleEvery) == 0;
            var samples = new List<ProbeSample>();
            // RESERVOIR sample so the displayed examples are REPRESENTATIVE of the whole probe distribution
            // (every type — arithmetic, fold, predicate, seq, expression, function, retrieval — has an equal
            // chance), not just the first N (which were always the pool probes a curriculum lists first).
            var sampleRng = new Random();
            var sampleSeen = 0;
            double aggQuality = 0, aggConf = 0; int aggN = 0, aggPlatonic = 0;
            var unitProgress = new List<UnitProgress>(); // UNIFIED per-lesson progress this cycle
            foreach (var unit in units)
            {
                double uQ = 0, uConf = 0; int uN = 0, uPlat = 0;
                // FOUNDATION readiness (Open/Closed): a unit whose success is a PROPERTY OF THE SPACE self-assesses
                // (e.g. the prebake's function-word separation). Grade by that readiness and skip surface probe-grading
                // — the production field won't echo a content word, so its surface probes would read 0% even when warm.
                if (unit.SelfAssess(runtime) is double selfAcc)
                {
                    var sa = Math.Clamp(selfAcc, 0.0, 1.0);
                    unit.RecordCycle(new CycleGrade(sa, 1.0, sa));
                    unitProgress.Add(new UnitProgress(unit.Name, unit.Difficulty, sa, unit.IsMastered));
                    aggQuality += sa; aggConf += sa; aggN += 1; aggPlatonic += 1;
                    continue;
                }
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
                    var pq = GenesisGrader.Quality(output, probe.Allowed, probe.RequiredDepth, neural, probe.RequirePlatonic, probe.AnswerVocabulary, probe.SurfaceStrict);
                    uQ += pq; if (!neural) uPlat++; uConf += res.Result.PlatonicConfidence;
                    // VALUE-correct ignoring route — distinguishes "wrong answer" from "right answer via the NEURAL
                    // fallback" (which platonic-mastery grading scores 0). Gates BOTH the display AND the Rung-1
                    // disruption: we must NOT repel a value-correct answer that merely routed neural.
                    var valueCorrect = pq > 0 || GenesisGrader.Quality(output, probe.Allowed, probe.RequiredDepth, neural, false, probe.AnswerVocabulary, probe.SurfaceStrict) > 0;
                    // LEARNING MODULES react to this graded probe in registry order: credit-assignment (strengthen/
                    // weaken the edges the answer used) → Rung-1 disruption (repel a value-wrong answer off the anchor)
                    // → Rung-2 function gradient. Each self-gates; adding a probe-stage mechanism is adding a module.
                    var outcome = new ProbeOutcome(runtime, probe, output, neural, valueCorrect, pq,
                        res.Result.Evidence ?? Array.Empty<PlatonicEvidence>());
                    foreach (var m in probeModules)
                        try { m.OnGradedProbe(in outcome); } catch { }
                    if (capture)
                    {
                        var sample = new ProbeSample(probe.Query, output, pq > 0, !neural, string.Join(" / ", probe.Allowed), valueCorrect);
                        sampleSeen++;
                        if (samples.Count < opt.SampleCount) samples.Add(sample);
                        else { var j = sampleRng.Next(sampleSeen); if (j < opt.SampleCount) samples[j] = sample; } // reservoir
                    }
                }
                var uAcc = uN > 0 ? uQ / uN : 0.0;
                unit.RecordCycle(new CycleGrade(uAcc, uN > 0 ? (double)uPlat / uN : 0.0, uN > 0 ? uConf / uN : 0.0));
                if (uN > 0) unitProgress.Add(new UnitProgress(unit.Name, unit.Difficulty, uAcc, unit.IsMastered));
                aggQuality += uQ; aggConf += uConf; aggN += uN; aggPlatonic += uPlat;
            }
            var acc = aggN > 0 ? aggQuality / aggN : 0.0;
            var purity = aggN > 0 ? (double)aggPlatonic / aggN : 0.0;
            var conf = aggN > 0 ? aggConf / aggN : 0.0;
            lastAcc = acc;
            // Collect the learning modules' activity counters into one dict (telemetry surface) — namespaced by module.
            var moduleMetrics = new Dictionary<string, double>();
            foreach (var kv in batchGate.Metrics()) moduleMetrics[$"{batchGate.Name}.{kv.Key}"] = kv.Value;
            foreach (var m in probeModules)
                foreach (var kv in m.Metrics()) moduleMetrics[$"{m.Name}.{kv.Key}"] = kv.Value;
            IReadOnlyList<long>? opBalance = null;
            try { opBalance = runtime.OpClassBalance; } catch { }
            onCycle?.Invoke(new CycleMetrics(cycle, curriculum.Difficulty, loss, acc, purity, conf, sw.Elapsed.TotalSeconds, samples,
                TrainedCount: batch.Count, GeneratedCount: generated.Count, OpClassBalance: opBalance, ModuleMetrics: moduleMetrics,
                Units: unitProgress));

            // Periodic AUTOSAVE during training — TrainAsync(savePath:null) persists NOTHING, so a long unattended
            // run otherwise risked losing every cycle on a crash. Time-based so the cadence is stable regardless of
            // cycle length; best-effort (a failed save never stops training). Writes the FULL checkpoint (NN +
            // platonic companion) to the gym's autosave path through the model gate (safe vs. REPL/training).
            if (opt.AutosaveSeconds > 0 && sinceSave.Elapsed.TotalSeconds >= opt.AutosaveSeconds)
            {
                try { await runtime.SaveAsync(runtime.AutoCheckpointPath); } catch { }
                sinceSave.Restart();
            }

            // Throttle: rest this % of the cycle's own time (floor keeps the UI + model gate responsive).
            var pct = opt.ThrottlePercent?.Invoke() ?? 0;
            var restMs = Math.Min(Math.Max(opt.MinRestMs, sw.Elapsed.TotalMilliseconds * (pct / 100.0)), 600000);
            try { await Task.Delay(TimeSpan.FromMilliseconds(restMs), ct); } catch { break; }
        }

        // Final save on pause/stop — the loop only autosaves on the interval, so without this a stop between
        // intervals would drop the most recent cycles. (SaveAsync ignores the cancelled ct and still runs.)
        if (opt.AutosaveSeconds > 0) { try { await runtime.SaveAsync(runtime.AutoCheckpointPath); } catch { } }

        if (baseLr > 0) { try { runtime.LearningRate = baseLr; } catch { } }
    }

    // Anti-oscillation LR curve (consolidated from the bootstrap regime + daemon): full LR until just below the
    // bar (a sub-target plateau needs MORE step, not less), then damp as it nears/holds the top.
    private static double AnnealFactor(double acc, double target) => MasteryAnneal.Factor(acc, target);
}
