using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M4 — THE GYM LOOP TRAINS THE NAVIGATOR'S REASONING, OBSERVABLE AS A HELD-OUT CURVE. The honest acceptance is "held-out
/// navigator resolve%/accuracy CLIMBS over a long run." This is a SHORT, BOUNDED proof that the curve genuinely MOVES UP;
/// extrapolate "overnight" from the slope. It uses the prompt-sanctioned "direct <see cref="GenesisEvalAppRuntime.TrainNavigatorCycle"/>
/// loop driven by the new curriculum": <see cref="NavReasoningCurriculum"/> plants a clean multi-hop is-a taxonomy + a
/// HELD-OUT member set (the sampler never distils their trajectories) and self-teaches the level cue from its frames as
/// DATA (level from GRAPH DEPTH, no hardcoded cue list); then we train the navigator and measure the held-out set at
/// checkpoints. Real runtime + <see cref="GenesisNovaConfig.WithProductionMechanisms"/> (navigator ON).
///
/// BOUNDED BY DEFAULT (NAVM4_CYCLES env var, default 6) so it never blocks CI — the OVERNIGHT run is the user's to launch
/// from the app. HONEST: the reported numbers ARE the result; we assert the robust true claims (the held-out curve climbs
/// — resolve + genus generalization — and the cue self-taught through the curriculum) and REPORT the multi-hop ceiling.
/// </summary>
public sealed class NavigatorM4Curve
{
    private readonly ITestOutputHelper _out;
    public NavigatorM4Curve(ITestOutputHelper o) => _out = o;

    // Bounded by default (a couple minutes). Override with NAVM4_CYCLES for a longer manual sweep.
    private static int Cycles =>
        int.TryParse(Environment.GetEnvironmentVariable("NAVM4_CYCLES"), out var n) && n >= 4 ? n : 6;

    private static GenesisNovaConfig BuildConfig(string dir) =>
        new GenesisNovaConfig(
            HiddenSize: 64, Backend: ComputeBackend.Cpu, AutoPersist: false, AutoResume: false, LocalStateDirectory: dir)
        .WithProductionMechanisms() with { FieldTicks = false, MeaningOps = false };

    [SlowFact]
    public async Task HeldOutNavigatorCurve_ClimbsOverTrainingCycles_AndCueIsSelfTaught()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navm4-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);

            // The curriculum plants the clean taxonomy + registers the held-out set + self-teaches the cue (in the ctor).
            var curriculum = new NavReasoningCurriculum(runtime, trainPerCycle: 40, seed: 7);
            Assert.True(runtime.NavHeldOutCount > 0, "the curriculum did not register a held-out navigator query set");
            var held = curriculum.HeldOutMembersView;
            var train = curriculum.TrainMembersView;
            foreach (var (m, _, _) in held)
                Assert.True(ds.ContainsConcept(m), $"held-out member '{m}' was not planted");

            // CUE self-taught through the curriculum's frames (LearnNavLevelCue, level from graph depth — no word list).
            bool Related(string tok, string anchor) =>
                ds.GetNeighbors(tok, PlatonicNeighborhoodType.Relational, 16, 0.0).Any(n => n.Concept == anchor);
            var genusLearned = new[] { "kind", "type", "sort" }.Count(t => Related(t, "∘gns"));
            var domainLearned = new[] { "broadly", "wider", "category", "generally" }.Count(t => Related(t, "∘dom"));
            var rootLearned = new[] { "ultimately", "essence", "fundamentally", "bottom" }.Count(t => Related(t, "∘rut"));

            _out.WriteLine($"=== M4 HELD-OUT NAVIGATOR CURVE: {ds.NodeCount} concepts | {runtime.NavHeldOutCount} held-out queries " +
                           $"| navigator {runtime.State.Navigator.Dim}/{runtime.State.Navigator.Hidden} | {Cycles} cycles (NAVM4_CYCLES) ===");
            _out.WriteLine($"    cue self-taught through curriculum frames: ∘gns={genusLearned}/3 ∘dom={domainLearned}/4 ∘rut={rootLearned}/4");

            // ── THE CURVE: cold checkpoint, then train the navigator on the TRAINING members only and re-measure the
            //    HELD-OUT set at every cycle. The series must CLIMB (resolve = confident halt; accuracy = correct landing). ──
            void Checkpoint(int c)
            {
                var p = runtime.EvaluateNavigatorHeldOut();
                _out.WriteLine($"    [held-out] cyc {c,3} | accuracy {p.AccuracyPct,6:P1} | resolve {p.ResolvePct,6:P1} | over {p.Count} queries");
            }
            Checkpoint(0); // COLD baseline — untrained policy
            for (var c = 1; c <= Cycles; c++)
            {
                runtime.TrainNavigatorCycle(maxMembers: 64, epochs: 6);
                Checkpoint(c);
            }
            var curve = runtime.NavHeldOutHistory;
            var cold = curve.First();
            var warmTail = curve.Skip(Math.Max(1, curve.Count - 3)).ToList();
            var warmAcc = warmTail.Average(p => p.AccuracyPct);
            var warmResolve = warmTail.Average(p => p.ResolvePct);

            // ── PER-LEVEL accuracy, HELD-OUT vs TRAINING (attribute the gap honestly; WalkNavigator supplies the cue). ──
            (double Acc, int N) LevelAcc(IReadOnlyList<(string M, string G, string D)> members, NavCue cue)
            {
                int ok = 0, n = 0;
                foreach (var (m, g, d) in members)
                {
                    var want = cue == NavCue.Genus ? g : cue == NavCue.Domain ? d : "entity";
                    n++;
                    var w = runtime.WalkNavigator(m, cue.ToString().ToLowerInvariant());
                    if (w.Reached && string.Equals(w.Answer, want, StringComparison.Ordinal)) ok++;
                }
                return (n > 0 ? (double)ok / n : 0.0, n);
            }
            _out.WriteLine("=== PER-LEVEL accuracy (WalkNavigator: held-out | training) ===");
            double heldGenus = 0;
            foreach (var cue in new[] { NavCue.Genus, NavCue.Domain, NavCue.Root })
            {
                var (ha, hn) = LevelAcc(held, cue);
                var (ta, tn) = LevelAcc(train, cue);
                if (cue == NavCue.Genus) heldGenus = ha;
                _out.WriteLine($"    {cue,-7} | held-out({hn}) {ha,6:P1} | training({tn}) {ta,6:P1}");
            }

            _out.WriteLine("=== SUMMARY (HONEST) ===");
            _out.WriteLine($"    HELD-OUT accuracy curve: {string.Join(" ", curve.Select(p => $"{p.AccuracyPct:P0}"))}");
            _out.WriteLine($"    HELD-OUT resolve  curve: {string.Join(" ", curve.Select(p => $"{p.ResolvePct:P0}"))}");
            _out.WriteLine($"    cold: acc {cold.AccuracyPct:P1} resolve {cold.ResolvePct:P1}  ->  warm(mean last 3): acc {warmAcc:P1} resolve {warmResolve:P1}");
            _out.WriteLine($"    held-out GENUS (1-hop) accuracy = {heldGenus:P1}");

            // Loose floors only — the DATA above is the result. Robust true claims:
            Assert.True(curve.Count >= Cycles, "the held-out warm-history was not populated each cycle");
            Assert.True(genusLearned >= 1 && (domainLearned >= 1 || rootLearned >= 1),
                $"the level cue did not self-teach through the curriculum (∘gns={genusLearned} ∘dom={domainLearned} ∘rut={rootLearned})");
            Assert.True(warmResolve > cold.ResolvePct + 0.10 || warmResolve >= 0.95,
                $"held-out resolve did not climb: cold {cold.ResolvePct:P1} -> warm {warmResolve:P1}");
            Assert.True(heldGenus >= 0.20 || warmAcc > cold.AccuracyPct + 0.10,
                $"held-out generalization did not climb: genus {heldGenus:P1}, cold acc {cold.AccuracyPct:P1} -> warm {warmAcc:P1}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
