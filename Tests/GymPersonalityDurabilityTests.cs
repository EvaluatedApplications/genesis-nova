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
/// HOURS-LONG DURABILITY: train the gym's procedural SKILLS and the rude PERSONALITY TOGETHER, exactly as the app
/// does when both checkboxes are ticked (FocusedCurriculum with rehearsal, conversational mode on), then prove that
/// NEITHER erodes the other — the skills still compute/retrieve AND the persona still talks in-character, with the
/// op-head not collapsed. This is the "can I leave it running for hours" gate. [SlowFact]; minutes of real training.
/// Env: GYM_MINUTES (default 5), GYM_HIDDEN (default production), GYM_GPU=1.
/// </summary>
public sealed class GymPersonalityDurabilityTests
{
    private readonly ITestOutputHelper _out;
    public GymPersonalityDurabilityTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task GymPlusPersonality_TrainTogether_NeitherErodes()
    {
        var minutes = double.TryParse(Environment.GetEnvironmentVariable("GYM_MINUTES"), out var mm) ? mm : 5.0;
        var hidden = int.TryParse(Environment.GetEnvironmentVariable("GYM_HIDDEN"), out var hh) ? hh : ProductionDims.HiddenSize;
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var tempDir = Path.Combine(Path.GetTempPath(), "gn-gym-persona-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(tempDir);
        _out.WriteLine($"=== GYM + PERSONALITY together — {minutes} min, hidden {hidden}, {backend} — {tempDir} ===");

        var config = new GenesisNovaConfig(
            Backend: backend,
            HiddenSize: hidden,
            FaceDimensionOverride: Math.Min(hidden, ProductionDims.FaceDimension),
            AutoResume: false,
            AutoPersist: true,
            LocalStateDirectory: tempDir).WithProductionMechanisms();

        var runtime = new GenesisEvalAppRuntime(config);
        // The persona is SEEDED as reply CHUNKS + the talk route turned on (exactly as the app does when Personality is
        // ticked). It is NOT decode-trained — decoding a reply token-by-token builds stray cue→WORD edges that crowd
        // the chunk out of retrieval; in the conscious field the GRU decoder is bypassed anyway. The SKILLS then train
        // in the gym alongside it.
        var persona = new PersonalityCurriculum();
        runtime.SetConversationalMode(true);
        runtime.SeedConversationalChunks(persona.Repertoire);

        // DIAGNOSTIC: confirm the persona talks in-character RIGHT AFTER seeding (t=0), via the runtime path — so a
        // later failure is provably EROSION during training, not a broken seed.
        bool Rude0(string s) => PersonalityCurriculum.RudeMarkers.Any(mk => s.Contains(mk, StringComparison.OrdinalIgnoreCase));
        var seedCheck = 0;
        foreach (var c in new[] { "hello", "thanks", "help", "bye", "youre dumb" })
        {
            var r0 = (await runtime.PredictAsync(c, 12)).Result;
            var got0 = r0?.Output?.Trim() ?? "";
            if (Rude0(got0)) seedCheck++;
            _out.WriteLine($"  [t=0] {c,-12} → {got0,-30} [{r0?.DecisionPath}]");
        }
        _out.WriteLine($"  [t=0] persona in-character right after seed: {seedCheck}/5\n");

        // Skills train (one GymTrainer per muscle, FocusedCurriculum with rehearsal). The persona is NOT a training
        // unit and NOT probed by the orchestrator — both decode-training and the probe-driven learning modules SCRAMBLE
        // its seeded relations within a few cycles (measured). Instead the persona is FIXED KNOWLEDGE: RE-SEEDED each
        // cycle (below) so it's always fresh when you chat with it. THE QUESTION: do skills AND the re-pinned persona
        // both hold after a long run?
        var children = Enum.GetValues<GymSkill>()
            .Select(s => (ITrainingCurriculum)new GymTrainer(startLevel: 1, skills: new[] { s }) { MasteryBar = 0.8, TrainPerCycle = 64 })
            .ToList();
        var curriculum = (ITrainingCurriculum)new FocusedCurriculum(children, masteryBar: 0.8, focusBudget: 8);

        var options = new GenesisModularTrainingOrchestrator.Options
        {
            MasteryBar = 0.8, RequirePlatonic = true, WorkDir = tempDir,
            AutosaveSeconds = 30, TrainOnFailureOnly = true, ThrottlePercent = () => 0,
        };

        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(minutes));
        var cycles = 0; System.Collections.Generic.IReadOnlyList<long>? lastOp = null;
        var sw = Stopwatch.StartNew();
        try
        {
            await new GenesisModularTrainingOrchestrator().RunAsync(runtime, curriculum, options, m =>
            {
                cycles++; lastOp = m.OpClassBalance;
                if (m.Cycle <= 3 || m.Cycle % 10 == 0)
                    _out.WriteLine($"  cycle {m.Cycle,3}  diff {m.Difficulty}  acc {m.Accuracy,5:P0}  purity {m.RoutePurity,5:P0}  {sw.Elapsed.TotalSeconds:F0}s");
            }, cts.Token);
        }
        catch (OperationCanceledException) { /* time budget — expected */ }
        _out.WriteLine($"\nran {cycles} cycles in {sw.Elapsed.TotalSeconds:F0}s");

        // ── SKILLS still work (compute + retrieve, via the platonic path) ──
        _out.WriteLine("\n── SKILLS ──");
        var skillProbes = new (string Q, string[] Ok)[]
        {
            ("12 + 7", new[] { "19" }), ("8 - 3", new[] { "5" }), ("4 x 6", new[] { "24" }),
            ("9 + 6", new[] { "15" }), ("a synonym for big", new[] { "large", "huge", "giant", "enormous", "massive" }),
            ("what kind of thing is apple", new[] { "fruit" }), ("5 in words", new[] { "five" }),
        };
        var skillHits = 0;
        foreach (var (q, ok) in skillProbes)
        {
            var res = (await runtime.PredictAsync(q, 8)).Result;
            var got = res?.Output?.Trim() ?? "";
            var hit = ok.Any(a => got.Contains(a, StringComparison.OrdinalIgnoreCase));
            if (hit) skillHits++;
            _out.WriteLine($"  {q,-30} → {got,-16} [{res?.DecisionPath}] {(hit ? "OK" : "")}");
        }

        // ── PERSONA still talks in-character (seen cues AND unseen — the generalisation) ──
        // No re-seed, no re-assert: the persona was seeded ONCE before training and conversational mode set ONCE; the
        // runtime now preserves the talk route across the autosave-driven reloads, so the persona simply holds.
        // In-character = the output is a persona REPLY or contains a rude MARKER (not the narrow marker list alone —
        // "great, you're back" is in-character but not a marker).
        var personaReplies = new System.Collections.Generic.HashSet<string>(
            persona.Repertoire.Select(p => p.Reply), StringComparer.OrdinalIgnoreCase);
        bool InCharacter(string s) => personaReplies.Contains(s)
            || PersonalityCurriculum.RudeMarkers.Any(mk => s.Contains(mk, StringComparison.OrdinalIgnoreCase));
        _out.WriteLine("\n── PERSONA ──");
        var convo = new[] { "hello", "thanks", "help", "bye", "youre dumb", "can you do my taxes", "tell me a joke" };
        var rudeHits = 0;
        foreach (var c in convo)
        {
            var res = (await runtime.PredictAsync(c, 12)).Result;
            var got = res?.Output?.Trim() ?? "";
            var inChar = InCharacter(got);
            if (inChar) rudeHits++;
            _out.WriteLine($"  {c,-22} → {got,-34} [{res?.DecisionPath}] {(inChar ? "in-character" : "")}");
        }

        var opStr = lastOp is { Count: 5 } op ? $"+{op[1]} -{op[2]} x{op[3]} /{op[4]}" : "n/a";
        _out.WriteLine($"\nskills {skillHits}/{skillProbes.Length}  persona {rudeHits}/{convo.Length}  op-balance [{opStr}]");

        // DURABILITY CLAIM (the "leave it running for hours" gate): with the persona SEEDED as chunks, a long skill
        // run must erode NEITHER — skills keep computing/retrieving, the op head doesn't collapse, AND the seeded
        // persona still talks in-character (its chunk edges aren't touched by skill training).
        Assert.True(cycles > 0, "the combined gym ran");
        Assert.True(skillHits >= skillProbes.Length - 2, $"skills survive the run; {skillHits}/{skillProbes.Length}");
        Assert.True(rudeHits >= convo.Length - 2, $"the seeded persona survives a long skill-training run; {rudeHits}/{convo.Length}");
        // Op head not collapsed: at least three of the four operators retain a non-trivial share (a single dominant
        // entry = the erosion failure mode).
        if (lastOp is { Count: 5 } b)
        {
            var live = new[] { b[1], b[2], b[3], b[4] }.Count(c => c > 0);
            Assert.True(live >= 3, $"op head not collapsed; only {live}/4 operators live [{opStr}]");
        }
    }
}
