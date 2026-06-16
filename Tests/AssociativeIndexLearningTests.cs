using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EMPIRICAL INVESTIGATION (2026-06-16): how does the platonic stack LEARN a flat associative INDEX —
/// many `find &lt;anchor&gt; =&gt; key` associations that must be RETRIEVED VIA THE PLATONIC PATH at scale,
/// without catastrophic forgetting? This is an INSTRUMENT, not a pass/fail gate: it trains controlled
/// synthetic indexes and reports retrieval% / routed% / output-diversity (collapse detector) as a function
/// of N (scale) and training regime (BROAD interleaved vs SEQUENTIAL one-at-a-time). Read the report at
/// %TEMP%/assoc_index_report.txt after a RUN_SLOW=1 run. The findings drive whether a dedicated core trainer
/// is warranted. Runs at PRODUCTION sizing (<see cref="ProductionDims.HiddenSize"/>) so the dynamics it
/// reports are the real substrate's, not a degenerate small-face proxy; these are opt-in [SlowFact].
/// </summary>
public sealed class AssociativeIndexLearningTests
{
    private readonly ITestOutputHelper _out;
    public AssociativeIndexLearningTests(ITestOutputHelper output) => _out = output;

    private static readonly int Hidden = ProductionDims.HiddenSize;

    private sealed record Engine(
        WhitespaceGenesisTokenizer Tok,
        GenesisNeuralModel Model,
        PlatonicSpaceMemory Mem,
        GenesisTrainer Trainer,
        GenesisInferenceEngine Infer);

    private static Engine Build(bool spaceAware = false, bool perceptionEdit = false, int repelNeighbors = 1, bool perceptionRouting = false)
    {
        var config = new GenesisNovaConfig(HiddenSize: Hidden, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { PerceptionRouting = perceptionRouting };
        var mem = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 1234);
        mem.RegisterOperationToken("find"); // op-token = route trigger, excluded from relation edges (daemon parity)
        var trainer = new GenesisTrainer(tok, model, mem, config) { SpaceAwareEdit = spaceAware, PerceptionEdit = perceptionEdit, RepelNeighbors = repelNeighbors };
        var infer = new GenesisInferenceEngine(tok, model, mem, null);
        trainer.SetInferencePolicy(infer);
        return new Engine(tok, model, mem, trainer, infer);
    }

    // Distinct 3-letter tokens from disjoint anchor/key alphabets so cues and keys never collide.
    private static string Tok3(int i, char prefix)
    {
        var a = (char)('a' + (i / 676) % 26);
        var b = (char)('a' + (i / 26) % 26);
        var c = (char)('a' + i % 26);
        return $"{prefix}{a}{b}{c}";
    }

    private static List<(string Cue, string Key)> MakeIndex(int n)
        => Enumerable.Range(0, n).Select(i => ($"find {Tok3(i, 'q')}", Tok3(i, 'z'))).ToList();

    private static string Canon(string s) => new((s ?? "").ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

    private static void TrainBroad(Engine e, List<(string Cue, string Key)> data, int epochs, Random rng)
    {
        var order = data.ToList();
        for (var ep = 0; ep < epochs; ep++)
        {
            for (var i = order.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
            foreach (var (cue, key) in order)
                e.Trainer.TrainStep(new GenesisExample(cue, key));
        }
    }

    // Train each association to "competence" one at a time, in order, with NO interleaving — induces forgetting.
    private static void TrainSequential(Engine e, List<(string Cue, string Key)> data, int stepsPerItem)
    {
        foreach (var (cue, key) in data)
            for (var s = 0; s < stepsPerItem; s++)
                e.Trainer.TrainStep(new GenesisExample(cue, key));
    }

    // retrieval% (output contains the key), routed% (answered via platonic, not neural fallback),
    // diversity% (distinct outputs / count — a COLLAPSE detector: ~0 means "same answer for everything").
    private (double Retrieval, double Routed, double Diversity) Measure(Engine e, List<(string Cue, string Key)> probes, int maxTokens = 12)
    {
        if (probes.Count == 0) return (0, 0, 0);
        var hits = 0; var routed = 0; var outputs = new HashSet<string>();
        foreach (var (cue, key) in probes)
        {
            var r = e.Infer.Generate(new GenerationRequest(cue, maxTokens));
            var outC = Canon(r.Output);
            if (outC.Contains(Canon(key))) hits++;
            if (r.UsedPlatonicQuery && !r.UsedNeuralFallback) routed++;
            outputs.Add(outC);
        }
        return (hits / (double)probes.Count, routed / (double)probes.Count, outputs.Count / (double)probes.Count);
    }

    // SPACE-AWARE investigation, phase A (SPACE_AWARE_GRU.md): characterise the BLIND edit head's core
    // weakness — can it UNDO a bad write? Poison each anchor with a strong WRONG edge, then retrain it toward
    // RIGHT, and measure how fast (if ever) retrieval flips WRONG→RIGHT vs a clean-from-scratch control. A large
    // recovery gap = the blind controller can't perceive/correct what it already wrote → motivates perception.
    private double MeasureKeys(Engine e, List<string> cues, List<string> keys, int maxTokens = 8)
    {
        var hits = 0;
        for (var i = 0; i < cues.Count; i++)
        {
            var o = Canon(e.Infer.Generate(new GenerationRequest(cues[i], maxTokens)).Output);
            if (Canon(keys[i]).Length > 0 && o.Contains(Canon(keys[i]))) hits++;
        }
        return hits / (double)cues.Count;
    }

    [SlowFact]
    public void SpaceAware_PoisonRecovery_BlindBaseline()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== BLIND edit head: can it UNDO a bad write? (HiddenSize={Hidden}) ===");
        var rng = new Random(21);
        const int N = 30;
        var cues = Enumerable.Range(0, N).Select(i => $"find {Tok3(i, 'q')}").ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var rightData = cues.Zip(right, (c, k) => (c, k)).ToList();
        var wrongData = cues.Zip(wrong, (c, k) => (c, k)).ToList();

        // CONTROL — clean A->RIGHT from scratch: how fast does retrieval reach RIGHT?
        Log("");
        Log("[control] clean A->RIGHT from scratch:  epoch -> RIGHT%");
        var ec = Build();
        for (var ep = 1; ep <= 20; ep++)
        {
            TrainBroad(ec, rightData, epochs: 1, rng);
            if (ep % 5 == 0) Log($"    e{ep,2}  RIGHT={MeasureKeys(ec, cues, right),5:P0}");
        }

        // POISONED — strong WRONG edge first, then retrain toward RIGHT; watch the flip (or the stall).
        Log("");
        Log("[poisoned] A->WRONG (15 epochs), then retrain A->RIGHT:");
        var ep2 = Build();
        for (var p = 0; p < 15; p++) TrainBroad(ep2, wrongData, epochs: 1, rng);
        Log($"    after poison:  WRONG={MeasureKeys(ep2, cues, wrong),5:P0}  RIGHT={MeasureKeys(ep2, cues, right),5:P0}");
        Log("    recovery:  epoch -> RIGHT% / WRONG% (RIGHT should rise, WRONG should fall)");
        for (var ep = 1; ep <= 20; ep++)
        {
            TrainBroad(ep2, rightData, epochs: 1, rng);
            if (ep % 5 == 0) Log($"    e{ep,2}  RIGHT={MeasureKeys(ep2, cues, right),5:P0}  WRONG={MeasureKeys(ep2, cues, wrong),5:P0}");
        }
        Log("");
        Log("=> if clean reaches ~100% but poisoned RIGHT lags / WRONG lingers, the blind head can't undo its");
        Log("   own write — the empirical case for a space-aware (read-before-write) edit policy.");

        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "space_aware_poison.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(report.Count > 0);
    }

    // Rank of `target` among `from`'s nearest neighbours over the WHOLE space (-1 if outside top-N). This is the
    // raw material of a perception vector: "is the answer already my nearest neighbour, and if not, how far?"
    private int RankOf(Engine e, string from, string target, int top = 32)
    {
        var near = e.Mem.GetNearestConcepts(from, candidates: null, maxNeighbors: top);
        for (var i = 0; i < near.Count; i++)
            if (Canon(near[i].Symbol) == Canon(target)) return i;
        return -1;
    }

    // SPACE-AWARE phase A (runs NOW on the blind engine): prove the PERCEPTION SIGNAL the GRU would read is
    // INFORMATIVE — it cleanly separates a well-learned anchor from a poisoned/colliding one. If the signal is
    // there, a controller that reads it can act on it (SPACE_AWARE_GRU.md §A,§F,§I).
    [SlowFact]
    public void SpaceAware_A_PerceptionVector_IsInformative()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== perception signal: rank-of-answer & nearest-neighbour, good vs poisoned (HiddenSize={Hidden}) ===");
        var rng = new Random(31);
        const int N = 20;
        var ops = Enumerable.Range(0, N).Select(i => Tok3(i, 'q')).ToList();   // anchor operand concepts
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var cues = ops.Select(o => $"find {o}").ToList();

        // GOOD: learn A->RIGHT cleanly.
        var good = Build();
        for (var ep = 0; ep < 20; ep++) TrainBroad(good, cues.Zip(right, (c, k) => (c, k)).ToList(), 1, rng);
        // POISONED: strong A->WRONG, only a token recovery toward RIGHT (left stuck on the wrong edge).
        var bad = Build();
        for (var ep = 0; ep < 18; ep++) TrainBroad(bad, cues.Zip(wrong, (c, k) => (c, k)).ToList(), 1, rng);
        for (var ep = 0; ep < 3; ep++) TrainBroad(bad, cues.Zip(right, (c, k) => (c, k)).ToList(), 1, rng);

        double meanRank(Engine e, List<string> tgt) => Enumerable.Range(0, N)
            .Select(i => { var r = RankOf(e, ops[i], tgt[i]); return r < 0 ? 32.0 : r; }).Average();
        // (F) is the winning distractor (the wrong concept) identifiable as the nearest neighbour on a poisoned anchor?
        var poisonNearestIsWrong = Enumerable.Range(0, N).Count(i => RankOf(bad, ops[i], wrong[i]) == 0) / (double)N;

        Log($"  GOOD     mean rank-of-RIGHT = {meanRank(good, right):F1}  (low = answer is the nearest neighbour)");
        Log($"  POISONED mean rank-of-RIGHT = {meanRank(bad, right):F1}  nearest-is-WRONG = {poisonNearestIsWrong:P0}");
        Log("  => a low good-rank vs high poisoned-rank means the perception readout SEPARATES them: a controller");
        Log("     that reads rank/nearest can tell 'already correct' from 'colliding with a distractor' and act.");

        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "space_aware_perception.txt"), string.Join(Environment.NewLine, report)); } catch { }
        // The signal must be informative: the answer is clearly nearer for good anchors than poisoned ones.
        Assert.True(meanRank(good, right) < meanRank(bad, right),
            "rank-of-answer should be lower (nearer) for cleanly-learned anchors than poisoned ones — else the perception vector carries no usable signal");
    }

    // SPACE-AWARE phase A (runs NOW): does the space's perceived CONFIDENCE predict correctness? If yes, the
    // route head can read it to choose platonic-vs-neural / abstain, instead of guessing (SPACE_AWARE_GRU.md §I).
    [SlowFact]
    public void SpaceAware_I_PerceivedConfidence_PredictsCorrectness()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== perceived confidence vs correctness (HiddenSize={Hidden}) ===");
        var rng = new Random(41);
        // Mixed difficulty: half trained well, half barely — so we get both correct and incorrect retrievals.
        var data = MakeIndex(40);
        var e = Build();
        for (var ep = 0; ep < 6; ep++) TrainBroad(e, data, 1, rng); // deliberately under-trained → a mix of hits/misses
        double cCorrect = 0, cWrong = 0; int nC = 0, nW = 0;
        foreach (var (cue, key) in data)
        {
            var r = e.Infer.Generate(new GenerationRequest(cue, 8));
            var ok = Canon(r.Output).Contains(Canon(key));
            if (ok) { cCorrect += r.PlatonicConfidence; nC++; } else { cWrong += r.PlatonicConfidence; nW++; }
        }
        var mc = nC > 0 ? cCorrect / nC : 0; var mw = nW > 0 ? cWrong / nW : 0;
        Log($"  mean confidence  correct={mc:F3} (n={nC})   incorrect={mw:F3} (n={nW})");
        Log("  => if correct >> incorrect, perceived confidence is a usable read for routing / abstention.");
        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "space_aware_confidence.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(report.Count > 0); // instrument (confidence may be 0 early — the report is the value)
    }

    // ── SPACE-AWARE phase B: require the perception vector wired into the controller (core change). Captured as
    //    SKIPPED so the plan isn't forgotten; un-skip once SPACE_AWARE_GRU.md §A is implemented. ──────────────

    // (B) READ-BEFORE-WRITE — the perceived "do I already retrieve the answer?" read cleanly flags which anchors
    // need an edit (act) vs not (skip): trained anchors read correct, untrained read needs-work.
    [SlowFact]
    public void SpaceAware_B_ReadBeforeWrite_GapFlagsAction()
    {
        var rng = new Random(52);
        var data = MakeIndex(30);
        var e = Build();
        var trained = data.Take(15).ToList();
        for (var ep = 0; ep < 20; ep++) TrainBroad(e, trained, 1, rng); // train only the first half
        var tCorrect = trained.Count(d => RankOf(e, d.Cue.Split(' ')[1], d.Key) == 0) / 15.0;
        var uCorrect = data.Skip(15).Count(d => RankOf(e, d.Cue.Split(' ')[1], d.Key) == 0) / 15.0;
        _out.WriteLine($"[B] already-retrieves read: trained={tCorrect:P0}  untrained={uCorrect:P0}");
        Assert.True(tCorrect > uCorrect, "the 'already retrieves?' read must distinguish anchors that need editing from those that don't");
    }

    // (C) DIFFERENTIABLE READOUT — the learned perception weight (_editPerceptionW) is a differentiable linear
    // readout over the (now 6-dim, distance-enriched) perception vector. Confirm the readout drives recovery:
    // a perception-fed head learns to recover from poison where a blind head can't.
    [SlowFact]
    public void SpaceAware_C_DifferentiableReadout_Helps()
    {
        const int N = 25;
        var cues = Enumerable.Range(0, N).Select(i => $"find {Tok3(i, 'q')}").ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var rightData = cues.Zip(right, (c, k) => (c, k)).ToList();
        var wrongData = cues.Zip(wrong, (c, k) => (c, k)).ToList();
        double Rec(bool perc)
        {
            var rng = new Random(81);
            var e = Build(perceptionEdit: perc);
            for (var p = 0; p < 15; p++) TrainBroad(e, wrongData, 1, rng);
            for (var ep = 0; ep < 22; ep++) TrainBroad(e, rightData, 1, rng);
            return MeasureKeys(e, cues, right);
        }
        var blind = Rec(false); var readout = Rec(true);
        _out.WriteLine($"[C] differentiable-readout recovery={readout:P0}  blind={blind:P0}");
        Assert.True(readout > blind + 0.05, $"the learned perception readout should drive recovery (readout={readout:P0}, blind={blind:P0})");
    }

    // (D) POINTER HEAD — repel the top-k nearest distractors (RepelNeighbors), not just the single nearest. With
    // MULTIPLE poisons per anchor, the multi-repel pointer clears them where single-repel leaves competitors.
    [SlowFact]
    public void SpaceAware_D_PointerHead_SelectsWhereToEdit()
    {
        const int N = 25;
        var cues = Enumerable.Range(0, N).Select(i => $"find {Tok3(i, 'q')}").ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong1 = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var wrong2 = Enumerable.Range(0, N).Select(i => Tok3(i, 'x')).ToList();
        var rightData = cues.Zip(right, (c, k) => (c, k)).ToList();
        double RecoverMultiPoison(int repelK)
        {
            var rng = new Random(82);
            var e = Build(perceptionEdit: true, repelNeighbors: repelK); // LEARNED: m decides how many to clear (1..repelK)
            for (var p = 0; p < 10; p++) TrainBroad(e, cues.Zip(wrong1, (c, k) => (c, k)).ToList(), 1, rng);
            for (var p = 0; p < 10; p++) TrainBroad(e, cues.Zip(wrong2, (c, k) => (c, k)).ToList(), 1, rng); // 2 distractors
            for (var ep = 0; ep < 20; ep++) TrainBroad(e, rightData, 1, rng);
            return MeasureKeys(e, cues, right);
        }
        var single = RecoverMultiPoison(1); var pointer = RecoverMultiPoison(3);
        _out.WriteLine($"[D] multi-poison recovery: single-repel(k=1)={single:P0}  pointer(k=3)={pointer:P0}");
        Assert.True(pointer >= single, $"repelling the top-k distractors should recover at least as well as single-repel (k3={pointer:P0}, k1={single:P0})");
    }

    // (E) GAP-MINIMISING EDIT — the lookahead signal is well-defined: a correct write REDUCES the rank-of-answer
    // gap (a wrong write would increase it), so an edit head can be supervised toward the gap-reducing action.
    [SlowFact]
    public void SpaceAware_E_GapMinimizingEdit_LookaheadSignalExists()
    {
        var rng = new Random(53);
        const int N = 20;
        var ops = Enumerable.Range(0, N).Select(i => Tok3(i, 'q')).ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var cues = ops.Select(o => $"find {o}").ToList();
        var e = Build();
        for (var ep = 0; ep < 4; ep++) TrainBroad(e, cues.Zip(wrong, (c, k) => (c, k)).ToList(), 1, rng); // poison-ish start
        double Mean(List<string> t) => Enumerable.Range(0, N).Select(i => { var r = RankOf(e, ops[i], t[i]); return r < 0 ? 32.0 : r; }).Average();
        var before = Mean(right);
        for (var ep = 0; ep < 8; ep++) TrainBroad(e, cues.Zip(right, (c, k) => (c, k)).ToList(), 1, rng); // the gap-reducing edit
        var after = Mean(right);
        _out.WriteLine($"[E] rank-of-answer gap: before={before:F1} -> after correct edits={after:F1}");
        Assert.True(after < before, "the correct write must reduce the rank-of-answer gap — the lookahead target a gap-minimising edit head needs");
    }

    // (F) CONTRASTIVE PERCEPTION — on a poisoned anchor the WINNING DISTRACTOR is the nearest neighbour, so the
    // controller can perceive and repel exactly the confuser.
    [SlowFact]
    public void SpaceAware_F_ContrastivePerception_DistractorIdentifiable()
    {
        var rng = new Random(54);
        const int N = 20;
        var ops = Enumerable.Range(0, N).Select(i => Tok3(i, 'q')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var cues = ops.Select(o => $"find {o}").ToList();
        var e = Build();
        for (var ep = 0; ep < 18; ep++) TrainBroad(e, cues.Zip(wrong, (c, k) => (c, k)).ToList(), 1, rng);
        var nearestIsWrong = Enumerable.Range(0, N).Count(i => RankOf(e, ops[i], wrong[i]) == 0) / (double)N;
        _out.WriteLine($"[F] winning distractor is the nearest neighbour on {nearestIsWrong:P0} of poisoned anchors");
        Assert.True(nearestIsWrong > 0.5, "the confuser should be perceivable as the nearest neighbour so it can be targeted for repulsion");
    }

    // (G) MEMORY-AUGMENTED READ/WRITE — content-addressable round-trip: WRITE an edge (relate), then READ it back
    // (the written partner is the nearest neighbour). The NTM/DNC primitive a space-aware controller would drive.
    [SlowFact]
    public void SpaceAware_G_MemoryReadWrite_RoundTrip()
    {
        var e = Build();
        var rng = new Random(55);
        var pairs = Enumerable.Range(0, 20).Select(i => (A: Tok3(i, 'q'), B: Tok3(i, 'z'))).ToList();
        // WRITE via the REAL write path (TrainStep → ObservePlatonicSpace), not raw ObserveContradiction — that
        // is how a controller actually writes an edge (coupling + edit head), the NTM/DNC write primitive.
        var data = pairs.Select(p => ($"find {p.A}", p.B)).ToList();
        for (var ep = 0; ep < 15; ep++) TrainBroad(e, data, 1, rng);
        // READ back via content addressing (GetNearestConcepts): the written partner is the nearest neighbour.
        var roundTrip = pairs.Count(p => RankOf(e, p.A, p.B) == 0) / (double)pairs.Count;
        _out.WriteLine($"[G] written edges that read back as nearest neighbour: {roundTrip:P0}");
        Assert.True(roundTrip > 0.8, "a written edge must read back as the nearest neighbour (content-addressable memory)");
    }

    // (H) PERCEPTION CURRICULUM — a READ lesson: teach a distinct read verb ("near <X> => Y") so the controller
    // can be queried ABOUT the space, not only to retrieve. If a read form is learnable alongside `find`, the
    // curriculum's read lessons (what-is-near-X / does-X-retrieve-Y) are trainable.
    [SlowFact]
    public void SpaceAware_H_PerceptionCurriculum_ReadLessonsLearnable()
    {
        var rng = new Random(83);
        const int N = 30;
        var e = Build();
        e.Mem.RegisterOperationToken("near"); // a READ verb, distinct from the find/retrieve verb
        var pairs = Enumerable.Range(0, N).Select(i => (A: Tok3(i, 'q'), B: Tok3(i, 'z'))).ToList();
        var readData = pairs.Select(p => ($"near {p.A}", p.B)).ToList();
        for (var ep = 0; ep < 20; ep++) TrainBroad(e, readData, 1, rng);
        var hit = pairs.Count(p => Canon(e.Infer.Generate(new GenerationRequest($"near {p.A}", 8)).Output).Contains(Canon(p.B))) / (double)N;
        _out.WriteLine($"[H] read-lesson 'near X' learnable: {hit:P0}");
        Assert.True(hit > 0.7, $"a read-query form must be learnable for the perception curriculum (hit={hit:P0})");
    }

    // INTEGRATIVE (headline): a SPACE-AWARE edit (read the winning distractor, repel it) recovers from a poisoned
    // anchor where the BLIND attract-only edit stalls (~63% in the baseline). This is the mechanism, not a probe.
    [SlowFact]
    public void SpaceAware_PoisonRecovery_AwareBeatsBlind()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== poison recovery: BLIND vs SPACE-AWARE edit (HiddenSize={Hidden}) ===");
        const int N = 30;
        var cues = Enumerable.Range(0, N).Select(i => $"find {Tok3(i, 'q')}").ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var rightData = cues.Zip(right, (c, k) => (c, k)).ToList();
        var wrongData = cues.Zip(wrong, (c, k) => (c, k)).ToList();

        double Recover(bool aware)
        {
            var rng = new Random(61); // SAME seed both runs → the only difference is space-aware editing
            var e = Build(spaceAware: aware);
            for (var p = 0; p < 15; p++) TrainBroad(e, wrongData, 1, rng);   // poison A->WRONG
            for (var ep = 0; ep < 20; ep++) TrainBroad(e, rightData, 1, rng); // recover A->RIGHT
            return MeasureKeys(e, cues, right);
        }

        var blind = Recover(false);
        var aware = Recover(true);
        Log($"  RIGHT% after 20 recovery epochs:  BLIND={blind:P0}   SPACE-AWARE(oracle)={aware:P0}");
        Log("  => space-aware editing reads the winning distractor and repels it, so the target can take the");
        Log("     nearest slot — undoing the bad write the blind attract-only edit cannot.");
        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "space_aware_recovery.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(aware > blind + 0.05,
            $"space-aware editing must recover from poison meaningfully better than blind (aware={aware:P0}, blind={blind:P0})");
    }

    // THE TRAINABILITY QUESTION: can the GRU LEARN the space-aware policy? The edit head is fed a perception
    // vector (rank-of-target / distractor-winning) and its magnitude scales the distractor repulsion; it is
    // trained ONLY by the within-step reward (no hand-coded repel strength). Compare BLIND (no perception) vs
    // GRU-LEARNED (perception-fed, reward-trained) vs the hand-coded ORACLE ceiling, on poison recovery.
    [SlowFact]
    public void SpaceAware_GruCanLearn_PoisonRecovery()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== can the GRU LEARN space-aware editing? poison recovery (HiddenSize={Hidden}) ===");
        const int N = 30;
        var cues = Enumerable.Range(0, N).Select(i => $"find {Tok3(i, 'q')}").ToList();
        var right = Enumerable.Range(0, N).Select(i => Tok3(i, 'z')).ToList();
        var wrong = Enumerable.Range(0, N).Select(i => Tok3(i, 'w')).ToList();
        var rightData = cues.Zip(right, (c, k) => (c, k)).ToList();
        var wrongData = cues.Zip(wrong, (c, k) => (c, k)).ToList();

        double Recover(bool spaceAware, bool perception)
        {
            var rng = new Random(71); // same seed across modes
            var e = Build(spaceAware: spaceAware, perceptionEdit: perception);
            for (var p = 0; p < 15; p++) TrainBroad(e, wrongData, 1, rng);    // poison
            for (var ep = 0; ep < 25; ep++) TrainBroad(e, rightData, 1, rng); // recover (a few more epochs to let the head LEARN)
            return MeasureKeys(e, cues, right);
        }

        var blind = Recover(false, false);
        var learned = Recover(false, true);   // GRU-learned: perception-fed edit head, reward-trained
        var oracle = Recover(true, false);     // hand-coded read-repel ceiling
        Log($"  RIGHT% after recovery:  BLIND={blind:P0}   GRU-LEARNED={learned:P0}   ORACLE(hand-coded)={oracle:P0}");
        Log("  => GRU-LEARNED > BLIND means the controller can be TRAINED to read the space and act on it");
        Log("     (the reward taught the perception-conditioned head to repel the distractor). Toward ORACLE = how close.");
        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "space_aware_gru_learn.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(learned > blind + 0.05,
            $"the GRU should LEARN space-aware recovery better than blind (learned={learned:P0}, blind={blind:P0}, oracle={oracle:P0})");
    }

    // (I) PERCEIVED-CONFIDENCE ROUTING (mechanism, not just signal): the route head reads a target-agnostic
    // perception ("can the space answer this query?") and is REINFORCED toward the resolved route. After training,
    // ANSWERABLE (trained) queries should route platonic at least as often as UNANSWERABLE (never-trained) ones —
    // the GRU learned to route from perceived retrievability, which token-only routing can't discriminate.
    [SlowFact]
    public void SpaceAware_I_PerceptionRouting_DiscriminatesAnswerable()
    {
        var rng = new Random(91);
        const int N = 30;
        var trained = MakeIndex(N);                                                              // answerable
        var untrained = Enumerable.Range(100, N).Select(i => $"find {Tok3(i, 'q')}").ToList();   // never trained → unanswerable
        var e = Build(perceptionRouting: true);
        for (var ep = 0; ep < 20; ep++) TrainBroad(e, trained, 1, rng);
        double Purity(System.Collections.Generic.IEnumerable<string> cues)
        {
            var c = cues.ToList();
            var routed = c.Count(q => { var r = e.Infer.Generate(new GenerationRequest(q, 4)); return r.UsedPlatonicQuery && !r.UsedNeuralFallback; });
            return c.Count == 0 ? 0 : routed / (double)c.Count;
        }
        var answerable = Purity(trained.Select(d => d.Cue));
        var unanswerable = Purity(untrained);
        _out.WriteLine($"[I] platonic-route purity: answerable={answerable:P0}  unanswerable={unanswerable:P0}");
        Assert.True(answerable >= unanswerable,
            $"perception routing should route answerable queries platonic at least as much as unanswerable ones (ans={answerable:P0}, unans={unanswerable:P0})");
    }

    [SlowFact]
    public void AssociativeIndex_MultiTokenKey_Probe()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== multi-token key construction (HiddenSize={Hidden}, ADEQUATE token budget) ===");
        Log("parts | key tokens | retrieval | routed | diversity   (N=30, 20 epochs)");
        var rng = new Random(99);
        foreach (var parts in new[] { 1, 2, 3, 4, 5 })
        {
            var e = Build();
            var data = Enumerable.Range(0, 30).Select(i =>
                ($"find {Tok3(i, 'q')}",
                 string.Join("-", Enumerable.Range(0, parts).Select(p => Tok3(i, (char)('z' - p)))))).ToList();
            TrainBroad(e, data, epochs: 20, rng);
            var (ret, routed, div) = Measure(e, data, maxTokens: parts * 2 + 4); // generous: never truncate the key
            Log($"  {parts}    | {(parts * 2 - 1),2} tokens   | {ret,5:P0}    | {routed,5:P0}  | {div,5:P0}");
        }
        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "assoc_multitoken_report.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(report.Count > 0);
    }

    [SlowFact]
    public void AssociativeIndex_AtomicConstruction_Report()
    {
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log($"=== atomic construction lessons (HiddenSize={Hidden}) — which sub-skill breaks? ===");
        var rng = new Random(13);

        // ATOM A — MULTI-TOKEN KEY: the daemon's real keys are hyphenated ("nova-claude-memory" tokenizes to
        // nova | - | claude | - | memory). Constructing a SEQUENCE is a different skill than emitting one token.
        Log("");
        Log("[A] MULTI-TOKEN (hyphenated) keys vs single-token keys, N=40 broad:");
        foreach (var (label, keyFn) in new (string, Func<int, string>)[]
        {
            ("single-token ", i => Tok3(i, 'z')),
            ("2-part hyphen", i => $"{Tok3(i, 'z')}-{Tok3(i, 'w')}"),
            ("3-part hyphen", i => $"{Tok3(i, 'z')}-{Tok3(i, 'w')}-{Tok3(i, 'v')}"),
        })
        {
            var e = Build();
            var data = Enumerable.Range(0, 40).Select(i => ($"find {Tok3(i, 'q')}", keyFn(i))).ToList();
            TrainBroad(e, data, epochs: 20, rng);
            var (ret, routed, div) = Measure(e, data);
            Log($"    {label}  retrieval={ret,5:P0}  routed={routed,5:P0}  diversity={div,5:P0}");
        }

        // ATOM B — ONE-TO-MANY: one anchor relates to several keys (a Type's members). Train each edge as a
        // single-member example; measure whether the anchor retrieves ANY valid member (recall) and how many.
        Log("");
        Log("[B] ONE-TO-MANY: 20 anchors, each with 4 distinct single-token members:");
        {
            var e = Build();
            var groups = Enumerable.Range(0, 20)
                .Select(i => (Anchor: $"find {Tok3(i, 'q')}", Members: Enumerable.Range(0, 4).Select(j => Tok3(i * 4 + j, 'z')).ToList()))
                .ToList();
            var data = groups.SelectMany(g => g.Members.Select(mem => (g.Anchor, mem))).ToList();
            TrainBroad(e, data, epochs: 20, rng);
            var anyHit = 0; double memberRecall = 0;
            foreach (var g in groups)
            {
                var outC = Canon(e.Infer.Generate(new GenerationRequest(g.Anchor, 8)).Output);
                var hit = g.Members.Count(m => outC.Contains(Canon(m)));
                if (hit >= 1) anyHit++;
                memberRecall += hit / (double)g.Members.Count;
            }
            Log($"    anchor retrieves >=1 valid member: {anyHit / 20.0:P0}   mean members recalled (free over-gen): {memberRecall / 20.0:P0}");
        }

        // ATOM C — SHARED KEY: many anchors map to ONE key (many keywords → one memory). Each anchor must
        // retrieve the shared key without the key collapsing all anchors together.
        Log("");
        Log("[C] SHARED KEY: 30 distinct anchors all mapping to the SAME single-token key:");
        {
            var e = Build();
            var key = "sharedkey";
            var data = Enumerable.Range(0, 30).Select(i => ($"find {Tok3(i, 'q')}", key)).ToList();
            TrainBroad(e, data, epochs: 20, rng);
            var (ret, routed, _) = Measure(e, data);
            Log($"    each anchor retrieves the shared key: {ret:P0}  routed={routed:P0}");
        }

        // ATOM D — SCALE: does single-token retrieval hold toward the daemon's set count?
        Log("");
        Log("[D] SCALE (single-token, broad 20 epochs):");
        foreach (var n in new[] { 200, 400 })
        {
            var e = Build();
            var data = MakeIndex(n);
            TrainBroad(e, data, epochs: 20, rng);
            var (ret, routed, div) = Measure(e, data);
            Log($"    N={n,3}  retrieval={ret,5:P0}  routed={routed,5:P0}  diversity={div,5:P0}");
        }

        try { File.WriteAllText(Path.Combine(Path.GetTempPath(), "assoc_atomic_report.txt"), string.Join(Environment.NewLine, report)); } catch { }
        Assert.True(report.Count > 0);
    }

    [SlowFact]
    public void AssociativeIndex_Scale_And_Forgetting_Report()
    {
        var reportPath = Path.Combine(Path.GetTempPath(), "assoc_index_report.txt");
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }

        Log($"=== associative-index learning investigation (HiddenSize={Hidden}) ===");
        var rng = new Random(7);

        // EXPERIMENT 1 — SCALE: broad-trained index, retrieval/routed/diversity vs N.
        Log("");
        Log("[1] SCALE (broad training, 20 epochs):  N | retrieval | routed-platonic | output-diversity");
        foreach (var n in new[] { 20, 50, 100 })
        {
            var e = Build();
            var data = MakeIndex(n);
            TrainBroad(e, data, epochs: 20, rng);
            var (ret, routed, div) = Measure(e, data);
            Log($"    N={n,3}  retrieval={ret,5:P0}  routed={routed,5:P0}  diversity={div,5:P0}");
        }

        // EXPERIMENT 2 — FORGETTING: same N, BROAD vs SEQUENTIAL; retention of the FIRST third vs the LAST third.
        Log("");
        Log("[2] FORGETTING (N=45): retention of the EARLIEST-trained third vs the LATEST third");
        const int fN = 45;
        var idx = MakeIndex(fN);
        var firstThird = idx.Take(15).ToList();
        var lastThird = idx.Skip(30).ToList();

        var eb = Build();
        TrainBroad(eb, idx, epochs: 15, rng);
        var (brFirst, brFr, brFd) = Measure(eb, firstThird);
        var (brLast, brLr, brLd) = Measure(eb, lastThird);
        Log($"    BROAD       first-third retrieval={brFirst,5:P0} (routed {brFr,4:P0})   last-third retrieval={brLast,5:P0} (routed {brLr,4:P0})");

        var es = Build();
        TrainSequential(es, idx, stepsPerItem: 15);
        var (seFirst, seFr, seFd) = Measure(es, firstThird);
        var (seLast, seLr, seLd) = Measure(es, lastThird);
        Log($"    SEQUENTIAL  first-third retrieval={seFirst,5:P0} (routed {seFr,4:P0})   last-third retrieval={seLast,5:P0} (routed {seLr,4:P0})");
        Log($"    => forgetting gap (broad first - sequential first) = {(brFirst - seFirst):P0}; sequential collapse diversity first={seFd:P0} last={seLd:P0}");

        try { File.WriteAllText(reportPath, string.Join(Environment.NewLine, report)); } catch { /* best effort */ }
        Log("");
        Log($"(report written to {reportPath})");

        // Minimal sanity: the harness ran end-to-end and produced measurements. The VALUE is the report; this
        // only guards that training + inference + the collapse detector are wired and that BROAD does not forget
        // the earliest third MORE than SEQUENTIAL (the hypothesis under test — broad >= sequential on first-third).
        Assert.True(brFirst >= seFirst - 1e-9,
            $"broad should retain the earliest third at least as well as sequential (broad={brFirst:P0}, seq={seFirst:P0})");
    }
}
