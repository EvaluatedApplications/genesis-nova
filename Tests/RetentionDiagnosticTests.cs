using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// RETENTION DIAGNOSIS (2026-06-14). Catastrophic forgetting is measured END-TO-END, so it could live in
/// any of three layers: the platonic relation EDGE, the ROUTER (PredictRoute), or the neural DECODE. This
/// test trains the prior lessons (number-word + retrieval) to competence, snapshots each probe's layer
/// state, trains arithmetic to induce forgetting, then RE-snapshots and classifies every regression as
/// EDGE-LOST / ROUTER-DRIFT / DECODE-FAIL. Writes a breakdown report to a temp file (read it after the run).
/// Not an assertion test — an instrument. It only asserts that it produced a regression breakdown.
/// </summary>
public sealed class RetentionDiagnosticTests
{
    private readonly ITestOutputHelper _out;
    public RetentionDiagnosticTests(ITestOutputHelper output) => _out = output;

    private sealed record Probe(string Input, string Expected, string Kind);

    private sealed record Snapshot(int Route, double RouteConf, string Path, bool UsedPlatonic, bool EdgeOk, bool Correct, string Output, string Relations);

    [Fact]
    public void Retention_Diagnosis_BreakdownByLayer()
    {
        var reportPath = Path.Combine(Path.GetTempPath(), "retention_diag.txt");
        var report = new List<string>();
        void Log(string s) { _out.WriteLine(s); report.Add(s); }
        Log("=== retention diagnosis: classify regressions by layer (edge / router / decode) ===");

        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);
        var inference = new GenesisInferenceEngine(
            tokenizer, model, memory, null);
        trainer.SetInferencePolicy(inference);

        var rng = new Random(7);
        void Shuffle(List<GenesisExample> d)
        {
            for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); }
        }

        // ── Prior lessons: number-word + retrieval ──────────────────────────────────────────────────
        var nw = new NumberWordCreator();
        var rc = new CategoryRetrievalCreator();
        var prior = new List<GenesisExample>();
        foreach (var (i, o) in nw.Generate(120, 0, true)) prior.Add(new GenesisExample(i, o, SourceCreatorName: nw.Name));
        foreach (var (i, o) in rc.Generate(120, 0, true)) prior.Add(new GenesisExample(i, o, SourceCreatorName: rc.Name));

        // Probes = a representative held set (drawn from the same generators, deduped).
        var probes = prior
            .DistinctBy(e => e.Input)
            .Take(40)
            .Select(e => new Probe(e.Input, e.Output, e.SourceCreatorName!.Contains("number") ? "number-word" : "retrieval"))
            .ToList();

        int CorrectCount() => probes.Count(p => AnswerEquivalence.Equivalent(
            inference.Generate(new GenerationRequest(p.Input, 4)).Output, p.Expected));

        var pool = prior.ToList();
        Shuffle(pool);
        var idx = 0;
        const int priorMax = 4000;
        for (var s = 0; s < priorMax; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]);
            if (s % 250 == 0 && s > 0 && CorrectCount() >= (int)(probes.Count * 0.9))
                break;
        }
        Log($"prior-lesson competence: {CorrectCount()}/{probes.Count} probes correct after training");

        // ── BASELINE snapshot (only probes the model actually got right) ────────────────────────────
        var baseline = probes.ToDictionary(p => p, p => Capture(p, model, memory, inference, tokenizer));
        var learned = probes.Where(p => baseline[p].Correct).ToList();
        Log($"learned (baseline-correct) probes: {learned.Count}/{probes.Count}");

        // ── Induce forgetting: train arithmetic add+sub ─────────────────────────────────────────────
        var arith = new List<GenesisExample>();
        foreach (var op in new[] { "add", "sub" })
        {
            var c = new ArithmeticCreator(op);
            foreach (var d in new[] { 0, 1 })
                foreach (var (i, o) in c.Generate(120, d, true)) arith.Add(new GenesisExample(i, o, SourceCreatorName: c.Name));
        }
        // Induce forgetting by training arithmetic UNTIL IT IS MASTERED, then measure retention — not by
        // burning a fixed step count. Once arithmetic is learned it has applied its full interference
        // pressure on the shared params; further steps only waste GPU. Held probes detect that mastery.
        var arithProbes = arith.DistinctBy(e => e.Input).Take(24).ToList();
        int ArithCorrect() => arithProbes.Count(e => AnswerEquivalence.Equivalent(
            inference.Generate(new GenerationRequest(e.Input, 4)).Output, e.Output));

        Shuffle(arith);
        idx = 0;
        const int arithMax = 2000;                                   // give-up cap (safety), not the normal stop
        var arithBar = (int)Math.Ceiling(arithProbes.Count * 0.80); // arithmetic majority-mastery (matches the regime)
        var arithStepsRun = 0;
        for (var s = 0; s < arithMax; s++)
        {
            if (idx >= arith.Count) { Shuffle(arith); idx = 0; }
            trainer.TrainStep(arith[idx++]);
            arithStepsRun++;
            if (arithStepsRun % 250 == 0 && ArithCorrect() >= arithBar)
                break;
        }
        Log($"arithmetic mastered in {arithStepsRun}/{arithMax} steps ({ArithCorrect()}/{arithProbes.Count} probes) — induces forgetting");

        // ── AFTER snapshot + classify regressions ───────────────────────────────────────────────────
        var after = learned.ToDictionary(p => p, p => Capture(p, model, memory, inference, tokenizer));
        var regressed = learned.Where(p => !after[p].Correct).ToList();

        Log("");
        Log($"RETENTION after arithmetic: {learned.Count - regressed.Count}/{learned.Count} retained, {regressed.Count} REGRESSED");
        Log("");

        var cat = new Dictionary<string, int>
        {
            ["EDGE-LOST"] = 0, ["ROUTER-DRIFT"] = 0, ["DECODE-FAIL (routed platonic)"] = 0, ["OTHER"] = 0
        };
        foreach (var p in regressed)
        {
            var b = baseline[p];
            var a = after[p];
            string verdict;
            if (!a.EdgeOk && b.EdgeOk)
                verdict = "EDGE-LOST";                                   // the relation edge itself decayed
            else if (b.Route == 1 && a.Route != 1)
                verdict = "ROUTER-DRIFT";                                // was platonic-direct, now routed elsewhere
            else if (a.Route == 1 && a.EdgeOk)
                verdict = "DECODE-FAIL (routed platonic)";               // routed right, edge fine, answer still wrong
            else
                verdict = "OTHER";
            cat[verdict]++;
            Log($"  [{verdict,-28}] '{p.Input}'->'{p.Expected}' kind={p.Kind} | " +
                $"baseline(route={b.Route},edge={b.EdgeOk},path={b.Path}) " +
                $"after(route={a.Route},edge={a.EdgeOk},path={a.Path},out='{a.Output}')");
            Log($"        relations baseline: [{b.Relations}]");
            Log($"        relations after:    [{a.Relations}]");
        }

        Log("");
        Log("── REGRESSION BREAKDOWN BY LAYER ──");
        foreach (var (k, v) in cat.OrderByDescending(kv => kv.Value))
            Log($"  {k,-30} {v}");

        File.WriteAllLines(reportPath, report);
        _out.WriteLine($"\nreport written to {reportPath}");

        Assert.True(learned.Count > 0, "no probes were learned to baseline — cannot diagnose retention");
        // Retention guard: after the operand-edge removal + number↔number relation skip + relation-first
        // retrieval, prior lessons must survive arithmetic. Bar at 83% (tolerates run noise; well above
        // the pre-fix ~72%). Pre-fix this was ~26/36; post-fix 36/36.
        var retained = learned.Count - regressed.Count;
        Assert.True(retained >= (int)Math.Ceiling(learned.Count * 0.83),
            $"retention regressed: only {retained}/{learned.Count} prior-lesson probes survived arithmetic. See breakdown.");
    }

    private static Snapshot Capture(
        Probe p, GenesisNeuralModel model, PlatonicSpaceMemory memory,
        GenesisInferenceEngine inference, IGenesisTokenizer tokenizer)
    {
        var tokens = tokenizer.Encode(p.Input);
        var (route, conf) = model.PredictRoute(tokens);
        var g = inference.Generate(new GenerationRequest(p.Input, 4));
        var correct = AnswerEquivalence.Equivalent(g.Output, p.Expected);

        // Edge check: from the input concept, is the EXPECTED answer reachable as a relational neighbour?
        var edgeOk = false;
        var relList = new List<string>();
        try
        {
            foreach (var n in memory.GetNeighbors(p.Input.Trim(), PlatonicNeighborhoodType.Relational, maxNeighbors: 8, minConfidence: 0.0))
            {
                relList.Add($"{n.Concept}:{n.Confidence:F2}");
                if (AnswerEquivalence.Equivalent(n.Concept, p.Expected)) edgeOk = true;
            }
        }
        catch { edgeOk = false; }

        return new Snapshot(route, conf, g.DecisionPath, g.UsedPlatonicQuery, edgeOk, correct, g.Output.Trim(),
            string.Join(" ", relList));
    }
}
