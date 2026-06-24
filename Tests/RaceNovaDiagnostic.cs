using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// DIAGNOSTIC (not a gate) — why does nova sit FLAT at ~47% held-out in RaceBench? Mirror the race's nova EXACTLY
// (GenesisRuntimeState + WithProductionMechanisms, same creators, flat TrainStep), then break the held-out down
// PER CREATOR and show sample answers + decision paths, so we see whether it's arithmetic failing, category not
// generalising to unseen members, abstention, or mis-routing — not guesswork.
public sealed class RaceNovaDiagnostic
{
    private readonly ITestOutputHelper _out;
    public RaceNovaDiagnostic(ITestOutputHelper o) => _out = o;

    [Fact] // WHY does nova's held-out FALL over epochs (84%->70% in the race)? Train many epochs on the LEARNABLE
           // association task and watch its accuracy + WHAT the answer decays into, to find the erosion mechanism.
    public void Why_Does_Association_Recall_Erode()
    {
        const int HIDDEN = 256, SEED = 7;
        var rng = new Random(SEED);
        var assoc = new GenesisNova.Data.Creators.AssociationRecallCreator();

        var ex = new List<(string, string)>();
        foreach (var diff in new[] { 0, 1, 2 }) ex.AddRange(assoc.Generate(400, diff, true));
        var uniq = ex.GroupBy(e => e.Item1).Select(g => g.First()).OrderBy(_ => rng.Next()).ToList();
        var nTrain = (int)(uniq.Count * 0.65);
        var train = uniq.Take(nTrain).ToList();
        var held = uniq.Skip(nTrain).ToList();

        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05, Seed: SEED).WithProductionMechanisms());
        GenerationResult Full(string i) => nova.Inference.Generate(new GenerationRequest(i, 8));
        double Acc() => held.Count(e => AnswerEquivalence.Equivalent(Full(e.Item1).Output?.Trim() ?? "", e.Item2)) / (double)Math.Max(1, held.Count);

        _out.WriteLine($"association: {train.Count} train / {held.Count} held");
        _out.WriteLine($"── FLAT (re-train everything every epoch — like the race) ──");
        _out.WriteLine($"epoch 0 (untrained): {Acc():P0}");
        for (var epoch = 1; epoch <= 8; epoch++)
        {
            foreach (var (i, o) in train.OrderBy(_ => rng.Next())) nova.Trainer.TrainStep(new GenesisExample(i, o));
            _out.WriteLine($"epoch {epoch}: {Acc():P0}");
            if (epoch == 1 || epoch == 8)
                foreach (var (i, o) in held.Take(6))
                { var r = Full(i); _out.WriteLine($"    '{i}' -> '{r.Output?.Trim()}' (want '{o}') [{r.DecisionPath}]"); }
        }

        // CURE: correctness-gating (what the gym's TrainOnFailureOnly does) — predict first, train ONLY the currently
        // WRONG examples, so a learned association stops being re-trained and can't accumulate framing-word/entity
        // noise. Fresh model, same data. If this does NOT erode, the cause IS the flat re-training of correct examples.
        var nova2 = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05, Seed: SEED).WithProductionMechanisms());
        GenerationResult Full2(string i) => nova2.Inference.Generate(new GenerationRequest(i, 8));
        double Acc2() => held.Count(e => AnswerEquivalence.Equivalent(Full2(e.Item1).Output?.Trim() ?? "", e.Item2)) / (double)Math.Max(1, held.Count);
        _out.WriteLine($"── CORRECTNESS-GATED (train only what's currently wrong — like the gym) ──");
        for (var epoch = 1; epoch <= 8; epoch++)
        {
            var trained = 0;
            foreach (var (i, o) in train.OrderBy(_ => rng.Next()))
                if (!AnswerEquivalence.Equivalent(Full2(i).Output?.Trim() ?? "", o)) { nova2.Trainer.TrainStep(new GenesisExample(i, o)); trained++; }
            _out.WriteLine($"epoch {epoch}: {Acc2():P0}  (trained {trained}/{train.Count})");
        }

        // The cure must hold: correctness-gated training LEARNS association-recall and does NOT erode it (flat
        // re-training rots to ~35%; gated holds ~96%). Guards the finding + nova's production training regime.
        Assert.True(Acc2() >= 0.85, $"correctness-gated training must learn AND hold association-recall; got {Acc2():P0}");
    }

    [Fact]
    public void Why_Is_Nova_Flat_In_The_Race()
    {
        const int HIDDEN = 256, SEED = 7;
        var rng = new Random(SEED);
        var creators = ExampleCreatorRegistry.All;

        // Same split shape the race uses: disjoint train / held-out per creator.
        var train = new List<(string Input, string Output)>();
        var heldByCreator = new Dictionary<string, List<(string Input, string Output)>>();
        foreach (var c in creators)
        {
            var ex = new List<(string, string)>();
            foreach (var diff in new[] { 0, 1 }) ex.AddRange(c.Generate(400, diff, true));
            var uniq = ex.GroupBy(e => e.Item1).Select(g => g.First()).OrderBy(_ => rng.Next()).ToList();
            var nTrain = Math.Min((int)(uniq.Count * 0.65), 160);
            train.AddRange(uniq.Take(nTrain));
            heldByCreator[c.Name] = uniq.Skip(nTrain).Take(60).ToList();
        }

        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05, Seed: SEED).WithProductionMechanisms());
        string Gen(string i) => nova.Inference.Generate(new GenerationRequest(i, 8)).Output?.Trim() ?? "";
        GenerationResult Full(string i) => nova.Inference.Generate(new GenerationRequest(i, 8));

        // EPOCH 0 — BEFORE any training. If nova already scores ~its plateau here, the race isn't measuring its
        // LEARNING at all: arithmetic is computed by the homomorphism and number-words by the codec (both built in,
        // training-invariant), so there is nothing to climb.
        {
            var line0 = creators.Select(c =>
            {
                var h = heldByCreator[c.Name];
                var acc = h.Count(e => AnswerEquivalence.Equivalent(Gen(e.Input), e.Output)) / (double)Math.Max(1, h.Count);
                return $"{c.Name.Split(':')[^1],-14} {acc,4:P0}";
            });
            _out.WriteLine($"epoch 0:  " + string.Join("   ", line0) + "   (UNTRAINED)");
        }

        // Train FLAT, exactly like the race (TrainStep), measuring per-creator held-out after each epoch to see if
        // anything CLIMBS or everything saturates after epoch 1.
        for (var epoch = 1; epoch <= 4; epoch++)
        {
            foreach (var (i, o) in train.OrderBy(_ => rng.Next())) nova.Trainer.TrainStep(new GenesisExample(i, o));
            var line = creators.Select(c =>
            {
                var h = heldByCreator[c.Name];
                var acc = h.Count(e => AnswerEquivalence.Equivalent(Gen(e.Input), e.Output)) / (double)Math.Max(1, h.Count);
                return $"{c.Name.Split(':')[^1],-14} {acc,4:P0}";
            });
            _out.WriteLine($"epoch {epoch}:  " + string.Join("   ", line));
        }

        // For each creator, show WHAT nova does on 4 held-out items: answer, whether it's right, decision path.
        var noFunctionWordLeak = true;
        foreach (var c in creators)
        {
            _out.WriteLine($"\n── {c.Name} ──");
            foreach (var (i, o) in heldByCreator[c.Name].Take(4))
            {
                var r = Full(i);
                var ans = r.Output?.Trim() ?? "";
                var ok = AnswerEquivalence.Equivalent(ans, o);
                _out.WriteLine($"  {(ok ? "OK " : "XX ")} '{i}' -> '{ans}' (want '{o}') [{r.DecisionPath}]");
                if (FunctionWordLeak.Contains(ans)) noFunctionWordLeak = false; // it should ABSTAIN, never emit a framing word
            }
        }

        // GUARD the two fixes. Arithmetic is COMPUTE (homomorphism) — including negatives now — so it must be high;
        // before the unary-sign fix add/sub were ~20-30%. And the relax fallback must never emit a framing word.
        double Acc(string name) { var h = heldByCreator[name]; return h.Count(e => AnswerEquivalence.Equivalent(Gen(e.Input), e.Output)) / (double)Math.Max(1, h.Count); }
        var add = Acc("arithmetic:add"); var sub = Acc("arithmetic:sub");
        var mul = Acc("arithmetic:mul"); var div = Acc("arithmetic:div");
        _out.WriteLine($"\narithmetic held-out: add {add:P0}  sub {sub:P0}  mul {mul:P0}  div {div:P0}");
        Assert.True(add >= 0.90, $"add (negatives via the unary-sign fix) must compute; {add:P0}");
        Assert.True((add + sub + mul + div) / 4.0 >= 0.80, "arithmetic compute (incl. negatives) must hold");
        Assert.True(noFunctionWordLeak, "the relax fallback must abstain, never emit a framing word like 'what'/'to'/'by'");
    }

    private static readonly System.Collections.Generic.HashSet<string> FunctionWordLeak =
        new(StringComparer.OrdinalIgnoreCase) { "what", "who", "the", "a", "an", "of", "to", "is", "by", "for", "with", "and", "from", "as", "than", "in", "on", "at" };
}
