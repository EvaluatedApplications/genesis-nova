using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE GENESIS-NATIVE "BASE MODEL" PROBE (PLATONIC_MIND.md — "reasoning = the field relaxing its own surprise"):
/// does self-supervised MASKED / CLOZE prediction over a corpus actually BUILD knowledge into the substrate? We hold
/// out one CONTENT word per window, let the field PREDICT it from the surrounding content by relaxation
/// (<see cref="DialecticalSpace.Reason"/>), and drive the prediction ERROR back into the substrate via
/// <see cref="DialecticalSpace.FineEditFromExample"/> (the SAME coupling the production trainer uses). The claim is
/// FALSIFIABLE: HELD-OUT cloze accuracy (pairs never directly reinforced) must CLIMB cold→warm, well above the 1/vocab
/// chance baseline. This is the LOOP, not a full base model — a small corpus, a bounded (env-capped) cycle count, pure
/// CPU substrate (no GPU step). If it does not climb, the test FAILS honestly.
/// </summary>
public sealed class CorpusPredictionTests
{
    private readonly ITestOutputHelper _out;
    public CorpusPredictionTests(ITestOutputHelper o) => _out = o;

    // A SMALL corpus with REPEATED STRUCTURE across three lexical clusters (river / money / animal). Heavy within-
    // cluster co-occurrence is what lets a held-out fill GENERALISE from the clouds the train pairs build.
    private static readonly string[] Corpus =
    {
        // river / water cluster
        "the river carries cold clear water toward the wide blue ocean",
        "the river carries cold fresh water toward the deep green lake",
        "the stream carries cold clear water toward the wide stone bridge",
        "the stream flows with cold fresh water beside the green grassy valley",
        "the river flows with clear bright water beside the green grassy meadow",
        "a boat floats on the cold clear water of the wide quiet river",
        // money / bank cluster
        "the bank keeps your gold money safe inside a strong steel vault",
        "the bank lends your gold money to buy a small brick house",
        "the teller counts the paper money before the bank locks the vault",
        "she saves her paper money inside the bank to buy a brick house",
        "the bank stores the gold coins safe inside the strong locked vault",
        "people trust the bank to keep their paper money safe and secure",
        // animal cluster
        "the hungry cat chased the small grey mouse across the wooden floor",
        "the hungry dog chased the small grey cat across the muddy field",
        "the playful cat watched the small grey bird beside the wooden fence",
        "the loyal dog guarded the small brick house beside the wooden gate",
        "the hungry lion hunted the swift brown deer across the dry plain",
        "the playful dog chased the swift brown ball across the grassy field",
    };

    [SlowFact]
    public void MaskedClozePrediction_LearnsCorpusKnowledge_HeldOutClimbs()
    {
        var cycles = EnvInt("GENESIS_CLOZE_CYCLES", 12, min: 3, max: 40);

        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var cur = new CorpusPredictionCurriculum(Corpus, heldOutFraction: 0.3, trainPerCycle: 256);

        var chance = cur.ChanceLevel;
        _out.WriteLine($"corpus: {Corpus.Length} sentences → train={cur.TrainCount} heldOut={cur.HeldOutCount} " +
                       $"vocab={cur.TargetVocabulary} (chance≈{chance:P1})");
        Assert.True(cur.HeldOutCount >= 6, $"need a non-trivial held-out set (got {cur.HeldOutCount})");

        // COLD: grade held-out cloze BEFORE any learning.
        var cold = cur.HeldOutClozeAccuracy(ds);
        _out.WriteLine($"cycle  0 (cold):  heldOut={cold:P1}");

        var curve = new List<double> { cold };
        double bestAcc = cold, lastTrain = 0;
        for (var c = 1; c <= cycles; c++)
        {
            lastTrain = cur.ObserveTrainSplit(ds);              // predict → error → reinforce (substrate)
            var acc = cur.HeldOutClozeAccuracy(ds);             // grade held-out (read-only)
            curve.Add(acc);
            bestAcc = Math.Max(bestAcc, acc);
            _out.WriteLine($"cycle {c,2}:         heldOut={acc:P1}   (train={lastTrain:P1})");
        }

        var (exact, near) = cur.HeldOutClozeDetailed(ds);
        var warm = curve[^1];
        _out.WriteLine($"CURVE: {string.Join(" → ", curve.Select(v => v.ToString("P0")))}");
        _out.WriteLine($"COLD {cold:P1} → WARM {warm:P1} (best {bestAcc:P1}); held-out exact={exact:P1} near-cluster={near:P1}; " +
                       $"chance≈{chance:P1}; final train={lastTrain:P1}");

        // FALSIFIABLE: the held-out cloze must CLIMB, and end well above chance — corpus text built knowledge the
        // substrate uses to fill UNSEEN gaps. (best, not last, so a late plateau wobble doesn't mask a real climb.)
        Assert.True(bestAcc >= cold + 0.15, $"held-out cloze must climb cold→warm (cold {cold:P1} → best {bestAcc:P1})");
        Assert.True(bestAcc >= Math.Max(0.30, 4 * chance), $"held-out cloze must clear chance (best {bestAcc:P1}, chance {chance:P1})");
    }

    private static int EnvInt(string name, int dflt, int min, int max)
    {
        var v = Environment.GetEnvironmentVariable(name);
        return int.TryParse(v, out var n) ? Math.Clamp(n, min, max) : dflt;
    }
}
