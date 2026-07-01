using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Does the relational fold ride GENERAL learning (not a crafted curriculum)? We train ONLY the self-supervised corpus
/// cloze objective (mask a word, predict from context, drive the error back — no category labels, no held-out generator,
/// no "X is-a Y" teaching). The corpus is plain sentences over three lexical clusters. AFTER general training we ask the
/// mirror-fold (TryBridgeInfer — the SAME primitive the hot-path bridge rung calls) to infer a property each member
/// LACKS from its embedding-neighbours, and score it by CLUSTER COHERENCE. The cluster labels are used ONLY to grade —
/// never to train. If the fold recovers cluster structure well above chance, it generalises off whatever clouds general
/// learning built; if not, that's an honest no. (Bounded [SlowFact], pure-CPU substrate, env-capped cycles.)
/// </summary>
public sealed class RelationalFoldGeneralLearningTests
{
    private readonly ITestOutputHelper _out;
    public RelationalFoldGeneralLearningTests(ITestOutputHelper o) => _out = o;

    // Plain corpus, heavy WITHIN-cluster co-occurrence (the only structure general learning can exploit). No labels.
    private static readonly string[] Corpus =
    {
        "the river carries cold clear water toward the wide blue ocean",
        "the river carries cold fresh water toward the deep green lake",
        "the stream carries cold clear water toward the wide stone bridge",
        "the stream flows with cold fresh water beside the green grassy valley",
        "the river flows with clear bright water beside the green grassy meadow",
        "a boat floats on the cold clear water of the wide quiet river",
        "the bank keeps your gold money safe inside a strong steel vault",
        "the bank lends your gold money to buy a small brick house",
        "the teller counts the paper money before the bank locks the vault",
        "she saves her paper money inside the bank to buy a brick house",
        "the bank stores the gold coins safe inside the strong locked vault",
        "people trust the bank to keep their paper money safe and secure",
        "the hungry cat chased the small grey mouse across the wooden floor",
        "the hungry dog chased the small grey cat across the muddy field",
        "the playful cat watched the small grey bird beside the wooden fence",
        "the loyal dog guarded the small brick house beside the wooden gate",
        "the hungry lion hunted the swift brown deer across the dry plain",
        "the playful dog chased the swift brown ball across the grassy field",
    };

    // EVALUATION ground truth ONLY (never trained on): a word belongs to the cluster of the SENTENCES it appears in —
    // the fair distributional label, DESCRIPTORS included (cloze builds descriptive co-occurrence, so a river member's
    // real neighbours are quiet/fresh/flows, not only the noun 'river'). Grading by canonical nouns alone is unfair.
    private static readonly Dictionary<string, string[]> Clusters = new()
    {
        ["river"]  = new[] { "river","stream","water","ocean","lake","boat","bridge","valley","meadow","cold","clear","fresh","bright","blue","deep","wide","quiet","flows","carries","floats","stone" },
        ["money"]  = new[] { "bank","money","gold","paper","vault","teller","coins","brick","house","keeps","lends","counts","saves","stores","locks","locked","steel","safe","secure","strong","people","trust","buy" },
        ["animal"] = new[] { "cat","dog","mouse","bird","lion","deer","ball","hungry","playful","loyal","chased","hunted","guarded","watched","swift","brown","fence","floor","gate","field","plain","muddy","dry" },
    };

    [SlowFact]
    public void GeneralCorpusLearning_TheFold_RecoversClusterStructure_AboveChance()
    {
        var cycles = int.TryParse(Environment.GetEnvironmentVariable("GENESIS_CLOZE_CYCLES"), out var n) ? Math.Clamp(n, 3, 40) : 12;

        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7) { SelfDiscriminatedIngestion = true };
        var cur = new CorpusPredictionCurriculum(Corpus, heldOutFraction: 0.3, trainPerCycle: 256);

        // GENERAL self-supervised training — predict masked word, reinforce the error. No labels, no fold-shaped data.
        var cold = cur.HeldOutClozeAccuracy(ds);
        double train = 0;
        for (var c = 1; c <= cycles; c++) train = cur.ObserveTrainSplit(ds);
        var warm = cur.HeldOutClozeAccuracy(ds);
        _out.WriteLine($"GENERAL cloze objective: cold {cold:P0} → warm {warm:P0} (train {train:P0}) over {cycles} cycles\n");

        // Members that are SPARSE in the corpus = the natural held-out gaps general coverage leaves (no crafted holdout).
        var members = Clusters.SelectMany(kv => kv.Value.Select(m => (word: m, cluster: kv.Key))).ToList();
        string ClusterOf(string w) => Clusters.FirstOrDefault(kv => kv.Value.Contains(w, StringComparer.OrdinalIgnoreCase)).Key ?? "?";

        int fired = 0, coherent = 0;
        foreach (var (word, cluster) in members)
        {
            if (!ds.ContainsConcept(word)) continue;
            if (!ds.TryBridgeInfer(word, out var inferred, out var conf, semK: 8, minVotes: 2)) continue;
            fired++;
            var landed = ClusterOf(inferred);
            var ok = landed == cluster;
            if (ok) coherent++;
            _out.WriteLine($"FOLD {word,-7} → '{inferred,-8}' [{landed,-6}] want {cluster,-6} conf={conf:F2} {(ok ? "coherent" : "—")}");
        }

        // Chance = if the fold inferred a random content word, P(same cluster). Weighted by cluster sizes.
        var vocab = Clusters.Values.Sum(v => v.Length);
        var chance = Clusters.Values.Sum(v => (double)v.Length / vocab * (v.Length - 1) / (vocab - 1));
        var rate = fired > 0 ? (double)coherent / fired : 0.0;
        _out.WriteLine($"\nFOLD cluster-coherence: {coherent}/{fired} = {rate:P0}   (chance≈{chance:P0})");
        _out.WriteLine(rate > chance + 0.15
            ? ">>> the fold RECOVERS cluster structure off GENERAL learning — no crafted curriculum, it reads the clouds general training built."
            : ">>> the fold did not clear chance on this general substrate — honest no (general cloze may not build the relational edges the fold needs).");

        Assert.True(cur.HeldOutCount >= 6, "non-trivial held-out cloze set");
        Assert.True(fired >= 8, $"the fold should fire on a non-trivial number of members after general training (fired {fired})");
        Assert.True(rate >= chance + 0.15, $"the fold must recover cluster structure above chance (got {rate:P0}, chance {chance:P0})");
    }
}
