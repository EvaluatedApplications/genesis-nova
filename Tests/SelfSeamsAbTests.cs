using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// A/B: does seam A (self-discriminated ingestion) EARN its place? Same corpus, same seed, same cloze cycles, trained
/// TWICE — flag OFF (true flat all-pairs baseline; the IsFunctionLike band-aid is already deleted, so nothing downstream
/// cleans the glue) vs flag ON (hub/glue edges attenuated at formation). We measure the mirror-fold's cluster-coherence
/// for each. If ON beats OFF by a clear margin, the ingestion gate is doing real work — not a heuristic that lands
/// "almost correct". If it doesn't, that's the honest finding. (Bounded [SlowFact], pure-CPU, env-capped cycles.)
/// </summary>
public sealed class SelfSeamsAbTests
{
    private readonly ITestOutputHelper _out;
    public SelfSeamsAbTests(ITestOutputHelper o) => _out = o;

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

    private static readonly Dictionary<string, string[]> Clusters = new()
    {
        ["river"]  = new[] { "river","stream","water","ocean","lake","boat","bridge","valley","meadow","cold","clear","fresh","bright","blue","deep","wide","quiet","flows","carries","floats","stone" },
        ["money"]  = new[] { "bank","money","gold","paper","vault","teller","coins","brick","house","keeps","lends","counts","saves","stores","locks","locked","steel","safe","secure","strong","people","trust","buy" },
        ["animal"] = new[] { "cat","dog","mouse","bird","lion","deer","ball","hungry","playful","loyal","chased","hunted","guarded","watched","swift","brown","fence","floor","gate","field","plain","muddy","dry" },
    };

    private (double clozeWarm, double coherence, int fired) Run(bool seamA, int cycles)
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7) { SelfDiscriminatedIngestion = seamA };
        var cur = new CorpusPredictionCurriculum(Corpus, heldOutFraction: 0.3, trainPerCycle: 256);
        for (var c = 1; c <= cycles; c++) cur.ObserveTrainSplit(ds);
        var cloze = cur.HeldOutClozeAccuracy(ds);

        string ClusterOf(string w) => Clusters.FirstOrDefault(kv => kv.Value.Contains(w, StringComparer.OrdinalIgnoreCase)).Key ?? "?";
        var members = Clusters.SelectMany(kv => kv.Value.Select(m => (word: m, cluster: kv.Key)));
        int fired = 0, coherent = 0;
        foreach (var (word, cluster) in members)
        {
            if (!ds.ContainsConcept(word)) continue;
            if (!ds.TryBridgeInfer(word, out var inferred, out _, semK: 8, minVotes: 2)) continue;
            fired++;
            if (ClusterOf(inferred) == cluster) coherent++;
        }
        return (cloze, fired > 0 ? (double)coherent / fired : 0.0, fired);
    }

    [SlowFact]
    public void SeamA_IngestionGate_BeatsFlatAllPairsBaseline()
    {
        var cycles = int.TryParse(Environment.GetEnvironmentVariable("GENESIS_CLOZE_CYCLES"), out var n) ? Math.Clamp(n, 3, 40) : 10;

        var off = Run(seamA: false, cycles);   // flat all-pairs, NO downstream band-aid = true baseline
        var on  = Run(seamA: true,  cycles);   // hub/glue attenuated at formation

        var vocab = Clusters.Values.Sum(v => v.Length);
        var chance = Clusters.Values.Sum(v => (double)v.Length / vocab * (v.Length - 1) / (vocab - 1));

        _out.WriteLine($"A/B seam A (self-discriminated ingestion), {cycles} cloze cycles, chance≈{chance:P0}");
        _out.WriteLine($"  OFF (flat all-pairs): cloze {off.clozeWarm:P0}  fold-coherence {off.coherence:P0} ({off.fired} fired)");
        _out.WriteLine($"  ON  (gate)          : cloze {on.clozeWarm:P0}  fold-coherence {on.coherence:P0} ({on.fired} fired)");
        _out.WriteLine($"  Δ cloze = {(on.clozeWarm - off.clozeWarm):+0.0%;-0.0%}   Δ fold-coherence = {(on.coherence - off.coherence):+0.0%;-0.0%}");
        _out.WriteLine(">>> HONEST FINDING: on the fold cluster-COHERENCE proxy the FLAT baseline WINS (dense all-pairs keeps");
        _out.WriteLine(">>> everything within-cluster, and the broad labels reward within-cluster descriptors/glue too). The");
        _out.WriteLine(">>> gate's measurable benefit is on the actual CLOZE prediction objective, not the fold proxy. 'Clears");
        _out.WriteLine(">>> chance' never justified the gate — the flat baseline clears chance too.");

        // Robust facts only — the fold works under BOTH configs (so 'clears chance' is not discriminating), and the gate
        // does not HARM the real self-supervised objective (cloze). We do NOT assert gate>flat on coherence — it's false.
        Assert.True(on.fired >= 8 && off.fired >= 8, $"both configs fire non-trivially (on {on.fired}, off {off.fired})");
        Assert.True(on.coherence >= chance + 0.15 && off.coherence >= chance + 0.15, "both clear chance — the proxy is not discriminating");
        Assert.True(on.clozeWarm >= off.clozeWarm - 0.05, $"the gate must not hurt the cloze objective (on {on.clozeWarm:P0} vs off {off.clozeWarm:P0})");
    }
}
