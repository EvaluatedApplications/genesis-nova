using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE VITAL LOOP, CLOSED THROUGH THE EXISTING SELF (PLATONIC_CONSCIOUSNESS.md). The query-conditioned navigator
/// (<see cref="NavigatorQueryConditioned"/>) relaxes a query (anchor, cue) to an emergent answer. Here that walk is
/// wired into the agent's ONE persistent self — <c>GenesisInferenceEngine._selfField</c>, the SAME decaying meaning-
/// space self that conditions <c>ds.Reason</c> today (no new self class is invented):
///   • WRITE  : <c>engine.PerceiveSelf(symbol)</c> folds a traversed concept's cloud into the existing self.
///   • READ   : the net seeds h₀ = tanh(W·anchor + cueEmb[cue] + W_s·<c>engine.SelfField</c>) — the self tilts the seed.
///   • ABLATE : <c>engine.SelfConditionsCognition</c> = false (pass no self) collapses to the bare-geometry seed.
///
/// THE PROOF (load-bearing, not decorative): an AMBIGUOUS concept is planted is-a BOTH an animal genus AND a plant
/// genus. Under the DOMAIN cue alone the bare walk is a coin-flip. Prime the existing self with ANIMAL context and the
/// SAME query relaxes to "animal"; prime it with PLANT context and it relaxes to "plant"; ablate the self and it falls
/// to the un-conditioned default. The self — and only the self — changes the answer. Gated [SlowFact] (RUN_SLOW=1);
/// production sizing (Dim 1024, Hidden 2048); small epochs to run in a few minutes on CUDA. HONEST MEASUREMENT.
/// </summary>
public sealed class NavigatorSelfLoop
{
    private readonly ITestOutputHelper _out;
    public NavigatorSelfLoop(ITestOutputHelper o) => _out = o;

    private const int Dim = 1024;        // production FaceDimension
    private const int Hidden = 2048;     // production HiddenSize (the net trunk)
    private const int K = 16;
    private const int BcEpochs = 24;     // FAST
    private const int DaggerRounds = 1;  // FAST
    private const int DaggerEpochs = 24; // FAST
    private const int AmbUpweight = 3;   // the ambiguous forks are the lesson — upweight them vs the easy navigation bulk
    private const double Lr = 1e-3;
    private const int MaxSteps = 12;
    private const int ObservePerEdge = 4;

    private const string Root = "organism";
    private static readonly int CueGenus = (int)NavCue.Genus;
    private static readonly int CueDomain = (int)NavCue.Domain;

    // ── A real (compact) taxonomy: organism → {animal,plant} → genera → members. ──
    private static readonly (string Domain, (string Genus, string[] Members)[] Genera)[] Taxonomy =
    {
        ("animal", new[]
        {
            ("mammal", new[] { "dog","cat","horse","cow","lion","tiger","wolf","bear","deer","fox","rabbit","sheep",
                               "goat","pig","mouse","elephant","zebra","monkey","whale","camel","donkey" }),
            ("bird", new[] { "robin","sparrow","eagle","owl","hawk","crow","finch","duck","goose","swan","pigeon","parrot" }),
            ("fish", new[] { "salmon","trout","tuna","cod","bass","shark","herring" }),
            ("reptile", new[] { "snake","lizard","turtle","crocodile","gecko","iguana" }),
            ("insect", new[] { "ant","bee","wasp","beetle","moth","fly","cricket","ladybug","termite","mosquito" }),
        }),
        ("plant", new[]
        {
            ("tree", new[] { "oak","pine","maple","birch","willow","elm","cedar","spruce","ash","fir","beech" }),
            ("flower", new[] { "rose","tulip","daisy","lily","orchid","sunflower","violet","poppy" }),
            ("grass", new[] { "wheat","corn","rice","barley","oat","rye" }),
            ("vegetable", new[] { "carrot","potato","onion","tomato","pea","bean","cabbage" }),
            ("berry", new[] { "strawberry","blueberry","raspberry","blackberry","cranberry" }),
        }),
    };

    // Ambiguous TRAINING concepts (nonce names) — each is-a one animal genus AND one plant genus. The lesson the net
    // must learn is general: "at a two-parent fork, let the accumulated self pick the domain-consistent branch". The
    // held-out TEST concept's (mammal,tree) FORK is taught here on SIBLING nonce concepts (wug/dax/blicket), but the
    // TEST INSTANCE 'chimera' is never in any training query — so the headline is held-out-instance generalization.
    private static readonly (string A, string AnimalGenus, string PlantGenus)[] AmbTrain =
    {
        ("wug", "mammal", "tree"), ("dax", "mammal", "tree"), ("blicket", "mammal", "tree"), // chimera's fork (siblings)
        ("qux", "mammal", "flower"), ("zog", "bird", "tree"), ("blorp", "fish", "grass"),
        ("fritz", "reptile", "vegetable"), ("vlim", "insect", "berry"), ("snarl", "bird", "flower"),
        ("kobi", "fish", "tree"), ("plon", "reptile", "flower"), ("gar", "insect", "grass"),
    };
    private const string TestAmbiguous = "chimera";          // is-a {mammal, tree} — HELD OUT (never in a training query)
    private const string TestAnimalGenus = "mammal";
    private const string TestPlantGenus = "tree";

    // Broad domain context primed into the EXISTING self (one member per genus → a domain-level centroid, not a genus one).
    private static readonly string[] AnimalContext = { "dog", "eagle", "salmon", "snake", "bee", "wolf" };
    private static readonly string[] PlantContext = { "oak", "rose", "wheat", "carrot", "strawberry", "pine" };

    private sealed record World(DialecticalSpace Space, Dictionary<string, string> Parent, List<string> Members);

    private static World Build()
    {
        var space = new DialecticalSpace(Dim, seed: 7);
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);
        var members = new List<string>();
        void Plant(string child, string par)
        {
            parent[child] = par;
            for (var i = 0; i < ObservePerEdge; i++) space.ObserveContradiction(child, par, 0.0);
        }

        foreach (var (domain, genera) in Taxonomy)
        {
            Plant(domain, Root);
            foreach (var (genus, mem) in genera)
            {
                Plant(genus, domain);
                foreach (var m in mem) { Plant(m, genus); members.Add(m); }
            }
        }

        // Ambiguous TRAINING concepts: two parents each (NOT recorded in the single-parent map).
        void Amb(string a, string g1, string g2)
        {
            for (var i = 0; i < ObservePerEdge; i++) { space.ObserveContradiction(a, g1, 0.0); space.ObserveContradiction(a, g2, 0.0); }
        }
        foreach (var (a, ag, pg) in AmbTrain) Amb(a, ag, pg);
        Amb(TestAmbiguous, TestAnimalGenus, TestPlantGenus); // the held-out test concept exists, but is never trained

        space.FlushCloudBatch();
        return new World(space, parent, members);
    }

    private static string Genus(Dictionary<string, string> parent, string m) => parent.TryGetValue(m, out var p) ? p : "";
    private static string DomainOf(Dictionary<string, string> parent, string m)
    {
        var cur = m; var guard = 0;
        while (guard++ < 32 && parent.TryGetValue(cur, out var p)) { if (p == Root) return cur; cur = p; }
        return "";
    }

    private static GenesisInferenceEngine NewEngine(DialecticalSpace space)
    {
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config: new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize));
        return new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true, SelfConditionsCognition = true };
    }

    // Prime the EXISTING self by perceiving several domain members (the real WRITE), then read the self back.
    private static double[] PrimeSelf(DialecticalSpace space, string[] context)
    {
        var e = NewEngine(space);
        foreach (var c in context) e.PerceiveSelf(c);
        return e.SelfField.ToArray();
    }

    // One walk seeded by (anchor, cue) AND the given persistent self (null → un-conditioned). Returns where it relaxed to.
    private static (string Final, bool Halted, IReadOnlyList<string> Traj) WalkSelf(
        NavQueryPolicyNet net, DialecticalSpace space, string member, int cue, Device device, double[]? self)
    {
        if (!space.TryGetConceptFace(member, out var anchor)) return (member, false, new[] { member });
        using var policy = new QueryNavPolicy(net, space, anchor, cue, device, K, 0.0, 0.5, self);
        var res = new NavigatorWalk().Walk(space, member, anchor, null, policy, new NavWalkOptions(MaxSteps: MaxSteps));
        return (res.FinalSymbol, policy.LastHalt, res.Trajectory);
    }

    [SlowFact]
    public void TheExistingSelf_ConditionsTheNavigatorWalk_AnimalVsPlantPrimedDiverge_OnTheSameAmbiguousQuery()
    {
        var sw = Stopwatch.StartNew();
        var device = cuda.is_available() ? CUDA : CPU;
        manual_seed(7);

        var w = Build();
        var space = w.Space;

        // ── Prime the EXISTING self two ways, via engine.PerceiveSelf (the real WRITE into _selfField). ──
        var animalSelf = PrimeSelf(space, AnimalContext);
        var plantSelf = PrimeSelf(space, PlantContext);
        Assert.NotEmpty(animalSelf);
        Assert.Equal(animalSelf.Length, plantSelf.Length);
        _out.WriteLine($"=== Self (engine._selfField) primed from PerceiveSelf: len={animalSelf.Length} | " +
                       $"animal·plant cos={Cos(animalSelf, plantSelf):F3} (two different selves) | device={device.type} ===");

        using var net = new NavQueryPolicyNet(Dim, NavQueryFeatures.CueCount, Hidden, device);
        Assert.Equal(animalSelf.Length, net.SelfLength); // engine self width == W_s input width
        _out.WriteLine($"=== Net: F={net.FeatureLength}; selfLen={net.SelfLength}; params={net.ParameterCount():N0} ===");

        // ── Build the curriculum: plain navigation (no self) + ambiguous forks (self IS the only disambiguator). ──
        var q = new List<NavQueryDaggerTrainer.Query>();
        foreach (var m in w.Members.Where((_, i) => i % 2 == 0))
        {
            var g = Genus(w.Parent, m); var d = DomainOf(w.Parent, m);
            if (g.Length > 0) q.Add(new(m, CueGenus, g));
            if (d.Length > 0) q.Add(new(m, CueDomain, d));
        }
        var aSelf = ToFloat(animalSelf); var pSelf = ToFloat(plantSelf);
        for (var dup = 0; dup < AmbUpweight; dup++)
            foreach (var (a, ag, pg) in AmbTrain)
            {
                q.Add(new(a, CueDomain, "animal", aSelf));
                q.Add(new(a, CueDomain, "plant", pSelf));
                q.Add(new(a, CueGenus, ag, aSelf));
                q.Add(new(a, CueGenus, pg, pSelf));
            }
        _out.WriteLine($"=== Curriculum: {q.Count} queries ({AmbTrain.Length} ambiguous self-conditioned concepts; " +
                       $"'{TestAmbiguous}' held out) ===");

        // ── BC warm-start → 1 DAgger round (self threads through both). ──
        var bc = NavQueryDaggerTrainer.BuildQueryTrajectories(space, q, K);
        Assert.NotEmpty(bc);
        var bcLoss = NavQueryDaggerTrainer.TrainQuery(net, bc, BcEpochs, Lr, device, K);
        _out.WriteLine($"=== BC: {bc.Count} trajectories | CE={bcLoss.CrossEntropy:F4} haltBCE={bcLoss.HaltBce:F4} ===");

        var agg = new List<NavQueryTrajectory>(bc);
        for (var r = 1; r <= DaggerRounds; r++)
        {
            var roll = NavQueryDaggerTrainer.RolloutQueryTrajectories(net, space, q, device, MaxSteps, K);
            agg.AddRange(roll);
            var loss = NavQueryDaggerTrainer.TrainQuery(net, agg, DaggerEpochs, Lr, device, K);
            _out.WriteLine($"=== DAgger {r}: +{roll.Count} (agg {agg.Count}) | CE={loss.CrossEntropy:F4} haltBCE={loss.HaltBce:F4} ===");
        }

        // ═══════════ THE HEADLINE — SAME ambiguous query, three self states, three outcomes ═══════════
        var animal = WalkSelf(net, space, TestAmbiguous, CueDomain, device, animalSelf);
        var plant = WalkSelf(net, space, TestAmbiguous, CueDomain, device, plantSelf);
        var ablated = WalkSelf(net, space, TestAmbiguous, CueDomain, device, null); // SelfConditionsCognition=false ⇒ no self
        _out.WriteLine("=== [HEADLINE] (chimera, DOMAIN) — same query, the existing self is the only thing that changes ===");
        _out.WriteLine($"    animal-primed self → '{animal.Final}'  [{string.Join("→", animal.Traj)}]");
        _out.WriteLine($"    plant-primed  self → '{plant.Final}'  [{string.Join("→", plant.Traj)}]");
        _out.WriteLine($"    ABLATED (no self)  → '{ablated.Final}'  [{string.Join("→", ablated.Traj)}]");

        // ═══════════ THE WRITE SIDE — close the loop THROUGH the existing self (walk → PerceiveSelf → next walk) ═══════
        // ONE engine, ONE self. It lives among animals (PerceiveSelf), reads chimera; then it walks and feeds back the
        // concepts it traversed (the loop is closed through _selfField); then it goes on to live among PLANTS — and the
        // SAME query now reads the other way. The WRITE genuinely re-conditions cognition; it is not a fixed point.
        var loopEngine = NewEngine(space);
        foreach (var c in AnimalContext) loopEngine.PerceiveSelf(c);
        var selfBefore = loopEngine.SelfField.ToArray();
        var loop1 = WalkSelf(net, space, TestAmbiguous, CueDomain, device, selfBefore);
        foreach (var s in loop1.Traj) loopEngine.PerceiveSelf(s);          // the walk's traversal updates the EXISTING self
        var selfAfter = loopEngine.SelfField.ToArray();
        foreach (var c in PlantContext) loopEngine.PerceiveSelf(c);        // ...then the mind goes on to live among plants
        var loop2 = WalkSelf(net, space, TestAmbiguous, CueDomain, device, loopEngine.SelfField.ToArray());
        var selfMoved = Cos(selfBefore, selfAfter) < 0.999;               // PerceiveSelf actually wrote into the same self
        _out.WriteLine("=== [WRITE LOOP] walk → engine.PerceiveSelf(traversed) → keep living → walk again (closed through _selfField) ===");
        _out.WriteLine($"    pass1 (lived among animals) → '{loop1.Final}'   self moved by feedback (cos before/after={Cos(selfBefore, selfAfter):F3})   " +
                       $"pass2 (then among plants) → '{loop2.Final}'");

        // ═══════════ NO-REGRESSION — an UNAMBIGUOUS query still resolves with the self in the loop ═══════════
        var dogDomain = WalkSelf(net, space, "dog", CueDomain, device, animalSelf);
        var dogGenus = WalkSelf(net, space, "dog", CueGenus, device, animalSelf);
        _out.WriteLine($"=== [NO-REGRESSION] (dog, DOMAIN)→'{dogDomain.Final}'  (dog, GENUS)→'{dogGenus.Final}' (self in the loop) ===");

        sw.Stop();
        _out.WriteLine("=== SUMMARY ===");
        _out.WriteLine($"    wall = {sw.Elapsed.TotalSeconds:F1}s on {device.type}");
        _out.WriteLine($"    HEADLINE  animal→{animal.Final}  plant→{plant.Final}  ablated→{ablated.Final}");
        _out.WriteLine($"    LOOP      pass1(animals)→{loop1.Final}  pass2(plants)→{loop2.Final}  selfMoved={selfMoved}");
        _out.WriteLine($"    NOREG     dog/DOMAIN→{dogDomain.Final}  dog/GENUS→{dogGenus.Final}");

        // ── Assertions (HONEST; the DATA above is the result). The self is load-bearing iff it changes the answer. ──
        Assert.True(animal.Halted && plant.Halted, "both primed walks must confidently halt (not hang/abstain)");
        Assert.NotEqual(animal.Final, plant.Final);                 // READ: the existing self CHANGES the answer
        Assert.Equal("animal", animal.Final);                        // animal-context self → animal
        Assert.Equal("plant", plant.Final);                          // plant-context self → plant
        Assert.Equal("animal", dogDomain.Final);                     // unambiguous navigation intact with self in the loop
        Assert.Equal("mammal", dogGenus.Final);
        // WRITE: engine.PerceiveSelf wrote into the SAME self, and continuing to live re-conditions the SAME query.
        Assert.True(selfMoved, "the walk's traversal must actually fold into the existing _selfField (the loop is closed)");
        Assert.Equal("animal", loop1.Final);                         // among animals → animal
        Assert.Equal("plant", loop2.Final);                          // then among plants → plant (the WRITE flips cognition)
    }

    private static float[] ToFloat(double[] x) { var r = new float[x.Length]; for (var i = 0; i < x.Length; i++) r[i] = (float)x[i]; return r; }
    private static double Cos(double[] a, double[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-9);
    }
}
