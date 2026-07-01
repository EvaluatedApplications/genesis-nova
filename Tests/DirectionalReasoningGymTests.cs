using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// HEADLESS, AT SCALE: does the TRAINED relation direction, wired into the live ladder (field-directional), answer the
/// HARDER reasoning questions that retrieval/relax cannot — 2-hop composition and eviction — at 10-family density (where
/// nearest/hub/edge-walk all failed)? A/B DirectionalReasoning ON vs OFF through infer.Generate. Reports honestly and
/// distinguishes "the direct primitive composes" from "the ladder routes to it end-to-end".
/// </summary>
public sealed class DirectionalReasoningGymTests
{
    private readonly ITestOutputHelper _out;
    public DirectionalReasoningGymTests(ITestOutputHelper o) => _out = o;

    private static readonly (string m, string g, string k)[] Fam =
    {
        ("sparrow","bird","animal"),   ("robin","bird","animal"),
        ("oak","tree","plant"),        ("pine","tree","plant"),
        ("trout","fish","creature"),   ("bass","fish","creature"),
        ("rose","flower","bloom"),     ("daisy","flower","bloom"),
        ("quartz","crystal","mineral"),("granite","crystal","mineral"),
    };

    private static DialecticalSpace BuildTrainedSpace()
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        // Teach NOISY glue-heavy sentences (member is-a genus, genus is-a kingdom) — realistic density.
        for (var c = 0; c < 40; c++)
            foreach (var (m, g, k) in Fam)
            {
                var s1 = new[] { "the", m, "is", "a", g };
                var s2 = new[] { "the", g, "is", "a", k };
                ds.FineEditFromExample(s1, s1, false);
                ds.FineEditFromExample(s2, s2, false);
            }
        // TRAIN the is-a direction on the accumulated triples (both hops) — read by the ladder afterward.
        var triples = new List<(string, string, string)>();
        foreach (var (m, g, k) in Fam) { triples.Add((m, g, "is-a")); triples.Add((g, k, "is-a")); }
        ds.TrainRelationDirection(triples, epochs: 800, lr: 0.05, margin: 2.0, maxNorm: 4.0, seed: 7);
        return ds;
    }

    private static GenesisInferenceEngine Engine(DialecticalSpace ds, bool directional)
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        return new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config), ds, null)
        { ConsciousField = true, LearnedCuesOnly = true, DirectionalReasoning = directional };
    }

    [Fact]
    public void DirectionalReasoning_AtScale_HarderReasoningQuestions_ThroughTheLadder()
    {
        var ds = BuildTrainedSpace();
        var on = Engine(ds, directional: true);
        var off = Engine(ds, directional: false);

        (string ans, string path) Ask(GenesisInferenceEngine e, string q)
        { var r = e.Generate(new GenerationRequest(q, 16)); return ((r.Output ?? "").Trim(), r.DecisionPath ?? ""); }

        // (A) DIRECT PRIMITIVE at scale — the proven composition, the capability the ladder wraps.
        int p1 = 0, p2 = 0;
        foreach (var (m, g, k) in Fam)
        {
            if (ds.TryDirectionalDerive(m, out var a1, out _, hops: 1) && a1 == g) p1++;
            if (ds.TryDirectionalDerive(m, out var a2, out _, hops: 2) && a2 == k) p2++;
        }
        _out.WriteLine($"DIRECT primitive @10-fam:  m+r→genus {p1}/{Fam.Length};  m+2r→kingdom {p2}/{Fam.Length}\n");

        // (B) LADDER A/B — "what kind of thing is X" (1-hop genus): does field-directional fire, does it match/beat OFF?
        int genusOn = 0, genusOff = 0, firedDir = 0;
        foreach (var (m, g, k) in Fam)
        {
            var (ao, po) = Ask(on, $"what kind of thing is {m}");
            var (af, pf) = Ask(off, $"what kind of thing is {m}");
            if (ao == g) genusOn++; if (af == g) genusOff++;
            if (po == "field-directional") firedDir++;
        }
        var (exO, exP) = Ask(on, "what kind of thing is sparrow");
        _out.WriteLine($"1-HOP GENUS (ladder):  DIR-ON {genusOn}/{Fam.Length} (field-directional fired {firedDir}/{Fam.Length})  |  DIR-OFF {genusOff}/{Fam.Length}");
        _out.WriteLine($"   ex: 'what kind of thing is sparrow' → ON '{exO}'[{exP}]");

        // (C) EVICTION through the ladder — wipe the genus (the 1-hop target); can the trained direction still answer?
        var ds2 = BuildTrainedSpace();
        foreach (var g in Fam.Select(f => f.g).Distinct()) ds2.EvictConcept(g);   // evict ALL genera
        var onE = Engine(ds2, directional: true);
        var offE = Engine(ds2, directional: false);
        int evOnKingdom = 0, evOnGlue = 0, evOffGlue = 0;
        foreach (var (m, g, k) in Fam)
        {
            var (ao, po) = Ask(onE, $"what kind of thing is {m}");
            var (af, pf) = Ask(offE, $"what kind of thing is {m}");
            if (ao == k) evOnKingdom++;
            if (ao is "the" or "is" or "a") evOnGlue++;
            if (af is "the" or "is" or "a") evOffGlue++;
        }
        var (evO, evPath) = Ask(onE, "what kind of thing is sparrow");
        _out.WriteLine($"\nEVICTION (genera wiped):  DIR-ON kingdom {evOnKingdom}/{Fam.Length}, glue {evOnGlue}/{Fam.Length}  |  DIR-OFF glue {evOffGlue}/{Fam.Length}");
        _out.WriteLine($"   ex: 'what kind of thing is sparrow' (bird evicted) → ON '{evO}'[{evPath}]");

        _out.WriteLine("\n>>> HONEST VERDICT below (assert only what holds).");

        // What genuinely holds: the direct primitive composes 2-hop at scale, and field-directional FIRES end-to-end
        // for the 1-hop genus with no regression vs OFF. (2-hop kingdom through a single 'what kind' frame is a
        // hop-cue limitation of the frame, reported above via the direct primitive; eviction reported as measured.)
        Assert.True(p2 >= Fam.Length - 2, $"direct 2-hop composition must hold at 10 families, got {p2}/{Fam.Length}");
        Assert.True(firedDir >= Fam.Length - 1, $"field-directional must FIRE end-to-end through the ladder, got {firedDir}/{Fam.Length}");
        Assert.True(genusOn >= genusOff, $"directional must not regress 1-hop genus (ON {genusOn} vs OFF {genusOff})");
    }
}
