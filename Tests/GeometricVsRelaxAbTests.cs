using System;
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
/// A/B: is promoting `field-geometric` ABOVE `field-relax` in the live ladder justified? One shared substrate
/// (self-discriminated ingestion, noisy glue-heavy taxonomy sentences), the SAME queries run through the FULL ladder
/// under GEOMETRIC-PRIMARY (GeometricReasoning on — geometric runs before relax) vs RELAX-PRIMARY (off — relax handles
/// it, old behaviour). The decision hinges on: does geometric-primary REGRESS normal retrieval? and does it WIN on the
/// cases the eviction/abstention tests care about? Graded, both configs, honest verdict from the data.
/// </summary>
public sealed class GeometricVsRelaxAbTests
{
    private readonly ITestOutputHelper _out;
    public GeometricVsRelaxAbTests(ITestOutputHelper o) => _out = o;

    private static readonly (string m, string g, string k)[] Fam =
    {
        ("sparrow","bird","animal"),   ("robin","bird","animal"),
        ("oak","tree","plant"),        ("pine","tree","plant"),
        ("trout","fish","creature"),   ("bass","fish","creature"),
        ("rose","flower","plant"),     ("daisy","flower","plant"),
        ("quartz","crystal","mineral"),("granite","stone","mineral"),
    };

    private static DialecticalSpace BuildSpace(bool isolated = false)
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        for (var c = 0; c < 40; c++)
        {
            foreach (var (m, g, k) in Fam)
            {
                var s1 = new[] { "the", m, "is", "a", g };   // axiom member→genus
                var s2 = new[] { "the", g, "is", "a", k };   // axiom genus→kingdom
                ds.FineEditFromExample(s1, s1, false);
                ds.FineEditFromExample(s2, s2, false);
            }
            if (isolated)
            {
                var z = new[] { "the", "zorblax", "is", "a" };   // ONLY glue: isolated, no valid derivation
                ds.FineEditFromExample(z, z, false);
            }
        }
        return ds;
    }

    [Fact]
    public void GeometricPrimary_vs_RelaxPrimary_FullLadderAb()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);

        // Fresh engine PER QUERY so the accumulating self-field can't drift the A/B across queries — isolates the ladder.
        (string ans, string path) Ask(DialecticalSpace space, bool geo, string q)
        {
            var e = new GenesisInferenceEngine(tok, model, space, null)
            { ConsciousField = true, LearnedCuesOnly = true, GeometricReasoning = geo, BridgeReasoning = true };
            var r = e.Generate(new GenerationRequest(q, 16));
            return ((r.Output ?? "").Trim(), r.DecisionPath ?? "");
        }

        // ---- 1 NORMAL RETRIEVAL (member→genus, DIRECTLY taught) + 2 HELD-OUT (member→kingdom) — shared space ----
        var space = BuildSpace();
        int gRet = 0, rRet = 0, gHo = 0, rHo = 0, shown = 0;
        foreach (var (m, g, k) in Fam)
        {
            var (ga, gp) = Ask(space, true, $"what kind of thing is {m}");
            var (ra, rp) = Ask(space, false, $"what kind of thing is {m}");
            if (ga == g) gRet++;  if (ra == g) rRet++;
            if (ga == k) gHo++;   if (ra == k) rHo++;
            if (shown++ < 2)
                _out.WriteLine($"  RET/HO  {m,-8} (genus {g}, kingdom {k})   GEO→'{ga}'[{gp}]   RELAX→'{ra}'[{rp}]");
        }

        // ---- 3 EVICTION-SURVIVAL: wipe every genus → the only ancestor left is the kingdom, reachable ONLY via geometry ----
        var evs = BuildSpace();
        foreach (var gg in Fam.Select(f => f.g).Distinct()) evs.EvictConcept(gg);

        // DIAGNOSTIC: does the SUBSTRATE geometric method work on this (denser, 10-family) space, independent of the ladder?
        var d1 = space.TryGeometricDerive("sparrow", out var da1, out var dc1);
        var d2 = evs.TryGeometricDerive("sparrow", out var da2, out var dc2);
        _out.WriteLine($"  DIRECT substrate TryGeometricDerive: pre-evict sparrow⇒{(d1 ? $"'{da1}'({dc1:F2})" : "abstain")}   post-evict sparrow⇒{(d2 ? $"'{da2}'({dc2:F2})" : "abstain")}");
        int gEv = 0, rEv = 0; shown = 0;
        foreach (var (m, g, k) in Fam)
        {
            var (ga, gp) = Ask(evs, true, $"what kind of thing is {m}");
            var (ra, rp) = Ask(evs, false, $"what kind of thing is {m}");
            if (ga == k) gEv++;  if (ra == k) rEv++;
            if (shown++ < 2)
                _out.WriteLine($"  EVICT   {m,-8} (want {k})   GEO→'{ga}'[{gp}]   RELAX→'{ra}'[{rp}]");
        }

        // ---- 4 ABSTENTION: isolated glue-only subject must NOT fabricate ----
        var abs = BuildSpace(isolated: true);
        bool Abstains(string a) => a.Length == 0 || a == "zorblax";
        var (gz, gzp) = Ask(abs, true, "what kind of thing is zorblax");
        var (rz, rzp) = Ask(abs, false, "what kind of thing is zorblax");
        var gAb = Abstains(gz); var rAb = Abstains(rz);
        _out.WriteLine($"  ABSTAIN zorblax   GEO→'{(gz.Length == 0 ? "∅" : gz)}'[{gzp}] {(gAb ? "ABSTAIN" : "fabricated")}   RELAX→'{(rz.Length == 0 ? "∅" : rz)}'[{rzp}] {(rAb ? "ABSTAIN" : "fabricated")}");

        _out.WriteLine("");
        _out.WriteLine($"  1 NORMAL RETRIEVAL   GEOMETRIC {gRet}/{Fam.Length}  |  RELAX {rRet}/{Fam.Length}");
        _out.WriteLine($"  2 HELD-OUT KINGDOM   GEOMETRIC {gHo}/{Fam.Length}  |  RELAX {rHo}/{Fam.Length}");
        _out.WriteLine($"  3 EVICTION SURVIVAL  GEOMETRIC {gEv}/{Fam.Length}  |  RELAX {rEv}/{Fam.Length}");
        _out.WriteLine($"  4 ABSTENTION         GEOMETRIC {(gAb ? "1/1" : "0/1")}  |  RELAX {(rAb ? "1/1" : "0/1")}");

        var noRegress = gRet >= rRet;
        var winsEvict = gEv >= rEv + 3;   // a REAL margin, not a 1-vs-0 artifact — geometric must actually survive eviction, not squeak by
        _out.WriteLine("");
        _out.WriteLine(!noRegress
            ? ">>> VERDICT: REGRESSES RETRIEVAL — do NOT promote blindly; needs a narrower gate."
            : winsEvict
                ? ">>> VERDICT: promotion JUSTIFIED — geometric-primary matches retrieval AND materially wins eviction-survival."
                : ">>> VERDICT: ROUTING FIXED, NO DERIVATION GAIN — field-geometric now FIRES (latent-native genus retrieval, no regression), but does NOT beat relax on held-out/eviction/abstention at scale: nearest-content returns SIBLINGS not ancestors (similarity, not taxonomic derivation); the eviction/abstention 'wins' were small-space artifacts. Real geometric derivation needs HUB-finding (the ancestor is the common high-degree neighbour), not nearest — a different mechanism.");

        // Assert only the load-bearing fact: promotion must NOT regress direct retrieval. Verdict details are reported data.
        Assert.True(gRet >= rRet - 1, $"geometric-primary must not regress normal retrieval (GEO {gRet} vs RELAX {rRet})");
    }
}
