using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// FOLD-FAITHFUL vs OVERFITTING: is `is-a` a CONSISTENT DIRECTION (offset) in the latent space? If member→genus is a
/// repeatable geometric offset (like the arithmetic homomorphism embed(a)+embed(b)=embed(a+b)), reasoning is directional
/// COMPOSITION — ancestor vs sibling separates for free (ancestor lies ALONG is-a; sibling is orthogonal), and no
/// nearest/hub heuristic is needed. If offsets are idiosyncratic (cos≈0) or member+µ still lands on siblings, the current
/// geometry does NOT carry clean relation-directions — the honest limit. Reported numbers, not a forced pass.
/// </summary>
public sealed class RelationDirectionTests
{
    private readonly ITestOutputHelper _out;
    public RelationDirectionTests(ITestOutputHelper o) => _out = o;

    private static readonly (string m, string g, string k)[] Fam =
    {
        ("sparrow","bird","animal"),   ("robin","bird","animal"),
        ("oak","tree","plant"),        ("pine","tree","plant"),
        ("trout","fish","creature"),   ("bass","fish","creature"),
        ("rose","flower","bloom"),     ("daisy","flower","bloom"),
        ("quartz","crystal","mineral"),("granite","crystal","mineral"),
    };

    private static double Cos(double[] a, double[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length && i < b.Length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        return d / (Math.Sqrt(na)*Math.Sqrt(nb) + 1e-12);
    }
    private static double[] Add(double[] a, double[] b, double s) { var r = new double[a.Length]; for (var i=0;i<a.Length;i++) r[i]=a[i]+s*b[i]; return r; }
    private static double[] Sub(double[] a, double[] b) { var r = new double[a.Length]; for (var i=0;i<a.Length;i++) r[i]=a[i]-b[i]; return r; }

    [Fact]
    public void IsA_IsItAConsistentLatentDirection()
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        for (var c = 0; c < 40; c++)
            foreach (var (m, g, k) in Fam)
            {
                var s1 = new[] { "the", m, "is", "a", g };
                var s2 = new[] { "the", g, "is", "a", k };
                ds.FineEditFromExample(s1, s1, false);
                ds.FineEditFromExample(s2, s2, false);
            }

        double[]? V(string s) => ds.SemanticVectorOf(s);
        var vocab = Fam.SelectMany(f => new[] { f.m, f.g, f.k }).Distinct().Where(s => V(s) != null).ToArray();

        // (1) is-a offset consistency: cosine between member→genus offsets from DIFFERENT families.
        var offs = Fam.Select(f => (f, off: Sub(V(f.g)!, V(f.m)!))).ToList();
        var cross = new List<double>();
        for (var i = 0; i < offs.Count; i++)
            for (var j = i + 1; j < offs.Count; j++)
                if (offs[i].f.g != offs[j].f.g)   // different genus = cross-family
                    cross.Add(Cos(offs[i].off, offs[j].off));
        var meanCross = cross.Average();
        _out.WriteLine($"(1) is-a offset cross-family cosine: mean={meanCross:F3}  (>0.5 = shared direction; ~0 = idiosyncratic)");

        // (2) directional derivation vs SIBLING: held-out mean offset µ (exclude the test family's genus), does m+µ land
        //     nearest the GENUS and beat the SIBLING?
        string Nearest(double[] q, string exclude) => vocab.Where(c => c != exclude)
            .OrderByDescending(c => Cos(q, V(c)!)).First();
        int genusWins = 0, genusBeatsSib = 0, tested = 0;
        foreach (var (m, g, k) in Fam)
        {
            var mu = offs.Where(o => o.f.g != g).Select(o => o.off).Aggregate(new double[V(m)!.Length], (a,o)=>Add(a,o,1));
            for (var i = 0; i < mu.Length; i++) mu[i] /= Math.Max(1, offs.Count(o => o.f.g != g));
            var q = Add(V(m)!, mu, 1);
            var near = Nearest(q, m);
            var sib = Fam.First(f => f.g == g && f.m != m).m;   // a same-genus sibling
            var cg = Cos(q, V(g)!); var cs = Cos(q, V(sib)!);
            if (near == g) genusWins++;
            if (cg > cs) genusBeatsSib++;
            tested++;
            if (tested <= 4) _out.WriteLine($"    {m,-8}+µ → nearest '{near}'  (want genus {g}); cos(genus)={cg:F3} vs cos(sib {sib})={cs:F3}  {(cg>cs?"genus>sib":"SIB WINS")}");
        }
        _out.WriteLine($"(2) m+µ: nearest==genus {genusWins}/{tested};  genus beats sibling {genusBeatsSib}/{tested}");

        // (3) 2-hop composition: m+2µ near the KINGDOM?
        int kingWins = 0;
        foreach (var (m, g, k) in Fam.Take(4))
        {
            var mu = offs.Where(o => o.f.g != g).Select(o => o.off).Aggregate(new double[V(m)!.Length], (a,o)=>Add(a,o,1));
            for (var i = 0; i < mu.Length; i++) mu[i] /= Math.Max(1, offs.Count(o => o.f.g != g));
            var q2 = Add(V(m)!, mu, 2);
            var near2 = Nearest(q2, m);
            if (near2 == k) kingWins++;
            _out.WriteLine($"    {m,-8}+2µ → nearest '{near2}'  (want kingdom {k})");
        }

        _out.WriteLine("");
        var real = meanCross > 0.5 && genusBeatsSib >= tested - 1;
        _out.WriteLine(real
            ? ">>> FOLD-FAITHFUL IS REAL: is-a is a consistent latent direction; reasoning can be geometric composition, ancestor separates from sibling by direction. Replace nearest/hub heuristics with direction-composition."
            : $">>> NOT AVAILABLE in current geometry: is-a offsets {(meanCross>0.5?"consistent":"NOT consistent (cos "+meanCross.ToString("F2")+")")}, m+µ beats sibling {genusBeatsSib}/{tested}. The latent space (distributional accumulation) does NOT carry clean relation-directions — fold-faithful directional reasoning needs a TRAINED mapping, not the raw clouds. Honest limit, not a cheap win.");

        Assert.True(vocab.Length >= 12, "taxonomy vocab built");
    }
}
