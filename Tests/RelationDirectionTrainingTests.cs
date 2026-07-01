using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE WRITE-SIDE FOLD. `RelationDirectionTests` proved is-a is NOT a latent direction in the raw distributional cloud
/// (cross-family offset cosine ~0.01, 2-hop composition fails) — arithmetic reasons geometrically only because the frozen
/// codec IMPOSES the additive direction on numbers. This trains that structure IN for relations (TransE): after
/// `TrainRelationDirection`, is-a becomes a CONSISTENT direction and orbital(m)+k·r reaches the k-hop ancestor — the
/// fold-faithful win the raw clouds can't give. Also measures the honest TENSION: does imposing the direction wreck the
/// distributional similarity retrieval relies on?
/// </summary>
public sealed class RelationDirectionTrainingTests
{
    private readonly ITestOutputHelper _out;
    public RelationDirectionTrainingTests(ITestOutputHelper o) => _out = o;

    private static readonly (string m, string g, string k)[] Fam =
    {
        ("sparrow","bird","animal"),   ("robin","bird","animal"),
        ("oak","tree","plant"),        ("pine","tree","plant"),
        ("trout","fish","creature"),   ("bass","fish","creature"),
        ("rose","flower","bloom"),     ("daisy","flower","bloom"),
        ("quartz","crystal","mineral"),("granite","crystal","mineral"),
    };

    private static double Cos(double[] a, double[] b)
    { double d = 0, na = 0, nb = 0; for (var i = 0; i < a.Length && i < b.Length; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; } return d / (Math.Sqrt(na)*Math.Sqrt(nb) + 1e-12); }
    private static double[] Add(double[] a, double[] b, double s) { var r = new double[a.Length]; for (var i=0;i<a.Length;i++) r[i]=a[i]+s*b[i]; return r; }
    private static double[] Sub(double[] a, double[] b) { var r = new double[a.Length]; for (var i=0;i<a.Length;i++) r[i]=a[i]-b[i]; return r; }

    [Fact]
    public void TrainedIsA_BecomesConsistentDirection_And2HopComposes()
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        for (var c = 0; c < 40; c++)
            foreach (var (m, g, k) in Fam)
            {
                var s1 = new[] { "the", m, "is", "a", g }; var s2 = new[] { "the", g, "is", "a", k };
                ds.FineEditFromExample(s1, s1, false); ds.FineEditFromExample(s2, s2, false);
            }

        double[]? V(string s) => ds.OrbitalOf(s);   // RAW orbital — composition stays in the trained (non-unit) space
        var vocab = Fam.SelectMany(f => new[] { f.m, f.g, f.k }).Distinct().Where(s => V(s) != null).ToArray();

        double CrossCos()
        {
            var offs = Fam.Select(f => Sub(V(f.g)!, V(f.m)!)).ToList();
            var xs = new List<double>();
            for (var i = 0; i < Fam.Length; i++) for (var j = i + 1; j < Fam.Length; j++)
                if (Fam[i].g != Fam[j].g) xs.Add(Cos(offs[i], offs[j]));   // cross-family
            return xs.Average();
        }
        string Nearest(double[] q, params string[] excl) => vocab.Where(c => !excl.Contains(c)).OrderByDescending(c => Cos(q, V(c)!)).First();

        // ── BEFORE: raw distributional cloud (the negative baseline).
        var beforeCos = CrossCos();
        int before2hop = 0;
        foreach (var (m, g, k) in Fam)
        {
            var mu = Fam.Where(f => f.g != g).Select(f => Sub(V(f.g)!, V(f.m)!)).Aggregate(new double[V(m)!.Length], (a, o) => Add(a, o, 1));
            var n = Math.Max(1, Fam.Count(f => f.g != g)); for (var i = 0; i < mu.Length; i++) mu[i] /= n;
            if (Nearest(Add(V(m)!, mu, 2), m) == k) before2hop++;
        }
        _out.WriteLine($"BEFORE (raw cloud): is-a cross-family cosine {beforeCos:F3};  m+2µ→kingdom {before2hop}/{Fam.Length}");

        // ── TRAIN a single is-a direction over BOTH hops (member→genus AND genus→kingdom).
        var triples = new List<(string, string, string)>();
        foreach (var (m, g, k) in Fam) { triples.Add((m, g, "is-a")); triples.Add((g, k, "is-a")); }
        ds.TrainRelationDirection(triples, epochs: 800, lr: 0.05, margin: 2.0, maxNorm: 4.0, seed: 7);
        var r = ds.RelationVector("is-a")!;

        // ── AFTER: the trained direction.
        var afterCos = CrossCos();
        int genusHit = 0, kingHit = 0;
        foreach (var (m, g, k) in Fam)
        {
            if (Nearest(Add(V(m)!, r, 1), m) == g) genusHit++;               // 1 hop → genus
            if (Nearest(Add(V(m)!, r, 2), m) == k) kingHit++;               // 2 hops → kingdom (composition)
        }
        // ── TENSION: after training, is a member's nearest still its SIBLING (distributional similarity preserved)?
        int sibNearest = 0;
        foreach (var (m, g, k) in Fam)
        {
            var sib = Fam.First(f => f.g == g && f.m != m).m;
            if (Nearest(V(m)!, m) == sib) sibNearest++;
        }

        _out.WriteLine($"AFTER  (trained)  : is-a cross-family cosine {afterCos:F3};  m+r→genus {genusHit}/{Fam.Length};  m+2r→kingdom {kingHit}/{Fam.Length}");
        _out.WriteLine($"TENSION: member's nearest is its sibling (similarity preserved) {sibNearest}/{Fam.Length}");
        _out.WriteLine(kingHit >= Fam.Length - 2 && afterCos > 0.5
            ? ">>> WRITE-SIDE FOLD IMPOSED: trained is-a is a consistent direction; 2-hop composition reaches the kingdom (fold-faithful). The clouds couldn't; a trained mapping can — exactly the codec's trick, for relations."
            : ">>> partial — report the numbers honestly.");

        // The negative baseline must hold (raw clouds carry no direction), and training must IMPOSE it:
        Assert.True(beforeCos < 0.2, $"raw is-a offsets are NOT a direction (baseline), got {beforeCos:F3}");
        Assert.True(afterCos > 0.5, $"training must make is-a a consistent direction, got {afterCos:F3}");
        Assert.True(genusHit >= Fam.Length - 1, $"m+r must reach the genus, got {genusHit}/{Fam.Length}");
        Assert.True(kingHit >= Fam.Length - 2, $"THE HEADLINE: m+2r must compose to the kingdom, got {kingHit}/{Fam.Length}");
    }
}
