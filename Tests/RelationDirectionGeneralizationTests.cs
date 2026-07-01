using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// HELD-OUT-ENTITY GENERALIZATION — the test that decides whether the trained is-a direction is a real HOMOMORPHISM
/// (generalizes to entities it was NEVER trained on) or a MEMORIZED fit (only works on the trained entities). We observe
/// ALL members so every entity has a positioned cloud, then train the direction on SOME members of each genus and test
/// composition on the HELD-OUT members (present in the space, excluded from the training triples). If orbital(heldout)+r
/// reaches the genus / +2r the kingdom, the direction GENERALIZES — that is what earns the word "homomorphism". Reported
/// honestly (held-out is a measurement, not forced); the SEEN case is asserted so the setup is valid.
/// </summary>
public sealed class RelationDirectionGeneralizationTests
{
    private readonly ITestOutputHelper _out;
    public RelationDirectionGeneralizationTests(ITestOutputHelper o) => _out = o;

    // genus → (kingdom, [members]); the LAST member of each genus is HELD OUT of direction-training.
    private static readonly (string g, string k, string[] members)[] Fam =
    {
        ("bird","animal",     new[]{"sparrow","robin","finch"}),
        ("tree","plant",      new[]{"oak","pine","elm"}),
        ("fish","creature",   new[]{"trout","bass","perch"}),
        ("flower","bloom",    new[]{"rose","daisy","tulip"}),
        ("crystal","mineral", new[]{"quartz","granite","topaz"}),
    };

    private static double Cos(double[] a, double[] b)
    { double d=0,na=0,nb=0; for(var i=0;i<a.Length&&i<b.Length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];} return d/(Math.Sqrt(na)*Math.Sqrt(nb)+1e-12); }
    private static double[] Add(double[] a, double[] b, double s){var r=new double[a.Length];for(var i=0;i<a.Length;i++)r[i]=a[i]+s*b[i];return r;}

    [Fact]
    public void TrainedDirection_GeneralizesToHeldOutEntities_OrMemorizes()
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        // Observe ALL members (seen + held-out) so every entity has a positioned cloud.
        for (var c = 0; c < 40; c++)
            foreach (var (g, k, members) in Fam)
            {
                foreach (var m in members) { var s = new[] { "the", m, "is", "a", g }; ds.FineEditFromExample(s, s, false); }
                var s2 = new[] { "the", g, "is", "a", k }; ds.FineEditFromExample(s2, s2, false);
            }

        double[]? V(string s) => ds.OrbitalOf(s);
        var vocab = Fam.SelectMany(f => f.members.Concat(new[]{f.g,f.k})).Distinct().Where(s => V(s) != null).ToArray();
        string Nearest(double[] q, params string[] excl) => vocab.Where(c => !excl.Contains(c)).OrderByDescending(c => Cos(q, V(c)!)).First();
        // CATEGORY-restricted nearest — the meaningful is-a answer space is the CATEGORIES (genera + kingdoms), not sibling
        // members. A mapped member lands in its genus's REGION where the nearest *concept* is a sibling; the nearest
        // *category* is the genus. That's the real "which category did it reach" test.
        var cats = Fam.SelectMany(f => new[]{f.g,f.k}).Distinct().Where(s => V(s) != null).ToArray();
        string NearestCat(double[] q, params string[] excl) => cats.Where(c => !excl.Contains(c)).OrderByDescending(c => Cos(q, V(c)!)).First();

        // TRAIN on SEEN members only. TYPED per level (member→genus vs genus→kingdom are DIFFERENT relations — the level
        // mixing is what confounded a single "is-a" map). In prod the level/typing comes from the ∘is thread; here the
        // curriculum knows it. TransE baseline still uses one "is-a" (its design), for the honest A/B.
        var triples = new List<(string,string,string)>();       // for the TransE baseline (single relation)
        var mgTriples = new List<(string,string,string)>();     // member→genus (typed)
        var gkTriples = new List<(string,string,string)>();     // genus→kingdom (typed)
        var heldOut = new List<(string m, string g, string k)>();
        foreach (var (g, k, members) in Fam)
        {
            for (var i = 0; i < members.Length - 1; i++) { triples.Add((members[i], g, "is-a")); mgTriples.Add((members[i], g, "mem-genus")); }
            triples.Add((g, k, "is-a")); gkTriples.Add((g, k, "genus-king"));
            heldOut.Add((members[^1], g, k));   // last member of each genus HELD OUT
        }
        int seenTot = Fam.Sum(f => f.members.Length - 1);

        // ── AFTER: the ENTITY-AGNOSTIC MAP (kernel-ridge) — fit on RAW orbitals, moves nothing. Predict FIRST, while the
        //    orbitals are still raw (TransE below mutates them). ŷ = ApplyRelationMap; a held-out subject generalises via
        //    the shared PER-LEVEL map. This is the fix under test.
        ds.TrainRelationMap(mgTriples, lambda: 2.0);
        ds.TrainRelationMap(gkTriples, lambda: 2.0);
        double[]? Map1(string s) { var x = V(s); return x is null ? null : ds.ApplyRelationMap(x, "mem-genus"); }
        double[]? Map2(string s) { var y = Map1(s); return y is null ? null : ds.ApplyRelationMap(y, "genus-king"); }

        int mSeen = 0;
        foreach (var (g, k, members) in Fam)
            for (var i = 0; i < members.Length - 1; i++) if (NearestCat(Map1(members[i])!) == g) mSeen++;
        int mHoGenus = 0, mHoKing = 0;
        foreach (var (m, g, k) in heldOut)
        {
            if (NearestCat(Map1(m)!) == g) mHoGenus++;         // 1-hop → which CATEGORY (genus), on an UNSEEN entity
            if (NearestCat(Map2(m)!, g) == k) mHoKing++;       // 2-hop composition → kingdom, on an UNSEEN entity
        }

        // ── BEFORE: TransE baseline (memorises). Run LAST — it mutates orbitals. Held-out stays low = the problem we fixed.
        ds.TrainRelationDirection(triples, epochs: 800, lr: 0.05, margin: 2.0, maxNorm: 4.0, seed: 7);
        var r = ds.RelationVector("is-a")!;
        int tHoGenus = 0;
        foreach (var (m, g, k) in heldOut) if (NearestCat(Add(V(m)!, r, 1)) == g) tHoGenus++;   // same metric, fair A/B

        _out.WriteLine($"BEFORE (TransE, memorises):  HELD-OUT m+r→genus {tHoGenus}/{heldOut.Count}");
        _out.WriteLine($"AFTER  (entity-agnostic map): SEEN {mSeen}/{seenTot};  HELD-OUT genus {mHoGenus}/{heldOut.Count};  kingdom(2-hop) {mHoKing}/{heldOut.Count}");
        _out.WriteLine(mHoGenus >= heldOut.Count - 1
            ? ">>> GENERALIZES — the shared map reaches the genus on entities NEVER trained on. Structure, not memorization: 'homomorphism' earned via a learned map."
            : $">>> map did NOT generalize cleanly ({mHoGenus}/{heldOut.Count}) — a LINEAR shared map is not enough; needs a nonlinear/NN encoder. Honest finding.");

        Assert.True(heldOut.Count >= 5, "non-trivial held-out set");
        Assert.True(mSeen >= seenTot - 1, $"the map must fit the seen members (setup valid), got {mSeen}/{seenTot}");
        // THE DELIVERABLE: the entity-agnostic map GENERALIZES to unseen entities where TransE memorizes.
        Assert.True(mHoGenus > tHoGenus, $"the map must beat TransE on unseen entities (generalization), map {mHoGenus} vs transE {tHoGenus}");
        Assert.True(mHoGenus >= heldOut.Count - 1, $"the map should reach the genus on nearly all unseen entities, got {mHoGenus}/{heldOut.Count}");
    }
}
