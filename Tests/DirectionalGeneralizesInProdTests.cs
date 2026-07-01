using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// DirectionalReasoning now composes the entity-agnostic MAP (ApplyRelationMap), not the memorized TransE vector, and picks
/// the nearest content-HUB (ancestor) rather than the nearest concept (sibling) — so the WIRED `TryDirectionalDerive`
/// reaches the genus on entities NEVER trained on. This proves the generalization holds through the live derivation method
/// (not just the raw map math). Held-out members are embedded but excluded from map training; the map + hub-pick must still
/// reach their genus. (2-hop kingdom needs cross-level composition — future; and the live observe-loop signal is a mixed
/// "assoc" label until ∘is typing, which degrades 1-hop — both noted, not asserted here.)
/// </summary>
public sealed class DirectionalGeneralizesInProdTests
{
    private readonly ITestOutputHelper _out;
    public DirectionalGeneralizesInProdTests(ITestOutputHelper o) => _out = o;

    private static readonly (string g, string k, string[] members)[] Fam =
    {
        ("bird","animal",     new[]{"sparrow","robin","finch"}),
        ("tree","plant",      new[]{"oak","pine","elm"}),
        ("fish","creature",   new[]{"trout","bass","perch"}),
        ("flower","bloom",    new[]{"rose","daisy","tulip"}),
        ("crystal","mineral", new[]{"quartz","granite","topaz"}),
    };

    [Fact]
    public void DirectionalDerive_ViaMap_ReachesGenus_OnUnseenEntities()
    {
        var ds = new DialecticalSpace(ProductionDims.FaceDimension, seed: 5) { SelfDiscriminatedIngestion = true };
        // Embed ALL members (seen + held-out) so every entity has a positioned cloud (same setup as the proven map test).
        for (var c = 0; c < 40; c++)
            foreach (var (g, k, members) in Fam)
            {
                foreach (var m in members) { var s = new[] { "the", m, "is", "a", g }; ds.FineEditFromExample(s, s, false); }
                var s2 = new[] { "the", g, "is", "a", k }; ds.FineEditFromExample(s2, s2, false);
            }
        ds.FlushCloudBatch();

        // Train the PER-LEVEL map on SEEN members only (last member of each genus held out). λ=2.0 generalizes.
        var mg = new List<(string, string, string)>();
        var heldOut = new List<(string m, string g)>();
        foreach (var (g, k, members) in Fam)
        {
            for (var i = 0; i < members.Length - 1; i++) mg.Add((members[i], g, "mg"));
            mg.Add((g, k, "gk"));
            heldOut.Add((members[^1], g));
        }
        ds.TrainRelationMap(mg, lambda: 2.0);

        // The WIRED live method: TryDirectionalDerive composes ApplyRelationMap, then picks the nearest content-HUB.
        int unseen = 0, seen = 0;
        foreach (var (m, g) in heldOut)
        {
            var ok = ds.TryDirectionalDerive(m, out var a, out _, hops: 1, label: "mg") && a == g;
            if (ok) unseen++;
            _out.WriteLine($"  UNSEEN {m,-8} → '{a}' (want {g}) {(ok ? "HIT" : "miss")}");
        }
        foreach (var (g, _, members) in Fam)
            if (ds.TryDirectionalDerive(members[0], out var a, out _, hops: 1, label: "mg") && a == g) seen++;

        _out.WriteLine($"\nTryDirectionalDerive via MAP:  SEEN {seen}/{Fam.Length}   UNSEEN {unseen}/{heldOut.Count}");
        _out.WriteLine(unseen >= heldOut.Count - 1
            ? ">>> DirectionalReasoning GENERALIZES to unseen entities through the wired derive path (map + hub-pick), where memorized TransE got 0."
            : ">>> did not generalize through the derive path — inspect the hub-pick / margin.");

        Assert.True(seen >= Fam.Length - 1, $"the wired method must resolve SEEN entities (setup valid), got {seen}/{Fam.Length}");
        Assert.True(unseen >= heldOut.Count - 1, $"the wired directional derive must GENERALIZE the genus to unseen entities, got {unseen}/{heldOut.Count}");
    }
}
