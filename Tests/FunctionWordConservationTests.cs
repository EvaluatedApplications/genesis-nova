using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// G6 CONSERVATION of the function-word distinction. The platonic WORLD only EXPANDS (G6: a distinction once made is
/// never unmade); only the SELF decays. Function-word-ness used to be RE-DERIVED from the live relation graph on every
/// call (neighbour-coherence ≤ an Otsu cut), so as the graph grew the measurement DRIFTED and the classification could
/// FLIP — even though the world only ADDED edges. That is a G6 violation ("the prebake gets worse over time"): a
/// conserved world distinction was being treated as a re-measured property.
///
/// The fix is GENERATE-BY-OBSERVATION → RETAIN: the live coherence is now only an EVIDENCE SOURCE; each warm reading
/// of a word bridging unrelated clusters ACCUMULATES a monotonic <see cref="Element.FunctionEvidence"/>, and once it
/// crosses the retention threshold the word is classified function-like FOREVER, regardless of later live drift. These
/// tests prove growth cannot unmake an established distinction, and that the conserved knowledge survives save/reload.
/// </summary>
public sealed class FunctionWordConservationTests
{
    private readonly ITestOutputHelper _out;
    public FunctionWordConservationTests(ITestOutputHelper o) => _out = o;

    // Eight fully-connected content clusters: a member co-occurs with its OWN kind, so its neighbours are mutually
    // connected → HIGH coherence = content. 8×8 = 64 content concepts (≥ FnMinWarm), each with degree ≥7.
    private static readonly string[][] Clusters =
    {
        new[]{"cat","dog","cow","pig","hen","fox","owl","bat"},
        new[]{"red","blue","green","pink","gray","black","white","brown"},
        new[]{"bob","sam","joe","amy","tom","kim","dan","liz"},
        new[]{"rome","paris","tokyo","cairo","lima","oslo","delhi","perth"},
        new[]{"iron","gold","tin","lead","zinc","copper","steel","brass"},
        new[]{"oak","elm","pine","ash","birch","maple","cedar","fir"},
        new[]{"rose","lily","tulip","daisy","ivy","fern","moss","reed"},
        new[]{"jazz","rock","blues","folk","punk","soul","metal","funk"},
    };
    private static readonly string[] Glue = { "the", "of", "is", "my" };

    // Build the content graph + bridge the glue words across the cluster ANCHORS (one per cluster, mutually unrelated),
    // then drive repeated amortized stats recomputes until the glue words have ACCRUED conserved evidence past retention.
    private static DialecticalSpace Establish(out string[] anchors)
    {
        var s = new DialecticalSpace(256, seed: 7);
        void Reinforce(string a, string b) { for (var i = 0; i < 6; i++) s.ObserveContradiction(a, b, 0.15); }
        foreach (var cl in Clusters)
            for (var i = 0; i < cl.Length; i++)
                for (var j = i + 1; j < cl.Length; j++)
                    Reinforce(cl[i], cl[j]);

        anchors = Clusters.Select(cl => cl[0]).ToArray(); // cat, red, bob, rome, iron, oak, rose, jazz — unrelated
        var anch = anchors;
        void Bridge() { foreach (var g in Glue) foreach (var a in anch) s.ObserveContradiction(g, a, 0.2); }
        for (var k = 0; k < 6; k++) Bridge();

        // ESTABLISH: each round bumps the active count past the amortization stamp (throwaway degree-1 pairs that only
        // advance the count), keeps the glue edges fresh, then calls the stats so EnsureFunctionStats re-runs and accrues.
        var filler = 0;
        for (var round = 0; round < 24 && Glue.Min(g => s.FunctionStats(g).Evidence) < 2.0; round++)
        {
            for (var f = 0; f < 20; f++) s.ObserveContradiction($"flr{filler}a", $"flr{filler++}b", 0.5);
            Bridge();
            foreach (var g in Glue) s.IsFunctionLike(g);
        }
        return s;
    }

    [Fact]
    public void GraphGrowth_CannotUnmake_AnEstablishedFunctionWord()
    {
        var s = Establish(out var anchors);

        var pre = s.FunctionStats("the");
        _out.WriteLine($"established: coh={pre.Centrality:F3} otsu={pre.Threshold:F3} evidence={pre.Evidence}");
        Assert.True(Glue.All(g => s.IsFunctionLike(g)), "glue words should be function-like after establishment");
        Assert.True(Glue.All(g => s.FunctionStats(g).Evidence >= 2.0), "conserved evidence did not accrue past retention");
        Assert.True(pre.Centrality <= pre.Threshold, "an established glue word should currently read BELOW the live cut");

        // A content word that NEVER bridged: high coherence, zero evidence, not function-like.
        Assert.False(s.IsFunctionLike("dog"), "a content word wrongly read function-like");
        Assert.Equal(0.0, s.FunctionStats("dog").Evidence);

        // GROW THE WORLD (G6 — only ADD edges): interconnect all 8 anchors so each glue word's neighbours become MUTUALLY
        // CONNECTED. Its LIVE neighbour-coherence then rises to ~1.0 (above the cut AND the diversity ceiling), so the
        // RE-MEASURED signal now reads it as CONTENT — exactly the drift that used to flip the classification at scale.
        for (var i = 0; i < anchors.Length; i++)
            for (var j = i + 1; j < anchors.Length; j++)
                for (var r = 0; r < 6; r++)
                    s.ObserveContradiction(anchors[i], anchors[j], 0.15);
        // force a fresh stats recompute (bump the count) so the live reading is current, not amortized-stale.
        for (var f = 0; f < 30; f++) s.ObserveContradiction($"post{f}a", $"post{f}b", 0.5);

        var post = s.FunctionStats("the");
        _out.WriteLine($"after growth: coh={post.Centrality:F3} otsu={post.Threshold:F3} floor={post.Floor:F2} " +
                       $"evidence={post.Evidence} fn?={s.IsFunctionLike("the")}");

        // The LIVE reading DRIFTED to non-function — coherence rose and now exceeds the diversity ceiling, so the live
        // test alone (coh ≤ Floor) would classify "the" as CONTENT...
        Assert.True(post.Centrality > pre.Centrality, $"growth should raise the live coherence ({pre.Centrality:F3}→{post.Centrality:F3})");
        Assert.True(post.Centrality > post.Floor, $"growth should drift live coherence above the diversity ceiling (coh={post.Centrality:F3} floor={post.Floor:F2})");

        // ...YET the conserved distinction is RETAINED. Growth could not unmake it. THE G6 PROOF.
        Assert.True(Glue.All(g => s.IsFunctionLike(g)), "G6 VIOLATION: graph growth UNMADE an established function-word distinction");
        // and the never-bridging content word is still not function-like.
        Assert.False(s.IsFunctionLike("dog"), "a content word wrongly read function-like after growth");
    }

    [Fact]
    public void FunctionEvidence_SurvivesSaveReload()
    {
        var s1 = Establish(out _);
        var before = Glue.ToDictionary(g => g, g => s1.FunctionStats(g).Evidence);
        Assert.True(before.Values.All(v => v >= 2.0), "precondition: evidence must be established before snapshot");

        var snap = s1.ExportSnapshot();
        // survive the on-disk JSON layer too (the sharded checkpoint serializes the snapshot with System.Text.Json).
        var json = System.Text.Json.JsonSerializer.Serialize(snap);
        var snapJson = System.Text.Json.JsonSerializer.Deserialize<GenesisNova.Cognition.PlatonicMemorySnapshot>(json)!;

        var s2 = new DialecticalSpace(256, seed: 7);
        s2.ImportSnapshot(snapJson);

        foreach (var g in Glue)
        {
            var ev = s2.FunctionStats(g).Evidence;
            _out.WriteLine($"  {g,-4} evidence: before={before[g]} after-reload={ev}");
            Assert.True(ev >= before[g], $"FunctionEvidence for '{g}' must survive reload (was {before[g]}, got {ev})");
            Assert.True(s2.IsFunctionLike(g), $"'{g}' must still read function-like after reload (conserved world-knowledge)");
        }
        Assert.Equal(0.0, s2.FunctionStats("dog").Evidence); // a content word carried no evidence across the reload
    }
}
