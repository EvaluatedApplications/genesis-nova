using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EXACT save/reload fidelity for the conscious-field substrate. A long training investment is worthless if a reload
/// silently loads a different brain, so the DialecticalSpace snapshot must round-trip EXACTLY: same concept set, same
/// learned orbital per concept (to the bit), same relations, same nearest-neighbour behaviour — INCLUDING atoms,
/// which the old node-snapshot dropped. This is the guard that the model you saved is the model you reload.
/// </summary>
public sealed class DialecticalSpaceFidelityTests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceFidelityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void ExportImport_RoundTripsExactly()
    {
        const int dim = 256;
        var s1 = new DialecticalSpace(dim, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 4; i++) s1.FineEditFromExample(new[] { a }, new[] { b }, false); }
        // A mix exercising every element kind: single-word associations (char atoms), multi-word reply CHUNKS (word
        // compositions), category/synonym facts, and word↔number equivalence.
        Rel("big", "large"); Rel("big", "huge"); Rel("apple", "fruit"); Rel("dog", "animal"); Rel("car", "vehicle");
        Rel("hello", "what the fuck do you want now"); Rel("bye", "finally fuck off"); Rel("thanks", "whatever now");
        Rel("help", "figure it out yourself"); Rel("one", "1"); Rel("two", "2");

        var snap = s1.ExportSnapshot();
        var s2 = new DialecticalSpace(dim, seed: 7);
        s2.ImportSnapshot(snap);

        // 1) IDENTICAL learned (non-numeric) concept set, atoms included. Numeric concepts carry NO learned state
        // (zero orbital, codec-exact identity) and are recreated on demand, so they are intentionally not exported.
        static bool IsNum(string s) => double.TryParse(s, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out _);
        var c1 = s1.ActiveConcepts.Where(x => !IsNum(x)).OrderBy(x => x, StringComparer.Ordinal).ToArray();
        var c2 = s2.ActiveConcepts.Where(x => !IsNum(x)).OrderBy(x => x, StringComparer.Ordinal).ToArray();
        _out.WriteLine($"learned concepts s1={c1.Length} s2={c2.Length}, exported elements={snap.Elements!.Length}");
        Assert.Equal(c1, c2);
        Assert.Contains(snap.Elements!, e => e.Symbol.StartsWith("atom:", StringComparison.Ordinal)); // atoms ARE exported (old node path dropped them)

        // 2) IDENTICAL learned orbital per concept — to the bit.
        var maxDelta = 0.0;
        foreach (var c in c1)
        {
            var v1 = s1.SemanticVectorOf(c); var v2 = s2.SemanticVectorOf(c);
            Assert.NotNull(v1); Assert.NotNull(v2);
            Assert.Equal(v1!.Length, v2!.Length);
            for (var i = 0; i < v1.Length; i++) maxDelta = Math.Max(maxDelta, Math.Abs(v1[i] - v2[i]));
        }
        _out.WriteLine($"max orbital delta = {maxDelta:E3}");
        Assert.True(maxDelta < 1e-9, $"orbitals must round-trip exactly; max delta {maxDelta:E3}");

        // 3) IDENTICAL relations + element counts on re-export.
        var snap2 = s2.ExportSnapshot();
        Assert.Equal(snap.Relations.Length, snap2.Relations.Length);
        Assert.Equal(snap.Elements!.Length, snap2.Elements!.Length);

        // 3b) Survives the CHECKPOINT JSON layer (the sharded store serializes the snapshot with System.Text.Json) —
        // orbitals still bit-exact after a JSON round-trip, so the on-disk checkpoint reloads the same brain.
        var json = System.Text.Json.JsonSerializer.Serialize(snap);
        var snapJson = System.Text.Json.JsonSerializer.Deserialize<GenesisNova.Cognition.PlatonicMemorySnapshot>(json)!;
        Assert.Equal(snap.Elements!.Length, snapJson.Elements!.Length);
        var s3 = new DialecticalSpace(dim, seed: 7);
        s3.ImportSnapshot(snapJson);
        var maxJsonDelta = 0.0;
        foreach (var c in c1)
        {
            var v1 = s1.SemanticVectorOf(c); var v3 = s3.SemanticVectorOf(c);
            for (var i = 0; i < v1!.Length; i++) maxJsonDelta = Math.Max(maxJsonDelta, Math.Abs(v1[i] - v3![i]));
        }
        _out.WriteLine($"max orbital delta after JSON round-trip = {maxJsonDelta:E3}");
        Assert.True(maxJsonDelta < 1e-9, $"orbitals must survive JSON serialization exactly; max delta {maxJsonDelta:E3}");

        // 4) IDENTICAL nearest-neighbour behaviour for the trained cues.
        // Compare the meaningful (non-numeric) neighbours: numeric concepts have zero orbital (noise-ranked) and are
        // recreated on demand, so whether one happens to be instantiated differs harmlessly. The LEARNED ordering matches.
        foreach (var cue in new[] { "big", "apple", "hello", "bye", "help" })
        {
            var n1 = s1.GetNearestConcepts(cue, null, 8).Select(n => n.Symbol).Where(x => !IsNum(x)).Take(5).ToArray();
            var n2 = s2.GetNearestConcepts(cue, null, 8).Select(n => n.Symbol).Where(x => !IsNum(x)).Take(5).ToArray();
            if (!n1.SequenceEqual(n2)) _out.WriteLine($"  {cue}: s1=[{string.Join(",", n1)}]  s2=[{string.Join(",", n2)}]");
            Assert.Equal(n1, n2);
        }
    }
}
