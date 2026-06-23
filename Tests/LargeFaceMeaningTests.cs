using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EXPERIMENT (user hypothesis): the large semantic face is "the best place to store complex AMBIGUOUS meaning, as
/// it stores a whole phrase of words." Test whether representing a concept's large-face meaning as a SUPERPOSITION
/// of its relational context (the words it associates with — distributional / Saussure, the genesis electron-orbital
/// cloud) (a) clusters related concepts far better than a single settled point, and (b) holds AMBIGUITY — a word
/// with two senses sits near BOTH sense-regions at once. Pure math, no core change yet — this decides whether to
/// rebuild the large-face representation.
/// </summary>
public sealed class LargeFaceMeaningTests
{
    private readonly ITestOutputHelper _out;
    public LargeFaceMeaningTests(ITestOutputHelper o) => _out = o;

    private const int Dim = 310; // the large face width at production (512 − 202)

    // A stable per-word "token vector" (the basis); a concept's MEANING is the superposition of its context's tokens.
    private static double[] Token(string w)
    {
        var rng = new Random(StableHash(w));
        var v = new double[Dim];
        var s = 0.0;
        for (var i = 0; i < Dim; i++) { v[i] = rng.NextDouble() * 2 - 1; s += v[i] * v[i]; }
        var inv = 1.0 / Math.Sqrt(s);
        for (var i = 0; i < Dim; i++) v[i] *= inv;
        return v;
    }
    private static int StableHash(string s) { var h = 2166136261u; foreach (var c in s) { h ^= c; h *= 16777619u; } return unchecked((int)h); }

    // The large-face meaning vector = normalized superposition of the concept's context words (a phrase of words).
    private static double[] Meaning(IEnumerable<string> context)
    {
        var m = new double[Dim];
        foreach (var w in context) { var t = Token(w); for (var i = 0; i < Dim; i++) m[i] += t[i]; }
        var s = Math.Sqrt(m.Sum(x => x * x));
        if (s > 1e-9) for (var i = 0; i < Dim; i++) m[i] /= s;
        return m;
    }
    private static double Cos(double[] a, double[] b) { var d = 0.0; for (var i = 0; i < Dim; i++) d += a[i] * b[i]; return d; }

    [Fact]
    public void LargeFace_AsSuperposition_ClustersRelated_AndHoldsAmbiguity()
    {
        // Each concept = a "phrase of words" (its relational context) stored in the large face.
        var ctx = new Dictionary<string, string[]>
        {
            ["cat"] = new[] { "animal", "pet", "fur", "purr" },
            ["dog"] = new[] { "animal", "pet", "fur", "bark" },
            ["lion"] = new[] { "animal", "wild", "fur", "roar" },
            ["car"] = new[] { "vehicle", "road", "engine", "drive" },
            ["truck"] = new[] { "vehicle", "road", "engine", "haul" },
            // pure senses for the ambiguity test:
            ["stream"] = new[] { "river", "water", "fish", "flow" },     // river sense
            ["fund"] = new[] { "money", "loan", "cash", "account" },     // money sense
            // the AMBIGUOUS word — its phrase spans BOTH senses at once:
            ["bank"] = new[] { "river", "water", "fish", "money", "loan", "cash" },
        };
        var m = ctx.ToDictionary(kv => kv.Key, kv => Meaning(kv.Value));

        // (1) related (share context) cluster; unrelated don't.
        var related = new[] { ("cat", "dog"), ("cat", "lion"), ("dog", "lion"), ("car", "truck") }.Select(p => Cos(m[p.Item1], m[p.Item2])).ToArray();
        var unrelated = new[] { ("cat", "car"), ("dog", "truck"), ("lion", "car"), ("cat", "truck") }.Select(p => Cos(m[p.Item1], m[p.Item2])).ToArray();
        _out.WriteLine($"related cos {related.Average():F3}   unrelated cos {unrelated.Average():F3}   separation {related.Average() - unrelated.Average():F3}");

        // (2) AMBIGUITY: bank sits near BOTH senses; the two senses are far from each other.
        var bankRiver = Cos(m["bank"], m["stream"]);
        var bankMoney = Cos(m["bank"], m["fund"]);
        var riverMoney = Cos(m["stream"], m["fund"]);
        _out.WriteLine($"bank↔river {bankRiver:F3}   bank↔money {bankMoney:F3}   river↔money {riverMoney:F3}");

        Assert.True(related.Average() - unrelated.Average() > 0.3, "distributional meaning must cluster related concepts strongly");
        Assert.True(bankRiver > riverMoney + 0.2 && bankMoney > riverMoney + 0.2,
            "the ambiguous word must sit near BOTH senses (superposition), while the senses stay far from each other");
    }

    // A concept's meaning = its OWN token + the superposition of its direct relations' tokens. The self-token is the
    // fix for the gym's SPARSE DIRECT relations (big↔large, 5↔five): without it two directly-related concepts hold
    // each other's (orthogonal) token and look dissimilar; with it they share both tokens and cluster.
    private static double[] MeaningOf(string c, Dictionary<string, HashSet<string>> rel)
    {
        var m = (double[])Token(c).Clone();
        if (rel.TryGetValue(c, out var ns))
            foreach (var n in ns) { var t = Token(n); for (var i = 0; i < Dim; i++) m[i] += t[i]; }
        var s = Math.Sqrt(m.Sum(x => x * x));
        if (s > 1e-9) for (var i = 0; i < Dim; i++) m[i] /= s;
        return m;
    }

    [Fact] // The gym-realistic SPARSE case: direct synonym/digit-word/category edges must still cluster (self-token).
    public void LargeFace_Distributional_HandlesSparseDirectRelations()
    {
        var rel = new Dictionary<string, HashSet<string>>();
        void Edge(string a, string b)
        {
            (rel.TryGetValue(a, out var sa) ? sa : rel[a] = new()).Add(b);
            (rel.TryGetValue(b, out var sb) ? sb : rel[b] = new()).Add(a);
        }
        Edge("big", "large"); Edge("big", "huge"); Edge("big", "giant");      // synonym hub
        Edge("small", "tiny"); Edge("small", "little");
        Edge("5", "five"); Edge("6", "six");                                   // digit↔word (sparse direct)
        Edge("apple", "fruit"); Edge("banana", "fruit");                       // co-category via shared hub

        double Sim(string a, string b) => Cos(MeaningOf(a, rel), MeaningOf(b, rel));
        _out.WriteLine($"big~large {Sim("big", "large"):F3}  5~five {Sim("5", "five"):F3}  apple~banana {Sim("apple", "banana"):F3}");
        _out.WriteLine($"big~5 {Sim("big", "5"):F3}  large~tiny {Sim("large", "tiny"):F3}  apple~big {Sim("apple", "big"):F3}");

        Assert.True(Sim("big", "large") > 0.3, "direct synonym pair clusters (self-token)");
        Assert.True(Sim("5", "five") > 0.3, "digit↔word clusters");
        Assert.True(Sim("apple", "banana") > 0.3, "co-category (shared 'fruit') clusters");
        Assert.True(Sim("big", "large") > Sim("big", "5") + 0.2, "related clearly beats unrelated");
    }
}
