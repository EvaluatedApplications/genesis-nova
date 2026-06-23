using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE ONE CLAIM under PLATONIC_MIND.md: **predictive relaxation is reasoning.** Concepts are distributional clouds
/// (the same token-superposition the substrate uses); the "field" relaxes a query toward the lowest-surprise basin
/// via a modern-Hopfield / attention update (x ← normalize(Σ softmax(β⟨x,μ⟩)·μ) — relaxation = thought). We test the
/// three falsifiable predictions: (1) RECALL — a noisy state completes to its attractor; (2) DISAMBIGUATION — the
/// SAME ambiguous word settles into DIFFERENT basins under context; (3) ABSTENTION — an unknown query stays
/// high-surprise (no near basin) instead of inventing an answer. Pure math, no NN — this earns (or kills) the
/// dynamics-face / self-model direction cheaply.
/// </summary>
public sealed class FieldRelaxationTests
{
    private readonly ITestOutputHelper _out;
    public FieldRelaxationTests(ITestOutputHelper o) => _out = o;

    private const int Dim = 310; // the large face

    private static int Hash(string s) { var h = 2166136261u; foreach (var c in s) { h ^= c; h *= 16777619u; } return unchecked((int)h); }
    private static double[] Token(string w) { var r = new Random(Hash(w)); var v = new double[Dim]; var s = 0.0; for (var i = 0; i < Dim; i++) { v[i] = r.NextDouble() * 2 - 1; s += v[i] * v[i]; } return Norm(v); }
    private static double[] Cloud(IEnumerable<string> ctx) { var m = new double[Dim]; foreach (var w in ctx) { var t = Token(w); for (var i = 0; i < Dim; i++) m[i] += t[i]; } return Norm(m); }
    private static double[] Norm(double[] v) { var s = Math.Sqrt(v.Sum(x => x * x)); if (s > 1e-12) for (var i = 0; i < v.Length; i++) v[i] /= s; return v; }
    private static double Dot(double[] a, double[] b) { var d = 0.0; for (var i = 0; i < Dim; i++) d += a[i] * b[i]; return d; }

    // Relaxation: x ← normalize(Σ softmax(β⟨x,μ⟩)·μ), iterated. Returns the settled basin, the INITIAL surprise
    // (1 − max sim of the raw query = free energy before collapse), and the final tightness.
    private static (int Best, double InitMaxSim, double FinalMaxSim) Relax(double[] query, IReadOnlyList<double[]> attractors, double beta = 12, int iters = 8)
    {
        var x = Norm((double[])query.Clone());
        var initMax = attractors.Max(a => Dot(x, a));
        for (var t = 0; t < iters; t++)
        {
            var sims = attractors.Select(a => Dot(x, a)).ToArray();
            var mx = sims.Max();
            var w = sims.Select(s => Math.Exp(beta * (s - mx))).ToArray();
            var z = w.Sum();
            var nx = new double[Dim];
            for (var i = 0; i < attractors.Count; i++) { var wi = w[i] / z; for (var d = 0; d < Dim; d++) nx[d] += wi * attractors[i][d]; }
            x = Norm(nx);
        }
        var fsims = attractors.Select(a => Dot(x, a)).ToArray();
        var fmax = fsims.Max();
        return (Array.IndexOf(fsims, fmax), initMax, fmax);
    }

    [Fact]
    public void PredictiveRelaxation_IsReasoning()
    {
        var names = new[] { "cat", "dog", "lion", "car", "stream", "fund" };
        var ctx = new Dictionary<string, string[]>
        {
            ["cat"] = new[] { "animal", "pet", "fur", "purr" },
            ["dog"] = new[] { "animal", "pet", "fur", "bark" },
            ["lion"] = new[] { "animal", "wild", "fur", "roar" },
            ["car"] = new[] { "vehicle", "road", "engine", "drive" },
            ["stream"] = new[] { "river", "water", "fish", "flow" }, // the river-sense basin
            ["fund"] = new[] { "money", "loan", "cash", "account" }, // the money-sense basin
        };
        var attractors = names.Select(n => Cloud(ctx[n])).ToList();
        int Idx(string n) => Array.IndexOf(names, n);

        // (1) RECALL — a noisy cat relaxes back to cat (pattern completion).
        var rng = new Random(7);
        var noisy = (double[])attractors[Idx("cat")].Clone();
        for (var i = 0; i < Dim; i++) noisy[i] += 0.03 * (rng.NextDouble() * 2 - 1);
        var recall = Relax(Norm(noisy), attractors);
        _out.WriteLine($"RECALL  noisy-cat → {names[recall.Best]}  (init {recall.InitMaxSim:F3} → settled {recall.FinalMaxSim:F3})");
        Assert.Equal("cat", names[recall.Best]);

        // (2) DISAMBIGUATION — the SAME word 'bank' relaxes into DIFFERENT basins under context.
        var rb = Relax(Cloud(new[] { "bank", "river", "water" }), attractors);
        var mb = Relax(Cloud(new[] { "bank", "money", "loan" }), attractors);
        _out.WriteLine($"DISAMBIG  bank+river → {names[rb.Best]}   bank+money → {names[mb.Best]}");
        Assert.Equal("stream", names[rb.Best]); // river context settles to the river sense
        Assert.Equal("fund", names[mb.Best]);   // money context settles to the money sense

        // (3) ABSTENTION — an unknown query stays high-surprise (no near basin); it never confidently belongs.
        var unknown = Relax(Cloud(new[] { "zxqv", "wmkp" }), attractors);
        _out.WriteLine($"ABSTAIN  unknown → init-surprise {1 - unknown.InitMaxSim:F3} (known cat surprise {1 - recall.InitMaxSim:F3})");
        Assert.True(unknown.InitMaxSim < 0.3, "an unknown query has no near basin — high free energy → abstain");
        Assert.True(recall.InitMaxSim - unknown.InitMaxSim > 0.3, "surprise cleanly separates known (settles) from unknown (does not)");
    }
}
