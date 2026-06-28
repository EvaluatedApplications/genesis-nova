using System;
using System.Diagnostics;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// PHASE-1 EXPERIMENT (SYSTEM_OVERVIEW.md — "statistics on the GPU"): the substrate's distributional hot loop is
// RecomputeCloud — Cloud(e) = token(e) + Σ_n aff(e,n)·token(n), recomputed for BOTH endpoints on EVERY observation.
// It is structurally a sparse matmul Cloud = A·T (affinity-adjacency × deterministic token matrix). This file:
//   (1) EQUIVALENCE — pins the live cloud to its mathematical definition (the reference distributional sum). This is the
//       bit-identical guard for Win #1 (token caching) AND the tolerance harness the batched-GPU path (Win #2) must pass.
//   (2) BENCHMARK   — measures observation throughput at realistic (hub-heavy) scale, establishing the baseline number
//       the GPU rebalance is judged against. [SlowFact] — opt-in, never in the fast suite.
public sealed class CloudRecomputeBenchmark
{
    private readonly ITestOutputHelper _out;
    public CloudRecomputeBenchmark(ITestOutputHelper o) => _out = o;

    private static double Clamp01(double v) => v < 0 ? 0 : v > 1 ? 1 : v;

    // The reference cloud straight from the definition, using only PUBLIC primitives (FaceCodec.Token + GetContradiction):
    // token(self) + Σ over neighbours of (1 − 2·contradiction)·token(neighbour), then unit-normalized — exactly what
    // SemanticVectorOf returns. Order-robust comparison is by cosine (float add is non-associative; the algebra is identical).
    private static double[] ReferenceCloud(DialecticalSpace space, string self, string[] neighbours, int dim)
    {
        var n = FaceCodec.SemanticLength(dim);
        var cloud = (double[])FaceCodec.Token(self, dim).Clone();
        foreach (var nb in neighbours)
        {
            var aff = 1.0 - 2.0 * Clamp01(space.GetContradiction(self, nb));
            var t = FaceCodec.Token(nb, dim);
            for (var i = 0; i < n; i++) cloud[i] += aff * t[i];
        }
        var s = 0.0; for (var i = 0; i < n; i++) s += cloud[i] * cloud[i];
        var inv = s > 1e-12 ? 1.0 / Math.Sqrt(s) : 0.0;
        for (var i = 0; i < n; i++) cloud[i] *= inv;
        return cloud;
    }

    private static double Cosine(double[] a, double[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length && i < b.Length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    [Fact]
    public void Live_cloud_equals_the_reference_distributional_sum()
    {
        // dim MUST exceed the word-face start (202) or the semantic length is 0 and every cloud is empty.
        const int dim = 512;
        var space = new DialecticalSpace(faceDimension: dim);
        // A small graph with a HUB ("the") wired to many content words + intra-cluster edges — the structure RecomputeCloud
        // walks. We track each concept's neighbour set so the reference can reproduce its cloud exactly.
        string[] content = { "cat", "dog", "cow", "red", "blue", "green", "bread", "cake", "soup", "car", "bus", "van" };
        var rng = new Random(11);
        var nbrs = content.ToDictionary(c => c, _ => new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal));
        void Obs(string a, string b, double k) { space.ObserveContradiction(a, b, k); if (nbrs.ContainsKey(a)) nbrs[a].Add(b); if (nbrs.ContainsKey(b)) nbrs[b].Add(a); }

        for (var step = 0; step < 4000; step++)
        {
            var a = content[rng.Next(content.Length)];
            var b = content[rng.Next(content.Length)];
            if (a != b) Obs(a, b, 0.1);
            Obs("the", content[rng.Next(content.Length)], 0.5); // hub ↔ any content (but "the" isn't a tracked content node)
        }

        foreach (var c in content)
        {
            var live = space.SemanticVectorOf(c);
            Assert.NotNull(live);
            var reference = ReferenceCloud(space, c, nbrs[c].ToArray(), dim);
            var cos = Cosine(live!, reference);
            _out.WriteLine($"{c,-7} cos(live, reference) = {cos:F12}");
            Assert.True(cos > 0.999999, $"{c}: live cloud diverged from the distributional reference (cos {cos:F9})");
        }
    }

    [Fact]
    public void Batched_gpu_cloud_matches_reference_within_float_tolerance()
    {
        const int dim = 512;
        var space = new DialecticalSpace(faceDimension: dim) { BatchedCloudGpu = true, CloudFlushInterval = 4096 };
        string[] content = { "cat", "dog", "cow", "red", "blue", "green", "bread", "cake", "soup", "car", "bus", "van" };
        var rng = new Random(11);
        var nbrs = content.ToDictionary(c => c, _ => new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal));
        void Obs(string a, string b, double k) { space.ObserveContradiction(a, b, k); if (nbrs.ContainsKey(a)) nbrs[a].Add(b); if (nbrs.ContainsKey(b)) nbrs[b].Add(a); }
        for (var step = 0; step < 4000; step++)
        {
            var a = content[rng.Next(content.Length)];
            var b = content[rng.Next(content.Length)];
            if (a != b) Obs(a, b, 0.1);
            Obs("the", content[rng.Next(content.Length)], 0.5);
        }
        space.FlushCloudBatch(); // fold the final partial batch

        foreach (var c in content)
        {
            var live = space.SemanticVectorOf(c); // also flushes if anything pending
            Assert.NotNull(live);
            var reference = ReferenceCloud(space, c, nbrs[c].ToArray(), dim);
            var cos = Cosine(live!, reference);
            _out.WriteLine($"{c,-7} cos(batched-gpu, reference) = {cos:F9}");
            // float32 on the GPU vs double reference — a looser bar than the bit-identical scalar path, but still tight.
            Assert.True(cos > 0.999, $"{c}: batched-GPU cloud diverged from the reference (cos {cos:F6})");
        }
    }

    // Drives the identical hub-heavy workload over a given space and returns elapsed seconds. dim=512 (production), a Zipf-ish
    // vocabulary with a few high-degree HUBS (function words) that make the cloud sum expensive (cost is O(degree)·semLen).
    private static double RunWorkload(DialecticalSpace space, int vocab, int observations)
    {
        var words = Enumerable.Range(0, vocab).Select(i => "w" + i).ToArray();
        string[] hubs = { "the", "of", "a", "and", "to" };
        var rng = new Random(7);                                            // SAME seed for every space → identical workload
        for (var i = 0; i < 20_000; i++) space.ObserveContradiction(hubs[rng.Next(hubs.Length)], words[rng.Next(vocab)], 0.4); // warm hubs
        var sw = Stopwatch.StartNew();
        for (var i = 0; i < observations; i++)
        {
            if (rng.NextDouble() < 0.7) space.ObserveContradiction(hubs[rng.Next(hubs.Length)], words[rng.Next(vocab)], 0.4); // 70% hub
            else { var a = words[rng.Next(vocab)]; var b = words[rng.Next(vocab)]; if (a != b) space.ObserveContradiction(a, b, 0.2); }
        }
        space.FlushCloudBatch();                                            // fold the final partial batch (no-op for scalar)
        sw.Stop();
        return sw.Elapsed.TotalSeconds;
    }

    [SlowFact]
    public void Bench_scalar_vs_batched_gpu_throughput()
    {
        const int dim = 512, vocab = 4000;
        // Headline baseline (CPU scalar, Win #1 caching on): 400k obs → 638 obs/s, 1568 µs/obs (10m40s). Default smaller here
        // so the comparison is quick; override with BENCH_OBS for the full run.
        var observations = int.TryParse(Environment.GetEnvironmentVariable("BENCH_OBS"), out var o) ? o : 120_000;

        var scalar = new DialecticalSpace(faceDimension: dim) { DischargeInterval = long.MaxValue };
        var scalarSec = RunWorkload(scalar, vocab, observations);

        var batched = new DialecticalSpace(faceDimension: dim) { DischargeInterval = long.MaxValue, BatchedCloudGpu = true, CloudFlushInterval = 8192 };
        var batchedSec = RunWorkload(batched, vocab, observations);

        _out.WriteLine($"dim={dim} vocab={vocab} observations={observations:N0}");
        _out.WriteLine($"SCALAR (CPU)      {scalarSec,8:F2}s   {observations / scalarSec,10:N0} obs/s   {scalarSec * 1e6 / observations,8:F2} µs/obs");
        _out.WriteLine($"BATCHED (GPU/CPU) {batchedSec,8:F2}s   {observations / batchedSec,10:N0} obs/s   {batchedSec * 1e6 / observations,8:F2} µs/obs");
        _out.WriteLine($"SPEEDUP           {scalarSec / batchedSec,8:F1}×   (dedup + batched matmul; final clouds equivalent — see the [Fact] tests)");
        Assert.True(batchedSec > 0 && scalarSec > 0);
    }
}
