using System;
using System.Linq;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// EXPERIMENT (2026-06-14): characterise what the TransformAccumulator's learned-function mechanism can
/// actually generalise, to decide how to wire it to the GRU. A learned transform is T(f)=avg(embed(out)-
/// embed(in)) and Apply(f,x)=embed(x)+T(f) — a CONSTANT translation in face space. Hypothesis: it
/// generalises a function from a FEW examples IFF that function is a constant face-translation, i.e. the
/// numeric/affine class (poly face: +k; log face: ×k), NOT arbitrary symbolic maps (char/word deltas vary).
/// </summary>
public sealed class TransformLearningExperimentTests
{
    private readonly ITestOutputHelper _out;
    public TransformLearningExperimentTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Transform_GeneralisesAffineNumericFunctions_FromFewExamples()
    {
        var dim = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize).FaceDimension;

        void Learn(TransformAccumulator acc, string fn, long i, long o)
            => acc.Learn(fn, PlatonicFaceComposer.Compose(i.ToString(), dim),
                             PlatonicFaceComposer.Compose(o.ToString(), dim));

        long Apply(TransformAccumulator acc, string fn, long x, int preferFace)
        {
            var pred = acc.Apply(fn, PlatonicFaceComposer.Compose(x.ToString(), dim))!;
            var (val, _, _) = PlatonicFaceDecoder.DecodeNumericFromPrediction(pred, dim, preferFace);
            return (long)Math.Round(val);
        }

        // (1) "+5" — a constant in the POLY face (embed(n+5)-embed(n) = 5·10^-i, same for every n).
        var plus5 = new TransformAccumulator(dim);
        foreach (var n in new long[] { 1, 3, 4, 8 }) Learn(plus5, "plus5", n, n + 5);
        var plus5Held = new long[] { 7, 10, 12, 20 };
        var plus5Hits = plus5Held.Count(n => Apply(plus5, "plus5", n, 1) == n + 5);
        foreach (var n in plus5Held) _out.WriteLine($"  +5: {n} -> {Apply(plus5, "plus5", n, 1)} (exp {n + 5})");

        // (2) "double" — a constant in the LOG face (embed(2n)-embed(n) = ln2·10^-i, same for every n).
        var dbl = new TransformAccumulator(dim);
        foreach (var n in new long[] { 2, 3, 5, 8 }) Learn(dbl, "double", n, n * 2);
        var dblHeld = new long[] { 7, 9, 11, 15 };
        var dblHits = dblHeld.Count(n => Apply(dbl, "double", n, 2) == n * 2);
        foreach (var n in dblHeld) _out.WriteLine($"  x2: {n} -> {Apply(dbl, "double", n, 2)} (exp {n * 2})");

        _out.WriteLine($"EXPERIMENT: +5 held-out {plus5Hits}/4, double held-out {dblHits}/4 (trained on 4 each)");

        // The claim under test: a learned transform generalises an affine/multiplicative function to
        // UNSEEN inputs from a handful of examples — the core few-shot-function value worth wiring.
        Assert.True(plus5Hits >= 3, $"+5 (poly) should generalise from 4 examples; got {plus5Hits}/4");
        Assert.True(dblHits >= 3, $"double (log) should generalise from 4 examples; got {dblHits}/4");
    }
}
