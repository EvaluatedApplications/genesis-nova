using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// PHASE 1 — DIMENSIONAL CONTRADICTION (the dialectic; PLATONIC_DIALECTIC.md §2, PLATONIC_THEORY.md Laws M/D).
///
/// The legacy scalar pull (ObserveContradiction → UpdateConceptGeometry → MessagePassUpdate with ONE affinity
/// applied to EVERY free dim) contracts all dims of a related pair toward the neighbour, driving the two concepts
/// toward IDENTICAL — it erases the contrasts that give them distinct meaning. That violates Law M ("meaning is
/// differential": cat and dog must agree on `animal` yet differ on `sound"). The per-dimension dialectic instead
/// pulls together in the dims where the pair AGREES and PRESERVES the dims where it CONTRADICTS, so a concept's
/// place is the synthesis of what it shares and what it opposes.
///
/// This probe measures the signature head-to-head, controlling for everything but the mechanism. We create a
/// related pair (cat↔dog) and an unrelated control (rock), partition cat/dog's word-face dims by initial
/// sign-agreement, observe the pair as related many times, and measure how the AGREEING vs CONTRADICTING dims
/// move. Baseline (DimensionalContradiction OFF): contradicting dims COLLAPSE alongside agreeing ones. Dialectic
/// (ON): contradicting dims are PRESERVED while agreeing dims still converge — and the related pair stays clearly
/// closer than the unrelated control (no separation regression). Fast, no NN, production face dimension.
/// </summary>
public sealed class PlatonicDialecticTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicDialecticTests(ITestOutputHelper o) => _out = o;

    private readonly record struct Signature(
        double AgreeConverge,   // word-face distance on AGREE dims, after/before (want < 1: drew closer)
        double OppPreserve,     // word-face distance on CONTRADICT dims, after/before (collapse→0; preserve→≈1)
        double DistRelated,     // final cat↔dog word-face distance
        double DistUnrelated);  // final cat↔rock word-face distance (control)

    private static double WordFaceDist(IReadOnlyList<double> a, IReadOnlyList<double> b, int start, int dim)
    {
        var s = 0.0;
        for (var i = start; i < dim; i++) { var d = a[i] - b[i]; s += d * d; }
        return Math.Sqrt(s);
    }

    private static double MeanAbsDiff(double[] a, double[] b, IReadOnlyList<int> dims)
    {
        if (dims.Count == 0) return 0.0;
        var s = 0.0;
        foreach (var i in dims) s += Math.Abs(a[i] - b[i]);
        return s / dims.Count;
    }

    private Signature RunCatDog(bool dimensional)
    {
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7)
        {
            DimensionalContradiction = dimensional,
        };
        var dim = space.FaceDimension;
        var wordStart = FaceLayout.WordFaceStart(dim);

        // Create the pair (related) + an unrelated control (rock, related only to stone). Snapshot AFTER the first
        // observe so the per-concept identity codes are already written into the word face.
        space.ObserveContradiction("cat", "dog", 0.1);   // related: low contradiction → attract
        space.ObserveContradiction("rock", "stone", 0.1); // control lives in its own pair, never tied to cat/dog
        Assert.True(space.TryGetConceptFace("cat", out var cat0));
        Assert.True(space.TryGetConceptFace("dog", out var dog0));
        cat0 = (double[])cat0.Clone();
        dog0 = (double[])dog0.Clone();

        // Partition the word face by current per-dimension agreement of the pair.
        var agreeDims = new List<int>();
        var oppDims = new List<int>();
        for (var i = wordStart; i < dim; i++)
            (cat0[i] * dog0[i] >= 0 ? agreeDims : oppDims).Add(i);
        Assert.True(agreeDims.Count > 8 && oppDims.Count > 8, "need both agreeing and contradicting word dims");

        var d0Agree = MeanAbsDiff(cat0, dog0, agreeDims);
        var d0Opp = MeanAbsDiff(cat0, dog0, oppDims);

        // Observe the pair as related many times — this is where the scalar pull collapses everything.
        for (var k = 0; k < 60; k++)
            space.ObserveContradiction("cat", "dog", 0.1);

        Assert.True(space.TryGetConceptFace("cat", out var cat1));
        Assert.True(space.TryGetConceptFace("dog", out var dog1));
        Assert.True(space.TryGetConceptFace("rock", out var rock1));
        var d1Agree = MeanAbsDiff((double[])cat1, (double[])dog1, agreeDims);
        var d1Opp = MeanAbsDiff((double[])cat1, (double[])dog1, oppDims);

        return new Signature(
            AgreeConverge: d0Agree > 1e-9 ? d1Agree / d0Agree : 1.0,
            OppPreserve: d0Opp > 1e-9 ? d1Opp / d0Opp : 1.0,
            DistRelated: WordFaceDist(cat1, dog1, wordStart, dim),
            DistUnrelated: WordFaceDist(cat1, rock1, wordStart, dim));
    }

    [Fact]
    public void DimensionalContradiction_PreservesContrast_WhereScalarCollapsesIt()
    {
        var scalar = RunCatDog(dimensional: false);
        var dialectic = RunCatDog(dimensional: true);

        _out.WriteLine("                agreeConverge  oppPreserve   related   unrelated");
        _out.WriteLine($"scalar baseline   {scalar.AgreeConverge:F3}         {scalar.OppPreserve:F3}        {scalar.DistRelated:F3}     {scalar.DistUnrelated:F3}");
        _out.WriteLine($"dialectic (P1)    {dialectic.AgreeConverge:F3}         {dialectic.OppPreserve:F3}        {dialectic.DistRelated:F3}     {dialectic.DistUnrelated:F3}");

        // 1. BASELINE: the scalar pull collapses the contradicting dims along with the agreeing ones (Law M
        //    violation — the pair loses its distinguishing aspects).
        Assert.True(scalar.OppPreserve < 0.9,
            $"premise: scalar pull should erode contrast on contradicting dims; oppPreserve={scalar.OppPreserve:F3}");

        // 2. THE FIX: the dialectic preserves contradicting-dim contrast markedly better than the scalar baseline.
        Assert.True(dialectic.OppPreserve > scalar.OppPreserve + 0.15,
            $"dialectic must preserve contradicting-dim contrast; dialectic={dialectic.OppPreserve:F3} vs scalar={scalar.OppPreserve:F3}");

        // 3. STILL LEARNS: the dialectic still draws the pair together on the dims where they AGREE.
        Assert.True(dialectic.AgreeConverge < 0.98,
            $"dialectic must still converge agreeing dims; agreeConverge={dialectic.AgreeConverge:F3}");

        // 4. NO SEPARATION REGRESSION: the related pair stays clearly closer than the unrelated control.
        Assert.True(dialectic.DistRelated < dialectic.DistUnrelated,
            $"related must stay closer than unrelated; related={dialectic.DistRelated:F3} unrelated={dialectic.DistUnrelated:F3}");
    }
}
