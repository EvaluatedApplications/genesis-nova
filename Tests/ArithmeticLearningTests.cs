using System;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The platonic space computes symbolic arithmetic EXACTLY via the face homomorphism (no NN). Fast,
/// deterministic — the foundational guarantee the learned paths route TO.
///
/// (The model-LEARNS-to-route / number-word-equivalence-geometry / worded-arithmetic experiments that
/// once lived here were one-off training investigations; their findings are recorded in memory and
/// the live capabilities are covered by the canonical end-to-end demos — CoreBootstrapRegimeTests and
/// GruQueryConstructionTests — plus the fast unit tests in QueryLabelTests / CoreBootstrapTests.)
/// </summary>
public sealed class ArithmeticLearningTests
{
    private readonly ITestOutputHelper _out;
    public ArithmeticLearningTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData(1, 1, "+", 2)]
    [InlineData(2, 3, "+", 5)]
    [InlineData(0, 0, "+", 0)]
    [InlineData(7, 4, "-", 3)]
    [InlineData(4, 5, "*", 20)]
    [InlineData(8, 2, "/", 4)]
    [InlineData(5, 5, "/", 1)]   // result == 1 → log slot ~0 (edge case for the decoder)
    [InlineData(1, 1, "*", 1)]
    public void PlatonicArithmetic_IsExact(double a, double b, string op, double expected)
    {
        var dim = ProductionDims.FaceDimension;
        var faceA = PlatonicFaceComposer.GetFreshNumericEmbedding(a, dim);
        var faceB = PlatonicFaceComposer.GetFreshNumericEmbedding(b, dim);
        var additive = op is "+" or "-";
        var sign = op is "-" or "/" ? -1.0 : 1.0;

        var blended = new double[dim];
        for (var i = 0; i < dim; i++)
            blended[i] = faceA[i] + (sign * faceB[i]);

        var preferFace = additive ? 1 : 2; // 1=poly (add/sub), 2=log (mul/div)
        var (value, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(blended, dim, preferFace);
        if (!additive && face == "log")
            value *= Math.Sign(a) * Math.Sign(b);

        _out.WriteLine($"{a}{op}{b}: value={value} expected={expected} quality={quality:F3} face={face}");
        Assert.Equal(expected, value, 6);
    }
}
