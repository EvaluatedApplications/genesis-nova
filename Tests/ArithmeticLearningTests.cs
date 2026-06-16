using System;
using System.Globalization;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The platonic space computes symbolic arithmetic EXACTLY via the face homomorphism (no NN). Fast,
/// deterministic — the foundational guarantee the learned paths route TO. Drives the REAL substrate
/// arithmetic path (<see cref="PlatonicGliderInterpreter.ComposeArithmetic"/> → R2 compose tick →
/// <see cref="PlatonicFaceDecoder"/>), not a hand-rolled face blend, so it verifies the production
/// component rather than a parallel reimplementation of it.
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
        var gliderOp = op switch
        {
            "+" => GliderOp.Add,
            "-" => GliderOp.Subtract,
            "*" => GliderOp.Multiply,
            "/" => GliderOp.Divide,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "unknown op"),
        };

        // Drive the REAL substrate arithmetic: compose the operands into a platonic Composition element
        // via the R2 compose tick, then decode the result through the face homomorphism — exactly the
        // production path. No hand-rolled face blend.
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        var interp = new PlatonicGliderInterpreter(space);
        var glider = new PlatonicGlider("arith",
            new Compute(gliderOp, new GliderBlock[] { new Operand(0), new Operand(1) }));

        var value = double.Parse(
            interp.Execute(glider, new[]
            {
                a.ToString(CultureInfo.InvariantCulture),
                b.ToString(CultureInfo.InvariantCulture),
            }),
            CultureInfo.InvariantCulture);

        _out.WriteLine($"{a}{op}{b}: value={value} expected={expected}");
        Assert.Equal(expected, value, 6);
    }
}
