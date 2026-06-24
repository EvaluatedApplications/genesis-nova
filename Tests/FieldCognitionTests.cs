using System.Collections.Generic;
using System.Globalization;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The NN-layer rework's first kernel (PLATONIC_MIND.md): the field GENERATES by relaxation. A thought seeded in a
/// domain stays coherent (it speaks only from settled basins), and a thought with no basin falls silent (abstains).
/// Pure relaxation, no learned weights yet — this proves the loop before the dynamics are made learnable/immanent.
/// </summary>
public sealed class FieldCognitionTests
{
    private readonly ITestOutputHelper _out;
    public FieldCognitionTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_GeneratesCoherently_ByRelaxation_AndAbstains()
    {
        var s = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        void Rel(string a, params string[] c) { foreach (var x in c) s.ObserveContradiction(a, x, 0.05); }
        var vehicles = new HashSet<string> { "car", "truck", "vehicle", "road", "engine", "drive", "haul", "wheel" };
        for (var ep = 0; ep < 10; ep++)
        {
            Rel("cat", "animal", "pet", "fur", "purr");
            Rel("dog", "animal", "pet", "fur", "bark");
            Rel("lion", "animal", "wild", "roar");
            Rel("car", "vehicle", "road", "engine");
            Rel("truck", "vehicle", "road", "haul");
        }
        var mind = new FieldCognition(s);

        // Coherent thematic generation: a thought seeded in the animal domain stays in it (never wanders to vehicles).
        var animal = mind.Think(new[] { "animal" }, 5);
        _out.WriteLine($"animal → {string.Join(" ", animal)}");
        Assert.True(animal.Count >= 3, "the field produces a continuation by relaxation");
        Assert.All(animal, w => Assert.DoesNotContain(w, vehicles));

        // Abstention: an unknown seed produces nothing — there is no coherent thought to speak.
        var nothing = mind.Think(new[] { "zxqv" }, 3);
        _out.WriteLine($"unknown → [{string.Join(" ", nothing)}] ({nothing.Count})");
        Assert.Empty(nothing);
    }

    [Fact] // The field COMPUTES, not just recalls: an operator (the exact homomorphism) fires when the context affords
           // it — generalising to unseen operands. This is the platonic space's computation integrated into the loop.
    public void Field_Computes_ExactArithmetic_AsAnOperator()
    {
        var s = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var mind = new FieldCognition(s, new IFieldOperator[] { new ArithmeticFieldOperator(s) });
        double R(params string[] p) { var r = mind.Think(p, 1); Assert.Single(r); return double.Parse(r[0], CultureInfo.InvariantCulture); }

        Assert.Equal(8.0, R("3", "+", "5"), 6);
        Assert.Equal(3.0, R("7", "-", "4"), 6);
        Assert.Equal(48.0, R("12", "x", "4"), 6);
        Assert.Equal(141.0, R("84", "+", "57"), 6); // unseen operands — derived, exact

        // No operator fires on a non-arithmetic prompt → the loop falls through to retrieval (here: empty world → silent).
        Assert.Empty(mind.Think(new[] { "hello", "world" }, 2));
    }
}
