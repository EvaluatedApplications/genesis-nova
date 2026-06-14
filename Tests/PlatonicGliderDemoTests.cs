using System;
using System.Collections.Generic;
using System.Globalization;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Small, focused tests for the GLIDER BLOCK vocabulary on the main-engine interpreter
/// (<see cref="PlatonicGliderInterpreter"/>). One test per block verifies it works on the real platonic
/// physics (relation hops + face homomorphism); a final test shows the SAME blocks recompose across
/// arithmetic AND retrieval (the anti-overfitting point). Fast, production face dimension.
/// </summary>
public sealed class PlatonicGliderDemoTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicGliderDemoTests(ITestOutputHelper output) => _out = output;

    private static readonly (int Value, string Word)[] Vocab =
    [
        (0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five"),
        (6, "six"), (7, "seven"), (8, "eight"), (9, "nine"), (10, "ten"), (11, "eleven"),
        (12, "twelve"), (13, "thirteen"), (14, "fourteen"), (15, "fifteen"), (16, "sixteen"),
        (17, "seventeen"), (18, "eighteen"),
    ];

    private const string Scaffold = "the answer is";

    // Seed the atoms the blocks compose: number-word equivalence, the scaffold chunk, and two FACT
    // relations (france→paris, italy→rome) so the same blocks can answer non-arithmetic questions.
    private static PlatonicSpaceMemory Space()
    {
        var space = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        foreach (var (value, word) in Vocab)
        {
            var digit = value.ToString(CultureInfo.InvariantCulture);
            for (var k = 0; k < 8; k++)
            {
                space.ObserveContradiction(word, digit, 0.02);
                space.ObserveContradiction(digit, word, 0.02);
            }
        }
        for (var k = 0; k < 8; k++)
        {
            space.ObserveContradiction(Scaffold, "arithmetic-answer", 0.02);
            space.ObserveContradiction("france", "paris", 0.02);
            space.ObserveContradiction("italy", "rome", 0.02);
        }
        return space;
    }

    private static PlatonicGliderInterpreter Interp() => new(Space());

    [Fact] // Operand + Literal + Seq: read inputs and assemble text.
    public void Operand_Literal_Seq_AssembleText()
    {
        var g = new PlatonicGlider("echo", new Seq([new Literal("you said"), new Operand(0)]));
        Assert.Equal("you said hello", Interp().Execute(g, ["hello"]));
    }

    [Fact] // Hop: the one general relational-retrieval primitive — resolve / format / retrieve.
    public void Hop_Resolves_Formats_AndRetrieves()
    {
        var interp = Interp();
        Assert.Equal("3", interp.Execute(new PlatonicGlider("resolve", new Hop(new Operand(0), HopTarget.Number)), ["three"]));
        Assert.Equal("seven", interp.Execute(new PlatonicGlider("format", new Hop(new Operand(0), HopTarget.Word)), ["7"]));
        Assert.Equal("paris", interp.Execute(new PlatonicGlider("fact", new Hop(new Operand(0), HopTarget.Any)), ["france"]));
    }

    [Fact] // Compute: exact arithmetic via the face homomorphism, generalising to held-out operands.
    public void Compute_IsExact_AndGeneralises()
    {
        var interp = Interp();
        Assert.Equal("2", interp.Execute(new PlatonicGlider("add", new Compute(GliderOp.Add, [new Operand(0), new Operand(1)])), ["one", "one"]));
        Assert.Equal("18", interp.Execute(new PlatonicGlider("add", new Compute(GliderOp.Add, [new Operand(0), new Operand(1)])), ["9", "9"])); // held-out
        Assert.Equal("3", interp.Execute(new PlatonicGlider("sub", new Compute(GliderOp.Subtract, [new Operand(0), new Operand(1)])), ["8", "5"]));
        Assert.Equal("6", interp.Execute(new PlatonicGlider("mul", new Compute(GliderOp.Multiply, [new Operand(0), new Operand(1)])), ["2", "3"]));
    }

    [Fact] // REFACTOR #1: Compute is now element-native — a platonic Composition element (R2) — and
           // produces identical results to the kept meta-layer oracle (ComputeDirect).
    public void Compute_IsAPlatonicCompositionElement_MatchingOracle()
    {
        var interp = Interp();

        // It is a genuine platonic form built by the substrate's R2 rule, not a C# value.
        var element = interp.ComposeArithmetic(GliderOp.Add, new[] { 2.0, 2.0 });
        Assert.Equal(ElementKind.Composition, element.Kind);
        Assert.StartsWith("R2:compose", element.GenerationPath);

        // Element-native glider output == oracle, across a sweep of all ops and operands.
        foreach (var op in new[] { GliderOp.Add, GliderOp.Subtract, GliderOp.Multiply, GliderOp.Divide })
        {
            var g = new PlatonicGlider("c", new Compute(op, new GliderBlock[] { new Operand(0), new Operand(1) }));
            for (var a = 0; a <= 9; a++)
            {
                for (var b = 1; b <= 9; b++)
                {
                    var got = double.Parse(interp.Execute(g, new[] { a.ToString(), b.ToString() }), CultureInfo.InvariantCulture);
                    Assert.Equal(interp.ComputeDirect(op, new[] { (double)a, b }), got);
                }
            }
        }
        _out.WriteLine("Compute refactored to R2 Composition element; matches oracle across add/sub/mul/div × [0..9].");
    }

    [Fact] // Const: parameterised computation (e.g. "double" = ×2).
    public void Const_ParameterisesComputation()
    {
        var dbl = new PlatonicGlider("double", new Compute(GliderOp.Multiply, [new Operand(0), new Const(2)]));
        Assert.Equal("8", Interp().Execute(dbl, ["4"]));
    }

    [Fact] // Fold: VARIADIC reduce of an op over all operands (the functional fold; arbitrary arity).
    public void Fold_ReducesVariadically()
    {
        var interp = Interp();
        var sum = new PlatonicGlider("sum", new Fold(GliderOp.Add, 0));
        Assert.Equal("6", interp.Execute(sum, ["1", "2", "3"]));
        Assert.Equal("100", interp.Execute(sum, ["10", "20", "30", "40"]));
        var product = new PlatonicGlider("product", new Fold(GliderOp.Multiply, 0));
        Assert.Equal("24", interp.Execute(product, ["2", "3", "4"]));
    }

    [Fact] // Compare: numeric predicate → boolean (1/0).
    public void Compare_YieldsPredicate()
    {
        var interp = Interp();
        Assert.Equal("1", interp.Execute(new PlatonicGlider("gt", new Compare(CompareOp.Greater, new Operand(0), new Operand(1))), ["5", "3"]));
        Assert.Equal("0", interp.Execute(new PlatonicGlider("gt", new Compare(CompareOp.Greater, new Operand(0), new Operand(1))), ["3", "5"]));
    }

    [Fact] // Branch: conditional selection on a predicate (yes/no answers, format decisions).
    public void Branch_SelectsOnCondition()
    {
        var g = new PlatonicGlider("answer:greater",
            new Seq([new Literal(Scaffold),
                     new Branch(new Compare(CompareOp.Greater, new Operand(0), new Operand(1)),
                                new Literal("yes"), new Literal("no"))]));
        var interp = Interp();
        Assert.Equal("the answer is yes", interp.Execute(g, ["5", "3"]));
        Assert.Equal("the answer is no", interp.Execute(g, ["3", "5"]));
        Assert.Equal("the answer is no", interp.Execute(g, ["two", "three"])); // words resolve through Compare
    }

    [Fact] // Ref: HIGHER-ORDER — a glider that invokes another named glider from the library.
    public void Ref_InvokesAnotherGlider()
    {
        var add = new PlatonicGlider("add", new Seq([new Literal(Scaffold), new Compute(GliderOp.Add, [new Operand(0), new Operand(1)])]));
        var library = new Dictionary<string, PlatonicGlider> { ["add"] = add };
        var interp = new PlatonicGliderInterpreter(Space(), library);
        Assert.Equal("the answer is 2", interp.Execute(new PlatonicGlider("wrapper", new Ref("add")), ["one", "one"]));
    }

    [Fact] // The SAME blocks recompose across tasks — general primitives, not a bespoke template.
    public void SameBlocks_Recompose_AcrossArithmeticAndRetrieval()
    {
        var interp = Interp();

        var addDigit = new PlatonicGlider("answer:add:digit",
            new Seq([new Literal(Scaffold), new Compute(GliderOp.Add, [new Operand(0), new Operand(1)])]));
        Assert.Equal("the answer is 2", interp.Execute(addDigit, ["one", "one"]));
        Assert.Equal("the answer is 18", interp.Execute(addDigit, ["9", "9"]));

        var addWord = new PlatonicGlider("answer:add:word",
            new Seq([new Literal(Scaffold), new Hop(new Compute(GliderOp.Add, [new Operand(0), new Operand(1)]), HopTarget.Word)]));
        Assert.Equal("the answer is two", interp.Execute(addWord, ["one", "one"]));

        var retrieve = new PlatonicGlider("answer:retrieve",
            new Seq([new Literal(Scaffold), new Hop(new Operand(0), HopTarget.Any)]));
        Assert.Equal("the answer is paris", interp.Execute(retrieve, ["france"]));
        Assert.Equal("the answer is rome", interp.Execute(retrieve, ["italy"]));

        _out.WriteLine("same blocks: add->2/18, add:word->two, retrieve->paris/rome");
    }
}
