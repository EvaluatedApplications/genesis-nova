using System.Collections.Generic;
using System.Globalization;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// An OPERATOR in the field (PLATONIC_MIND.md): an element whose dynamics TRANSFORM the thought rather than answer
/// it. Where a concept is a basin the thought settles INTO (retrieval), an operator is something the thought
/// applies (compute / compose / transform). The field reasons over concepts AND operators; the GRU's old job of
/// "pick the glider / op / shape" becomes the field choosing the lowest-surprise next move. <see cref="TryApply"/>
/// returns the operator's result if it fires for the current thought-context, or null if it abstains.
/// </summary>
public interface IFieldOperator
{
    string Name { get; }
    string? TryApply(IReadOnlyList<string> context);
}

/// <summary>
/// The exact ARITHMETIC operator — the homomorphism as an operator the field can apply. When the thought-context
/// holds two numeric operands and an operation cue, it computes the result EXACTLY via the substrate's own glider
/// interpreter (the production arithmetic path), generalising to unseen operands with no stored facts. This is the
/// first operator; gliders / transforms / fold-paths follow the same shape.
/// </summary>
public sealed class ArithmeticFieldOperator : IFieldOperator
{
    private readonly DialecticalSpace _field;
    public ArithmeticFieldOperator(DialecticalSpace field) => _field = field;
    public string Name => "arithmetic";

    public string? TryApply(IReadOnlyList<string> context)
    {
        var nums = new List<string>();
        GliderOp? op = null;
        foreach (var tok in context ?? new List<string>())
        {
            if (double.TryParse(tok, NumberStyles.Float, CultureInfo.InvariantCulture, out _)) nums.Add(tok);
            else { var o = ParseOp(tok); if (o.HasValue) op = o; }
        }
        if (nums.Count < 2 || op is null) return null;

        var interp = new PlatonicGliderInterpreter(_field);
        var glider = new PlatonicGlider("op", new Compute(op.Value, new GliderBlock[] { new Operand(0), new Operand(1) }));
        try { return interp.Execute(glider, new[] { nums[0], nums[1] }); }
        catch { return null; }
    }

    private static GliderOp? ParseOp(string t) => t switch
    {
        "+" or "plus" or "add" => GliderOp.Add,
        "-" or "minus" or "sub" => GliderOp.Subtract,
        "x" or "*" or "times" or "mul" => GliderOp.Multiply,
        "/" or "div" or "over" => GliderOp.Divide,
        _ => (GliderOp?)null,
    };
}
