using System;
using System.Collections.Generic;

namespace GenesisNova.Cognition;

/// <summary>
/// The catalogue of NAMED, REUSABLE shapes the composer can reference — the element-native realization of
/// "shapes are <see cref="Core.ElementKind.Function"/> elements in the space." Each registered shape is
/// (1) a glider (a composition of the primitive block vocabulary — NOT a premade answer table), and
/// (2) a positioned Function element written into <see cref="PlatonicSpaceMemory"/> whose RelatedTo points
/// at the other shape-elements it composes. A Ref shape is therefore a Function element referencing other
/// Function elements; the substrate executes it by traversing + composing them (the interpreter's recursive
/// <c>EvalRef</c> over <see cref="Library"/>). The GRU only SELECTS the shape (the plan head); the shape's
/// definition is block composition, and its result is computed on the substrate for any operands.
///
/// The seeds here are the higher-order vocabulary the Ref shape needs:
///  • <c>larger</c>      — max(a,b): Compare(Greater) → Branch(a, b). Pure blocks.
///  • <c>twicelarger</c> — 2·max(a,b): Compute(Multiply, Ref("larger"), Const(2)) — a composition-OF-
///    compositions (it references <c>larger</c>), the canonical glider-of-gliders.
/// </summary>
public sealed class PlatonicShapeRegistry
{
    /// <summary>The single higher-order Ref shape plan-kind 8 selects (2·max(a,b)).</summary>
    public const string TwiceLargerShape = "twicelarger";
    /// <summary>The reusable max(a,b) sub-shape that <see cref="TwiceLargerShape"/> references.</summary>
    public const string LargerShape = "larger";

    private readonly Dictionary<string, PlatonicGlider> _shapes =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly IPlatonicSpace _memory;

    public PlatonicShapeRegistry(IPlatonicSpace memory)
    {
        _memory = memory ?? throw new ArgumentNullException(nameof(memory));
        SeedPrimitiveShapes();
    }

    /// <summary>The executable shape library, passed to the interpreter so Ref blocks resolve recursively.</summary>
    public IReadOnlyDictionary<string, PlatonicGlider> Library => _shapes;

    public bool TryGet(string name, out PlatonicGlider glider) => _shapes.TryGetValue(name, out glider!);

    /// <summary>
    /// Register a shape: store its glider definition AND materialize it as a positioned Function element in
    /// the space (RelatedTo = the shape-elements it composes). Idempotent.
    /// </summary>
    public void Register(string name, GliderBlock root, params string[] references)
    {
        _shapes[name] = new PlatonicGlider(name, root);
        _memory.RegisterFunctionElement(name, references);
    }

    private void SeedPrimitiveShapes()
    {
        // max(a,b) — the difference-sign predicate selecting the bigger operand. Block vocabulary only.
        Register(LargerShape,
            new Branch(
                new Compare(CompareOp.Greater, new Operand(0), new Operand(1)),
                new Operand(0),
                new Operand(1)));

        // 2·max(a,b) — references the larger shape (composition-of-compositions), then scales on the
        // substrate. The ×2 is the SHAPE'S DEFINITION (what "twice-larger" means), not a memorized answer.
        Register(TwiceLargerShape,
            new Compute(GliderOp.Multiply, new GliderBlock[] { new Ref(LargerShape), new Const(2) }),
            LargerShape);
    }
}
