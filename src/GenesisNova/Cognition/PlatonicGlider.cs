using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using GenesisNova.Core;

namespace GenesisNova.Cognition;

/// <summary>
/// REUSABLE GLIDER BLOCKS — the small set of GENERAL, composable operations on the platonic substrate
/// from which gliders are built. Decomposed (2026-06-13) from the first monolithic "answer-template"
/// glider so we are NOT overfitting one problem: the SAME blocks recompose into many gliders (arithmetic
/// answers, retrieval answers, equivalence, format conversion). Blocks NEST — a block's inputs are other
/// blocks — and a glider is a single root block the interpreter evaluates against the query's operands.
///
/// The blocks (Conway analogy: these are the reusable patterns; a glider is a composition of them):
///  • <see cref="Operand"/>(i)        — read the i-th operand token from the input.
///  • <see cref="Literal"/>(chunk)    — emit a stored text chunk (a scaffold / known phrase).
///  • <see cref="Hop"/>(src, target)  — follow the strongest RELATION from src to a related concept of
///                                      the target kind. The ONE general retrieval primitive: serves
///                                      resolve (one→1), format (2→two), category (apple→fruit), facts
///                                      (france→paris) — only the target kind differs.
///  • <see cref="Compute"/>(op, args) — exact arithmetic over numeric args via the face homomorphism.
///  • <see cref="Fold"/>(op, from)    — REDUCE op across all operands from a slot onward (variadic compute;
///                                      the functional fold — sum/product of an arbitrary-length sequence).
///  • <see cref="Seq"/>(parts)        — concatenate the parts into the final text (numbers → digits).
///
/// This block set is the TARGET vocabulary the GRU will later learn to SEQUENCE — composing general ops,
/// never memorising a bespoke template. No NN here: the interpreter is the deterministic substrate physics.
/// </summary>
public abstract record GliderBlock;

/// <summary>Read the i-th operand token from the query input.</summary>
public sealed record Operand(int Slot) : GliderBlock;

/// <summary>Emit a stored, reusable text chunk (scaffold / known phrase).</summary>
public sealed record Literal(string Chunk) : GliderBlock;

/// <summary>Follow the strongest relation from <paramref name="Source"/> to a related concept of <paramref name="Target"/> kind.</summary>
public sealed record Hop(GliderBlock Source, HopTarget Target) : GliderBlock;

/// <summary>Exact arithmetic over numeric args via the face homomorphism.</summary>
public sealed record Compute(GliderOp Op, IReadOnlyList<GliderBlock> Args) : GliderBlock;

/// <summary>
/// HIGHER-ORDER (a FUNCTION over a sequence): REDUCE <paramref name="Op"/> across ALL operands from
/// <paramref name="FromSlot"/> onward — the variadic sibling of <see cref="Compute"/> (which is fixed-arity).
/// Folds "1 + 2 + 3 + …" / "2 × 3 × 4 × …" for ANY operand count, so the glider isn't tied to a 2-operand
/// shape. Element-native: the whole sequence composes via one R2 compose (the homomorphism is associative,
/// so a single multi-operand compose == repeated pairwise folds). For Subtract/Divide the tail operands are
/// composed via the complement (a − b − c = a + ¬b + ¬c), matching <see cref="ComposeArithmetic"/>.
/// </summary>
public sealed record Fold(GliderOp Op, int FromSlot = 0) : GliderBlock;

/// <summary>Concatenate child outputs into the final text.</summary>
public sealed record Seq(IReadOnlyList<GliderBlock> Parts) : GliderBlock;

/// <summary>A literal numeric constant — for parameterised computation (e.g. "double" = ×<see cref="Const"/>(2)).</summary>
public sealed record Const(double Value) : GliderBlock;

/// <summary>Predicate: compare two numeric blocks, yielding a boolean (1 = true, 0 = false). Enables
/// predicate answers ("is X greater than Y") and the conditions a <see cref="Branch"/> selects on.</summary>
public sealed record Compare(CompareOp Op, GliderBlock Left, GliderBlock Right) : GliderBlock;

/// <summary>Conditional / branch: if <paramref name="Condition"/> is truthy return <paramref name="WhenTrue"/>,
/// else <paramref name="WhenFalse"/>. Covers the format conditional and yes/no branching.</summary>
public sealed record Branch(GliderBlock Condition, GliderBlock WhenTrue, GliderBlock WhenFalse) : GliderBlock;

/// <summary>HIGHER-ORDER: invoke another named glider from the interpreter's library — a glider built
/// FROM gliders (the "glider gun"). The referenced glider runs on the SAME operands.</summary>
public sealed record Ref(string GliderName) : GliderBlock;

/// <summary>What kind of related concept a <see cref="Hop"/> retrieves.</summary>
public enum HopTarget { Number, Word, Any }

/// <summary>The arithmetic a <see cref="Compute"/> performs via the face homomorphism.</summary>
public enum GliderOp { Add, Subtract, Multiply, Divide }

/// <summary>The predicate a <see cref="Compare"/> evaluates.</summary>
public enum CompareOp { Greater, Less, Equal }

/// <summary>A named glider = a root block. The same blocks recompose into different gliders.</summary>
public sealed record PlatonicGlider(string Name, GliderBlock Root);

/// <summary>A value flowing between blocks — either a number or a piece of text.</summary>
public readonly record struct GliderValue(double? Number, string? Text)
{
    public static GliderValue Num(double n) => new(n, null);
    public static GliderValue Str(string s) => new(null, s);
    public bool IsNumber => Number.HasValue;
}

/// <summary>
/// Evaluates a glider (a composition of blocks) on the existing platonic physics — relation hops and the
/// numeric face homomorphism. Deterministic; no NN. The hand-composed glider is the target the GRU will
/// later learn to assemble FROM THESE BLOCKS.
/// </summary>
public sealed class PlatonicGliderInterpreter
{
    private readonly IPlatonicSpace _space;
    private readonly int _faceDim;
    private readonly IReadOnlyDictionary<string, PlatonicGlider> _library;

    public PlatonicGliderInterpreter(
        IPlatonicSpace space,
        IReadOnlyDictionary<string, PlatonicGlider>? library = null)
    {
        _space = space ?? throw new ArgumentNullException(nameof(space));
        _faceDim = space.FaceDimension;
        // No hardcoded capability library: the blocks are a COMPOSABLE VOCABULARY the GRU assembles, not a
        // table of premade gliders that lock tokens (compare/double/sum) to fixed meanings. A library can be
        // supplied (e.g. for Ref composition or tests); default is empty.
        _library = library ?? new Dictionary<string, PlatonicGlider>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>Evaluate the glider against the query's operand tokens, returning the assembled text.</summary>
    public string Execute(PlatonicGlider glider, IReadOnlyList<string> operandTokens)
        => AsText(Eval(glider.Root, operandTokens));

    private GliderValue Eval(GliderBlock block, IReadOnlyList<string> operands) => block switch
    {
        Operand o => GliderValue.Str(operands[o.Slot]),
        Literal l => GliderValue.Str(l.Chunk),
        Hop h => EvalHop(h, operands),
        Compute c => GliderValue.Num(EvalCompute(c, operands)),
        Fold f => GliderValue.Num(EvalFold(f, operands)),
        Const k => GliderValue.Num(k.Value),
        Compare cmp => GliderValue.Num(EvalCompare(cmp, operands) ? 1.0 : 0.0),
        Branch b => IsTruthy(Eval(b.Condition, operands)) ? Eval(b.WhenTrue, operands) : Eval(b.WhenFalse, operands),
        Ref r => EvalRef(r, operands),
        Seq s => GliderValue.Str(string.Join(' ', s.Parts.Select(p => AsText(Eval(p, operands))))),
        _ => throw new InvalidOperationException($"glider: unknown block {block.GetType().Name}"),
    };

    // ELEMENT-NATIVE (REFACTOR #2, 2026-06-13): the predicate is grounded in a platonic COMPOSITION —
    // the DIFFERENCE a + ¬b (subtract via the complement, reusing REFACTOR #1) — and the comparison
    // reads that composed element's SIGN. The substrate computes the difference; the predicate is its
    // sign. Identical results to the prior direct l/r comparison (the kept oracle).
    private bool EvalCompare(Compare cmp, IReadOnlyList<string> operands)
    {
        var l = AsNumber(Eval(cmp.Left, operands));
        var r = AsNumber(Eval(cmp.Right, operands));
        var values = new[] { l, r };
        var diff = DecodeArithmetic(GliderOp.Subtract, ComposeArithmetic(GliderOp.Subtract, values), values);
        return cmp.Op switch
        {
            CompareOp.Greater => diff > 0,
            CompareOp.Less => diff < 0,
            CompareOp.Equal => Math.Abs(diff) < 1e-9,
            _ => false,
        };
    }

    private GliderValue EvalRef(Ref reference, IReadOnlyList<string> operands)
    {
        if (!_library.TryGetValue(reference.GliderName, out var glider))
            throw new InvalidOperationException($"glider: no glider named '{reference.GliderName}' in the library");
        return Eval(glider.Root, operands);
    }

    private static bool IsTruthy(GliderValue v)
        => v.IsNumber
            ? Math.Abs(v.Number!.Value) > 1e-9
            : !string.IsNullOrEmpty(v.Text) && v.Text is not ("no" or "false" or "0");

    // Relational retrieval: from the source concept, follow the strongest relation to a neighbour of the
    // requested kind. General — resolve/format/category/fact are all this, differing only in Target.
    private GliderValue EvalHop(Hop hop, IReadOnlyList<string> operands)
    {
        var key = AsText(Eval(hop.Source, operands));
        foreach (var n in _space.GetNeighbors(key, PlatonicNeighborhoodType.Relational, maxNeighbors: 8, minConfidence: 0.35))
        {
            var isNum = TryParseNumber(n.Concept, out var v);
            switch (hop.Target)
            {
                case HopTarget.Number when isNum: return GliderValue.Num(v);
                case HopTarget.Word when !isNum: return GliderValue.Str(n.Concept);
                case HopTarget.Any: return isNum ? GliderValue.Num(v) : GliderValue.Str(n.Concept);
            }
        }
        throw new InvalidOperationException($"glider: no {hop.Target} relation from '{key}'");
    }

    // ELEMENT-NATIVE compute (REFACTOR #1, 2026-06-13): the arithmetic is now a first-class platonic
    // COMPOSITION element built by the substrate's own R2 rule (TickExecutor), decoded via the face
    // homomorphism — not an inline C# sum. The meta-layer direct sum is kept as the oracle ComputeDirect
    // and verified identical in tests. See PROJECT_GLIDER.md.
    private double EvalCompute(Compute compute, IReadOnlyList<string> operands)
    {
        var values = compute.Args.Select(a => AsNumber(Eval(a, operands))).ToArray();
        if (values.Length == 0)
            return 0.0;
        return DecodeArithmetic(compute.Op, ComposeArithmetic(compute.Op, values), values);
    }

    // VARIADIC FOLD: reduce the op across every operand from FromSlot onward. Reuses the SAME element-native
    // path as Compute (ComposeArithmetic handles N operands via one R2 compose), so a fold of 5 operands is
    // a single platonic composition, not 4 chained C# ops. Needs ≥1 operand; ≥2 to be a meaningful reduce.
    private double EvalFold(Fold fold, IReadOnlyList<string> operands)
    {
        var from = Math.Max(0, fold.FromSlot);
        if (from >= operands.Count)
            throw new InvalidOperationException($"glider: fold from slot {from} but only {operands.Count} operands");
        var values = new double[operands.Count - from];
        for (var i = from; i < operands.Count; i++)
            values[i - from] = AsNumber(GliderValue.Str(operands[i]));
        if (values.Length == 1)
            return Math.Round(values[0]);
        return DecodeArithmetic(fold.Op, ComposeArithmetic(fold.Op, values), values);
    }

    /// <summary>
    /// ELEMENT-NATIVE: build the arithmetic result as a platonic COMPOSITION element via the substrate's
    /// R2 compose rule (<see cref="TickExecutor"/>, <see cref="TickKind.Compose"/>). Subtract/divide
    /// compose with the COMPLEMENT (negated embedding, ¬b = −embed(b)) so a−b = a + ¬b in the poly face
    /// and a/b = a·¬b in the log face — G4 symmetry. The returned element is a genuine
    /// <see cref="PlatonicElement"/> (Kind=Composition, GenerationPath "R2:compose:…"), not a C# value.
    /// </summary>
    public PlatonicElement ComposeArithmetic(GliderOp op, IReadOnlyList<double> values)
    {
        var invertTail = op is GliderOp.Subtract or GliderOp.Divide; // compose with the complement
        var operandElements = new PlatonicElement[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            var face = PlatonicFaceComposer.GetFreshNumericEmbedding(values[i], _faceDim);
            if (invertTail && i > 0)
                for (var d = 0; d < face.Length; d++) face[d] = -face[d]; // ¬operand (complement)
            operandElements[i] = new PlatonicElement(
                Id: i, Kind: ElementKind.Object, Embedding: face,
                Symbol: values[i].ToString(CultureInfo.InvariantCulture), GeneratedAtTick: 0);
        }

        var primaryId = values.Count;
        var primary = new PlatonicElement(
            Id: primaryId, Kind: ElementKind.Object, Embedding: new double[_faceDim],
            Symbol: $"compose:{op}", GeneratedAtTick: 0,
            RelatedTo: ImmutableArray.CreateRange(Enumerable.Range(0, values.Count)));

        var state = new PlatonicState(
            ImmutableArray.CreateRange(operandElements.Append(primary)),
            EmbeddingDimension: _faceDim,
            NextId: primaryId + 1);

        var (_, created) = TickExecutor.ExecuteTick(new TickAction(TickKind.Compose, primaryId), state);
        return created.First(e => e.Kind == ElementKind.Composition);
    }

    /// <summary>Decode a composition element's embedding into a numeric value via the face homomorphism.</summary>
    private double DecodeArithmetic(GliderOp op, PlatonicElement composition, IReadOnlyList<double> values)
    {
        var additive = op is GliderOp.Add or GliderOp.Subtract;
        var preferFace = additive ? 1 : 2; // 1 = poly (add/sub), 2 = log (mul/div)
        var (value, _, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(composition.Embedding, _faceDim, preferFace);
        if (!additive && face == "log")
            value *= values.Aggregate(1.0, (s, v) => s * Math.Sign(v == 0 ? 1.0 : v));
        return Math.Round(value);
    }

    /// <summary>
    /// ORACLE: the original meta-layer direct-sum compute, kept verbatim so tests can confirm the
    /// element-native <see cref="ComposeArithmetic"/> path produces identical results.
    /// </summary>
    public double ComputeDirect(GliderOp op, IReadOnlyList<double> values)
    {
        if (values.Count == 0)
            return 0.0;
        var additive = op is GliderOp.Add or GliderOp.Subtract;
        var sign = op is GliderOp.Subtract or GliderOp.Divide ? -1.0 : 1.0;
        var acc = PlatonicFaceComposer.GetFreshNumericEmbedding(values[0], _faceDim);
        for (var k = 1; k < values.Count; k++)
        {
            var f = PlatonicFaceComposer.GetFreshNumericEmbedding(values[k], _faceDim);
            for (var i = 0; i < acc.Length; i++)
                acc[i] += sign * f[i];
        }
        var preferFace = additive ? 1 : 2;
        var (value, _, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(acc, _faceDim, preferFace);
        if (!additive && face == "log")
            value *= values.Aggregate(1.0, (s, v) => s * Math.Sign(v == 0 ? 1.0 : v));
        return Math.Round(value);
    }

    // Coerce a value to a number: numeric as-is; numeric text parsed; otherwise RESOLVE via a relation
    // hop to a number — so Compute works whether an operand arrives as "1" or "one" (the resolve block
    // folded in for convenience; an explicit Hop(_, Number) does the same thing).
    private double AsNumber(GliderValue v)
    {
        if (v.IsNumber)
            return v.Number!.Value;
        if (TryParseNumber(v.Text!, out var parsed))
            return parsed;
        foreach (var n in _space.GetNeighbors(v.Text!, PlatonicNeighborhoodType.Relational, maxNeighbors: 8, minConfidence: 0.35))
            if (TryParseNumber(n.Concept, out var resolved))
                return resolved;
        throw new InvalidOperationException($"glider: cannot coerce '{v.Text}' to a number");
    }

    private static string AsText(GliderValue v)
        => v.IsNumber ? ((long)Math.Round(v.Number!.Value)).ToString(CultureInfo.InvariantCulture) : v.Text!;

    private static bool TryParseNumber(string token, out double value)
        => double.TryParse(token, NumberStyles.Float | NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value);
}
