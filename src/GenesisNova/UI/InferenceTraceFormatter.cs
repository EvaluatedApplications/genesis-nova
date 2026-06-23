namespace GenesisNova.UI;

/// <summary>
/// PURE decision-path → English formatting for the REPL inference trace. Extracted from
/// <see cref="MainWindow"/> (SOLID/SRP): these methods map a <see cref="GenesisNova.Infer.GenerationResult"/>
/// / decision path to display text with NO WinForms dependency, so the formatting can be tested and reasoned
/// about independently of the window. MainWindow's control-touching code (box.Text, BeginInvoke, _runtime)
/// stays in MainWindow and calls these for the strings.
/// </summary>
internal static class InferenceTraceFormatter
{
    // Classify HOW the answer was produced — calculated (homomorphism/transform), remembered (stored relation
    // edge), composed (glider shape), or generated (neural decoder). Drives the trace's plain-English mechanism.
    public static (string Label, string Why) MechanismOf(GenesisNova.Infer.GenerationResult r)
    {
        if (r.UsedNeuralFallback) return ("GENERATED", "the neural decoder produced it token-by-token — no platonic compute or retrieval");
        var d = (r.DecisionPath ?? string.Empty).ToLowerInvariant();
        // Order matters: 'expression-chain' before 'chain'; specific routes before generic substrings.
        if (d.Contains("expression-chain"))
            return ("CALCULATED", "chained several compute-elements (a multi-operator expression), each operator classified from context");
        if (d.Contains("gru-query") || d.Contains("arithmetic"))
            return ("CALCULATED", "computed via the numeric homomorphism (poly/log) — generalizes, not a stored fact");
        if (d.Contains("learned-op") || d.Contains("learned-function") || r.RoutedTransform is not null)
            return ("CALCULATED", "applied a learned function as a transform vector — computed by composition, not recalled");
        if (d.Contains("glider") || d.Contains("plan") || d.Contains("fold") || d.Contains("predicate") || d.Contains("seq"))
            return ("COMPOSED", "assembled a glider shape and ran it on the substrate");
        if (d.Contains("geometric"))
            return ("REMEMBERED", "found the nearest concept by POSITION in the semantic face (geometric content-addressing)");
        if (d.Contains("relation-edge"))
            return ("REMEMBERED", "followed a learned relation edge (a stored cue↔answer association)");
        if (d.Contains("concept") || d.Contains("chain") || d.Contains("retriev"))
            return ("REMEMBERED", "walked the relation graph (multi-hop) from your input concepts");
        return ("PLATONIC", "answered via the platonic substrate");
    }

    // Human description of the glider SHAPE encoded in a 'platonic-glider-plan:<shape>' decision path.
    public static string ShapeOf(string decisionPath)
    {
        var dp = decisionPath ?? string.Empty;
        var i = dp.IndexOf(':');
        var s = (i >= 0 && i + 1 < dp.Length ? dp[(i + 1)..] : dp).ToLowerInvariant();
        return s switch
        {
            "predicate" => "predicate — Compare→Branch (the difference's sign → greater/less/equal)",
            "fold-sum" => "fold-sum — one N-way R2 compose over all operands (poly-sum)",
            "fold-product" => "fold-product — one N-way R2 compose over all operands (log-sum)",
            "seq" => "seq — a mined scaffold chunk ∘ Fold(Add) (Concatenate-composition)",
            "arith-word" => "arith→word — Compute the value, then Hop the digit to its number-word",
            _ => "a glider shape"
        };
    }

    // Best-effort op/operand/face extraction for the trace's "how it computed it" explanation.
    public static (string Op, string Face, long[] Operands)? ParseArith(string input)
    {
        var t = (input ?? string.Empty).ToLowerInvariant();
        string? op = t.Contains('+') || t.Contains("plus") || t.Contains("add") || t.Contains("sum") ? "add"
            : t.Contains('-') || t.Contains("minus") || t.Contains("subtract") || t.Contains("difference") ? "sub"
            : t.Contains('*') || t.Contains(" x ") || t.Contains("times") || t.Contains("multipl") || t.Contains("product") ? "mul"
            : t.Contains('/') || t.Contains("divide") || t.Contains("quotient") ? "div"
            : null;
        if (op is null) return null;
        var nums = System.Text.RegularExpressions.Regex.Matches(input ?? string.Empty, @"-?\d+")
            .Select(m => long.Parse(m.Value)).ToArray();
        if (nums.Length == 0) return null;
        return (op, op is "add" or "sub" ? "poly" : "log", nums);
    }

    public static string DescribeConfidence(double confidence)
    {
        if (confidence >= 0.85)
            return "High confidence: strong structural match.";
        if (confidence >= 0.60)
            return "Moderate confidence: useful structure, some ambiguity.";
        if (confidence > 0.0)
            return "Low confidence: weak structural support.";
        return "No platonic confidence signal (neural-only path).";
    }
}
