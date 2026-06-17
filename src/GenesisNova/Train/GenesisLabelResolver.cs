using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Model;
using GenesisNova.Tokenization;

namespace GenesisNova.Train;

/// <summary>
/// Derives training SUPERVISION LABELS from an example's OWN structure — deterministic, read-only, no space
/// mutation. Extracted from <see cref="GenesisTrainer"/> (single-responsibility): the plan-head SHAPE label,
/// the query-head op/operand label, and the structural detectors (Seq scaffold, twice-larger). The trainer
/// delegates here. (ResolveRouteLabel stays in the trainer — it queries the platonic space, so it is not a
/// pure structural label.)
/// </summary>
public sealed class GenesisLabelResolver
{
    private readonly IGenesisTokenizer _tokenizer;

    public GenesisLabelResolver(IGenesisTokenizer tokenizer)
        => _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

    /// <summary>
    /// Derives the PLAN-head label (which block-composition SHAPE) from the example's OWN structure — no
    /// surface grammar: 2=predicate, 4=arithmetic→word, 5/6=fold-sum/product, 7=seq, 8=ref, 1=arithmetic
    /// (digit), 3=retrieval, 0=none. The GRU learns to predict this so the composer assembles the right glider.
    /// </summary>
    public int? ResolvePlanLabel(string? input, string? output)
    {
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return null;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var outT = output.Trim().ToLowerInvariant();
        var toks = input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var numericOperands = toks.Count(t => double.TryParse(t, ns, inv, out _));
        var outIsNumber = double.TryParse(outT, ns, inv, out _);

        if (outT is "greater" or "less" or "equal") return 2;        // predicate
        // Fold (variadic reduce): ≥3 numeric operands whose sum/product equals the numeric output.
        if (numericOperands >= 3 && double.TryParse(outT, ns, inv, out var foldOut))
        {
            var vals = toks.Where(t => double.TryParse(t, ns, inv, out _)).Select(t => double.Parse(t, ns, inv)).ToList();
            if (Math.Abs(vals.Sum() - foldOut) < 1e-6) return 5;                          // fold-sum
            if (Math.Abs(vals.Aggregate(1.0, (s, v) => s * v) - foldOut) < 1e-6) return 6; // fold-product
        }
        if (numericOperands >= 2 && GenesisNova.Core.NumberWordVocabulary.WordToValue.ContainsKey(outT))
            return 4;                                                // arithmetic → word-formatted result

        // SEQ (plan-kind 7) — a Concatenate-Composition: a scaffold chunk bound to a computed value (the
        // output is "<scaffold words> <sum-of-operands>"). Derived from the output's own structure.
        if (numericOperands >= 2 && TrySeqSegments(toks, outT, out _))
            return 7;

        // EXPRESSION-CHAIN (plan-kind 8) — a MULTI-OPERATOR expression evaluated with precedence by CHAINING
        // compute-elements (e.g. "2 x 7 + 3" → 17). Pure same-op runs are already caught as fold (5/6) above,
        // so this owns the MIXED-operator cases. The op of each operator is classified from CONTEXT by the op
        // head at inference (never a symbol→op map); this oracle interprets the standard arithmetic symbols
        // ONLY to derive the label's truth. Checked before the single-op arithmetic fallback.
        if (IsExpressionChain(toks, outT))
            return 8;

        if (numericOperands >= 2 && outIsNumber) return 1;           // arithmetic (digit)
        if (numericOperands == 0 && toks.Length >= 1 && !outIsNumber) return 3; // retrieval
        return 0;                                                    // none/abstain
    }

    /// <summary>
    /// SEQ scaffold for an (input, output) pair — the scaffold words a Seq glider's Literal emits — if the
    /// output has the Seq structure. Used by the trainer to MINE the scaffold into the chunk-element store.
    /// </summary>
    public bool TryGetSeqScaffold(string? input, string? output, out string scaffold)
    {
        scaffold = string.Empty;
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return false;
        var toks = input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var outT = output.Trim().ToLowerInvariant();
        return TrySeqSegments(toks, outT, out scaffold);
    }

    /// <summary>
    /// SEQ structure detector: is the output a scaffold chunk (one+ non-numeric words) followed by a single
    /// number equal to the SUM of the input's numeric operands? Returns the scaffold words on a match.
    /// </summary>
    public static bool TrySeqSegments(IReadOnlyList<string> inputToks, string output, out string scaffold)
    {
        scaffold = string.Empty;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var outToks = output.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (outToks.Length < 2)
            return false; // need scaffold + a computed segment
        var last = outToks[^1];
        if (!double.TryParse(last, ns, inv, out var tail))
            return false; // final segment must be the computed number
        // The leading segment(s) must be non-numeric scaffold words.
        for (var k = 0; k < outToks.Length - 1; k++)
            if (double.TryParse(outToks[k], ns, inv, out _))
                return false;
        var operands = inputToks.Where(t => double.TryParse(t, ns, inv, out _))
                                .Select(t => double.Parse(t, ns, inv)).ToList();
        if (operands.Count < 2)
            return false;
        if (Math.Abs(operands.Sum() - tail) > 1e-6)
            return false; // computed segment must equal the operands' sum (the Compute(Add) part)
        scaffold = string.Join(' ', outToks.Take(outToks.Length - 1));
        return true;
    }

    /// <summary>
    /// EXPRESSION-CHAIN structure detector (plan-kind 8 ORACLE): the input is a MULTI-OPERATOR arithmetic
    /// expression (≥2 operators over ≥3 operands) whose standard precedence-evaluation (× ÷ before + −, each
    /// pass left-to-right) equals the numeric output. This derives the supervised label's TRUTH from the
    /// example's own structure — it interprets the standard arithmetic operator symbols (+ - x * /) ONLY for
    /// grading; the MODEL never sees this map and must classify each operator from CONTEXT (the op head) at
    /// inference. Pure single-operator runs are owned by fold (5/6) / digit-arithmetic (1), so they are
    /// excluded by the ≥2-operator requirement (and fold is checked first in <see cref="ResolvePlanLabel"/>).
    /// </summary>
    public static bool IsExpressionChain(IReadOnlyList<string> inputToks, string output)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        if (!double.TryParse(output, ns, inv, out var target))
            return false;

        var operands = new List<double>();
        var ops = new List<char>();
        var expectOperand = true;
        foreach (var tok in inputToks)
        {
            if (expectOperand)
            {
                if (!double.TryParse(tok, ns, inv, out var v))
                    return false; // a non-operand where an operand was expected → not a clean expression
                operands.Add(v);
                expectOperand = false;
            }
            else
            {
                var op = tok switch { "+" => '+', "-" => '-', "x" or "*" => '*', "/" => '/', _ => '\0' };
                if (op == '\0')
                    return false; // not a standard arithmetic operator symbol
                ops.Add(op);
                expectOperand = true;
            }
        }
        if (!expectOperand)
            return false; // trailing operator → malformed
        if (ops.Count < 2 || operands.Count != ops.Count + 1)
            return false; // MULTI-operator only (single binary / fold handled elsewhere)

        // Precedence evaluation (the oracle's ground truth).
        var vals = operands.ToList();
        var oo = ops.ToList();
        for (var i = 0; i < oo.Count;)
        {
            if (oo[i] is '*' or '/')
            {
                if (oo[i] == '/' && Math.Abs(vals[i + 1]) < 1e-12) return false;
                vals[i] = oo[i] == '*' ? vals[i] * vals[i + 1] : vals[i] / vals[i + 1];
                vals.RemoveAt(i + 1);
                oo.RemoveAt(i);
            }
            else i++;
        }
        var acc = vals[0];
        for (var i = 0; i < oo.Count; i++)
            acc = oo[i] == '+' ? acc + vals[i + 1] : acc - vals[i + 1];
        return Math.Abs(acc - target) < 1e-6;
    }

    /// <summary>
    /// Derives the GRU query-head supervision (op id + operand mask) from the example's numeric structure:
    /// exactly two maximal signed digit runs (L, R) and a numeric output O where exactly one face op
    /// (add/sub/mul/div) satisfies op(L,R)==O. Ambiguous/non-conforming → null (unsupervised).
    /// </summary>
    public GenesisQueryLabel? ResolveQueryLabel(IReadOnlyList<int> inputTokens, string? output)
    {
        if (inputTokens.Count == 0 || string.IsNullOrWhiteSpace(output))
            return null;
        if (!double.TryParse(output.Trim(), System.Globalization.NumberStyles.AllowLeadingSign
                | System.Globalization.NumberStyles.AllowDecimalPoint,
                System.Globalization.CultureInfo.InvariantCulture, out var target))
            return null;

        var vocab = _tokenizer.Vocabulary;
        bool IsDigitToken(int id) =>
            id >= 0 && id < vocab.Count && vocab[id].Length == 1 && vocab[id][0] is >= '0' and <= '9';
        bool IsMinusToken(int id) => id >= 0 && id < vocab.Count && vocab[id] == "-";

        var runs = new List<(int Start, int End, double Value)>();
        var i = 0;
        while (i < inputTokens.Count)
        {
            var negative = false;
            var start = i;
            if (IsMinusToken(inputTokens[i])
                && i + 1 < inputTokens.Count && IsDigitToken(inputTokens[i + 1])
                && (i == 0 || !IsDigitToken(inputTokens[i - 1])))
            {
                negative = true;
                i++;
            }
            if (i >= inputTokens.Count || !IsDigitToken(inputTokens[i])) { i = start + 1; continue; }
            var value = 0.0;
            while (i < inputTokens.Count && IsDigitToken(inputTokens[i]))
            {
                value = (value * 10.0) + (vocab[inputTokens[i]][0] - '0');
                i++;
            }
            runs.Add((start, i, negative ? -value : value));
        }
        if (runs.Count != 2)
            return null;

        var (l, r) = (runs[0].Value, runs[1].Value);
        var matches = new List<int>();
        if (Math.Abs((l + r) - target) < 1e-9) matches.Add(1);
        if (Math.Abs((l - r) - target) < 1e-9) matches.Add(2);
        if (Math.Abs((l * r) - target) < 1e-9) matches.Add(3);
        if (Math.Abs(r) > 1e-12 && Math.Abs((l / r) - target) < 1e-9) matches.Add(4);
        if (matches.Count != 1)
            return null; // ambiguous (e.g. 2+2 == 2*2) or no face op fits — leave unsupervised.

        var mask = new bool[inputTokens.Count];
        foreach (var run in runs)
            for (var t = run.Start; t < run.End; t++)
                mask[t] = true;

        return new GenesisQueryLabel(matches[0], mask);
    }
}
