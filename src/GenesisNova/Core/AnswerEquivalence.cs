using System;
using System.Globalization;

namespace GenesisNova.Core;

/// <summary>
/// FACE-AWARE RETURN GRADER (2026-06-14). Judges a predicted answer against the target by their
/// PLATONIC VALUE, not raw surface — so a digit and its number-word ("2" ≡ "two") both count as
/// correct, honouring the equivalence the interface learned (number-word-equiv) instead of discarding
/// it at the correctness gate.
///
/// This is a GROUND-TRUTH ORACLE (it uses <see cref="NumberWordVocabulary"/>, the same reference table
/// the creators use), NOT the model's own belief — so it cannot self-grade/drift, and the model must
/// still LEARN the equivalence to produce a correct answer. Non-numeric answers fall back to exact
/// (case-sensitive) match, so retrieval/predicate answers ("paris", "greater") are unaffected.
/// </summary>
public static class AnswerEquivalence
{
    /// <summary>True when predicted and expected are the same answer — exact, or the same numeric value
    /// across digit/word surfaces (e.g. "2" ≡ "two", "1 8" ≡ "18" ≡ "eighteen").</summary>
    public static bool Equivalent(string? predicted, string? expected)
    {
        var a = (predicted ?? string.Empty).Trim();
        var b = (expected ?? string.Empty).Trim();
        if (a.Length == 0 || b.Length == 0)
            return string.Equals(a, b, StringComparison.Ordinal);
        if (string.Equals(a, b, StringComparison.Ordinal))
            return true; // exact surface match (keeps text answers case-sensitive)
        if (TryNumericValue(a, out var va) && TryNumericValue(b, out var vb))
            return Math.Abs(va - vb) < 1e-9;
        return false;
    }

    /// <summary>Canonicalise an answer surface to its numeric value: a digit (digit-run spaces folded,
    /// e.g. "1 8" → 18) or a reference number-word ("eighteen" → 18). Returns false for non-numeric text.</summary>
    public static bool TryNumericValue(string token, out double value)
    {
        value = 0;
        if (string.IsNullOrWhiteSpace(token))
            return false;

        var compact = token.Replace(" ", string.Empty);
        if (double.TryParse(
                compact,
                NumberStyles.Float | NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture,
                out value))
            return true;

        if (NumberWordVocabulary.WordToValue.TryGetValue(token.Trim(), out var wordValue))
        {
            value = wordValue;
            return true;
        }

        return false;
    }
}
