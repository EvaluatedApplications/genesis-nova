using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Core;

namespace GenesisNova.Train;

/// <summary>
/// CONTEXTUAL, LLM-tolerant grading. The answer only needs to OCCUR (value-aware: 20 ≡ twenty), surrounded by
/// ANY filler / personality — "the answer is twenty, am I right?" scores full. The penalty is OVER-GENERATION OF
/// THE ANSWER'S OWN TYPE, nothing else:
///   • Numeric answer → extra NUMBERS are what's penalized. A competing/wrong number ("20 30 40") is heavy
///     (hedging); a duplicate of the right one ("20 20") is mild. Words/punctuation are free.
///   • One-to-many → <paramref name="requiredDepth"/> distinct valid answers are expected (coverage).
///   • require-platonic → a neural-fallback answer scores 0 (capability-mastery).
/// Non-numeric domains fall back to value-aware occurrence (+ a light runaway-length tax), since "competing"
/// can't be identified without an answer vocabulary.
/// </summary>
public static class GenesisGrader
{
    // word → value (shared number-word vocab) so "twenty" grades as 20 and counts as a competing number.
    private static readonly Dictionary<string, long> WordValue =
        NumberWordVocabulary.Entries.ToDictionary(e => e.Word.ToLowerInvariant(), e => (long)e.Value);

    public static double Quality(string output, IReadOnlyList<string> allowed, int requiredDepth, bool usedNeuralFallback, bool requirePlatonic,
        IReadOnlyList<string>? answerVocabulary = null)
    {
        if (requirePlatonic && usedNeuralFallback) return 0.0;
        if (allowed is null || allowed.Count == 0 || string.IsNullOrWhiteSpace(output)) return 0.0;

        var expectedValues = allowed.Select(TryValue).Where(v => v.HasValue).Select(v => v!.Value).ToHashSet();
        var numericDomain = expectedValues.Count > 0 && allowed.All(a => TryValue(a).HasValue);
        return numericDomain ? NumericQuality(output, expectedValues, requiredDepth)
                             : SetQuality(output, allowed, requiredDepth, answerVocabulary);
    }

    // Numeric domain: the answer is a number → grade by which NUMBERS appear, ignore everything else.
    private static double NumericQuality(string output, HashSet<long> expected, int requiredDepth)
    {
        var nums = NumbersIn(output).ToList();              // every numeric token's value (digits + number-words)
        if (nums.Count == 0) return 0.0;
        var correct = nums.Where(expected.Contains).ToList();
        if (correct.Count == 0) return 0.0;                 // the answer never occurred
        var distinctCorrect = correct.Distinct().Count();
        var required = Math.Max(1, Math.Min(requiredDepth, expected.Count));
        var coverage = Math.Min(distinctCorrect, required) / (double)required;

        var wrong = nums.Count(n => !expected.Contains(n)); // competing numbers — hedging, heavy penalty
        var dupes = correct.Count - distinctCorrect;        // extra copies of a right number — mild over-gen
        var cleanliness = Math.Clamp(1.0 - 0.5 * wrong - 0.15 * dupes, 0.0, 1.0);
        return coverage * cleanliness;
    }

    // Non-numeric: value-aware OCCURRENCE of any valid answer (filler/personality free). When an answer
    // VOCABULARY is supplied, over-generation is type-aware like the numeric path — a COMPETING vocab word that
    // isn't a valid answer ("animal" for a fruit) is penalized, while non-vocab filler stays free. Without a
    // vocabulary we can't tell competing from filler, so we credit presence + a light runaway-length tax.
    private static double SetQuality(string output, IReadOnlyList<string> allowed, int requiredDepth, IReadOnlyList<string>? vocab)
    {
        var outCanon = Canon(output);
        bool Present(string a)
        {
            var c = Canon(a);
            if (c.Length > 0 && outCanon.Contains(c)) return true;
            try { return AnswerEquivalence.Equivalent(output, a); } catch { return false; }
        }
        var matched = allowed.Where(Present).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        if (matched.Count == 0) return 0.0;
        var required = Math.Max(1, Math.Min(requiredDepth, allowed.Count));
        var coverage = Math.Min(matched.Count, required) / (double)required;

        double cleanliness;
        if (vocab is { Count: > 0 })
        {
            var allowedSet = new HashSet<string>(allowed, StringComparer.OrdinalIgnoreCase);
            var competing = vocab.Count(v => !allowedSet.Contains(v) && Canon(v).Length > 0 && outCanon.Contains(Canon(v)));
            cleanliness = Math.Clamp(1.0 - 0.5 * competing, 0.2, 1.0); // wrong same-type answers = hedging
        }
        else
        {
            var excess = Math.Max(0, Tokens(output) - matched.Sum(Tokens) - 6); // ~6 filler tokens free; runaway taxed
            cleanliness = Math.Clamp(1.0 - 0.1 * excess, 0.5, 1.0);
        }
        return coverage * cleanliness;
    }

    private static long? TryValue(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        var t = s.Trim().TrimEnd('.', ',', '!', '?', ';', ':');
        if (long.TryParse(t, out var n)) return n;
        return WordValue.TryGetValue(t.ToLowerInvariant(), out var v) ? v : null;
    }

    private static IEnumerable<long> NumbersIn(string text)
    {
        foreach (var tok in text.Split(new[] { ' ', ',', '.', '!', '?', ';', ':', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries))
        {
            var v = TryValue(tok);
            if (v.HasValue) yield return v.Value;
        }
    }

    private static int Tokens(string s) => (s ?? string.Empty).Split(new[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries).Length;
    private static string Canon(string s) => new((s ?? string.Empty).ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());
}
