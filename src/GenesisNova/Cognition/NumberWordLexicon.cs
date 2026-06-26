using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// The LEARNED number-word lexicon — the de-hardcoded replacement for the hardcoded <c>NumberWordVocabulary</c> codec
/// inside the engine. It holds only the ARBITRARY, language-specific part: which WORD denotes which VALUE
/// ("five"→5, "forty"→40, "hundred"→100). Those atoms are LEARNED by OBSERVATION (the gym's digit↔word examples; see
/// <c>GenesisInferenceEngine.LearnNumberWord</c>), so the same mechanism learns "cinco"→5 if taught Spanish, and
/// ABSTAINS on a word it was never taught. The UNIVERSAL part — base-10 place value, "a power-of-ten scale multiplies
/// the current group" — is COMPUTED here, not stored, so it needs no language list.
///
/// Genesis-axiom check (this is a learned VALUE ANNOTATION per word, NOT a κ relation):
///   G1 observer — entries are written only on observation (training); G3 — a composed number ("forty seven"→47) is a
///   composite produced by observation; G6 — the map only grows (monotone, never deletes a learned atom);
///   numbers-never-edge / G2 — there is NO number↔number relation edge here, so arithmetic stays unpolluted.
/// </summary>
public sealed class NumberWordLexicon
{
    private readonly Dictionary<string, long> _atom = new(StringComparer.OrdinalIgnoreCase); // word -> value (LEARNED)

    private static string N(string t) => t.Trim().ToLowerInvariant();
    private static bool IsScale(long v) => v >= 100 && IsPowerOfTen(v);
    private static bool IsPowerOfTen(long v) { for (var p = 1L; p <= v; p *= 10) if (p == v) return true; return false; }

    /// <summary>Record a learned atom word↔value (monotone add — G6). Single tokens only; phrases compose at use.</summary>
    public void LearnAtom(string word, long value)
    {
        var w = N(word);
        if (w.Length == 0 || w.Contains(' ')) return;
        _atom[w] = value; // last write wins; the value of a number-word is stable, so re-teaching is idempotent
    }

    public bool KnowsAtom(string word) => _atom.ContainsKey(N(word));
    public int AtomCount => _atom.Count;

    /// <summary>LEARN from one digit→words example ("5 in words"→"five", "147 in words"→"one hundred forty seven").
    /// Single-word output → a direct atom. Multi-word output → learn the ONE still-unknown word COMPOSITIONALLY by
    /// solving the place-value equation against the already-known atoms (e.g. one/forty/seven known + value 147 ⇒ the
    /// residual "hundred" must be 100). Learns nothing (abstains) when more than one word is unknown.</summary>
    public void LearnFromDigitWords(long value, IReadOnlyList<string> outputWords)
    {
        var words = outputWords.Select(N).Where(w => w.Length > 0 && w.All(char.IsLetter)).ToList();
        if (words.Count == 0) return;
        if (words.Count == 1) { LearnAtom(words[0], value); return; }
        // Multi-word: solve for a single unknown by treating it as a scale multiplier in the standard place-value parse.
        var unknown = words.Where(w => !_atom.ContainsKey(w)).Distinct().ToList();
        if (unknown.Count != 1) return;                 // can only disambiguate one new word per example
        var u = unknown[0];
        // The unknown is almost always a SCALE word (hundred/thousand/…). Try scale candidates that make the parse exact.
        foreach (var cand in new long[] { 100, 1000, 1_000_000, 1_000_000_000 })
        {
            _atom[u] = cand;
            if (TryParse(words, out var got) && got == value) return; // learned the scale compositionally
        }
        _atom.Remove(u);                                 // no scale fit → don't guess (abstain)
    }

    /// <summary>WORD→VALUE: universal base-10 place-value composition over the LEARNED atoms. Non-atom tokens (framing
    /// words like "as"/"a"/"number") are ignored; returns false if NO number-word atom was present at all.</summary>
    public bool TryParse(IReadOnlyList<string> tokens, out long value)
    {
        long total = 0, current = 0; var any = false;
        foreach (var tok in tokens)
        {
            if (!_atom.TryGetValue(N(tok), out var v)) continue; // framing word → skip
            any = true;
            if (IsScale(v))
            {
                if (v >= 1000) { total += (current == 0 ? 1 : current) * v; current = 0; }
                else current = (current == 0 ? 1 : current) * v; // hundred multiplies the running group
            }
            else current += v;
        }
        value = total + current;
        return any;
    }

    /// <summary>VALUE→WORDS: universal base-10 decomposition, emitting the LEARNED word for each part. ABSTAINS (false)
    /// if any required atom (a digit 0-19, a ten, or the needed scale) has not been learned yet — honest over wrong.</summary>
    public bool TryToWords(long value, out string words)
    {
        words = string.Empty;
        if (value < 0) { if (!TryToWords(-value, out var pos)) return false; words = "negative " + pos; return true; }
        if (!TryCompose(value, out var w)) return false;
        words = w;
        return true;
    }

    private bool TryWord(long value, out string word) // a single learned atom for exactly this value
    {
        word = _atom.FirstOrDefault(kv => kv.Value == value).Key ?? string.Empty;
        return word.Length > 0;
    }

    private bool TryScaleWord(long scale, out string word) => TryWord(scale, out word);

    private bool TryCompose(long n, out string words)
    {
        words = string.Empty;
        if (n < 20) return TryWord(n, out words);                                  // 0-19 atoms
        if (n < 100)
        {
            if (!TryWord(n / 10 * 10, out var tens)) return false;                 // twenty/.../ninety
            if (n % 10 == 0) { words = tens; return true; }
            if (!TryWord(n % 10, out var ones)) return false;
            words = tens + " " + ones; return true;
        }
        // Find the largest learned SCALE that divides into n (hundred, thousand, …) — universal, no language list.
        var scale = LargestScaleAtMost(n);
        if (scale <= 1) return false;                                              // need a scale atom we haven't learned
        if (!TryScaleWord(scale, out var scaleWord)) return false;
        if (!TryCompose(n / scale, out var head)) return false;                    // "one"… / "twelve"… multiplier group
        words = head + " " + scaleWord;
        var rem = n % scale;
        if (rem != 0) { if (!TryCompose(rem, out var tail)) return false; words += " " + tail; }
        return true;
    }

    private long LargestScaleAtMost(long n)
    {
        long best = 1;
        foreach (var v in _atom.Values) if (IsScale(v) && v <= n && v > best) best = v;
        return best;
    }

    // Persistence (the learned lexicon is just these atoms).
    public IReadOnlyList<(string Word, long Value)> Export() => _atom.Select(kv => (kv.Key, kv.Value)).ToList();
    public void Import(IEnumerable<(string Word, long Value)> rows) { foreach (var (w, v) in rows) _atom[N(w)] = v; }
}
