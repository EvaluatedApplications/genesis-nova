namespace GenesisNova.Tokenization;

/// <summary>
/// A learned casing model for detokenization. The tokenizer case-folds tokens so the semantic unit is shared
/// ("Find" / "find" / "FIND" are one concept, op-tokens still match) — but folding throws away the surface
/// casing, so "WhitespaceGenesisTokenizer" could only ever decode as "whitespacegenesistokenizer". This learns
/// the surface form from the corpus: every <see cref="Observe"/> during encoding tallies the original spelling
/// for a folded token, and <see cref="Restore"/> returns the majority surface at decode time.
///
/// Unigram (per folded token) — casing of an identifier is intrinsic to the token, not its neighbours, so a
/// majority vote over observed spellings recovers "WhitespaceGenesisTokenizer", "POSITIONED", "GRU" exactly.
/// Nothing is hardcoded; the spellings are whatever the corpus showed. Unseen tokens fall back to the folded
/// form. (Limitation: one folded token with two genuine casings — a lowercase verb vs a PascalCase symbol —
/// resolves to whichever was seen more; rare in this identifier/key domain.)
/// </summary>
public sealed class LearnedCasingModel
{
    // foldedToken -> (surfaceForm -> count)
    private readonly Dictionary<string, Dictionary<string, int>> _forms = new(StringComparer.Ordinal);

    /// <summary>Tally a surface spelling. Records every form (including all-lowercase) so the majority vote is
    /// honest — a token usually seen lowercase must stay lowercase.</summary>
    public void Observe(string surface)
    {
        if (string.IsNullOrEmpty(surface))
            return;
        var folded = surface.ToLowerInvariant();
        if (!_forms.TryGetValue(folded, out var counts))
        {
            counts = new Dictionary<string, int>(StringComparer.Ordinal);
            _forms[folded] = counts;
        }
        counts.TryGetValue(surface, out var n);
        counts[surface] = n + 1;
    }

    /// <summary>Return the majority surface spelling for a folded token, or the folded token itself if unseen.</summary>
    public string Restore(string folded)
    {
        if (!_forms.TryGetValue(folded, out var counts) || counts.Count == 0)
            return folded;
        var best = folded;
        var bestCount = -1;
        foreach (var kv in counts)
        {
            // Tie-break toward the form that differs from the bare fold (prefer the cased spelling on a tie).
            if (kv.Value > bestCount || (kv.Value == bestCount && !string.Equals(kv.Key, folded, StringComparison.Ordinal)))
            {
                best = kv.Key;
                bestCount = kv.Value;
            }
        }
        return best;
    }

    public CasingModelSnapshot Export() => new(
        _forms.SelectMany(t => t.Value.Select(f => new CasingStat(t.Key, f.Key, f.Value))).ToArray());

    public void Import(CasingModelSnapshot? snapshot)
    {
        _forms.Clear();
        if (snapshot is null)
            return;
        foreach (var s in snapshot.Forms)
        {
            if (!_forms.TryGetValue(s.Folded, out var counts))
            {
                counts = new Dictionary<string, int>(StringComparer.Ordinal);
                _forms[s.Folded] = counts;
            }
            counts[s.Surface] = s.Count;
        }
    }
}

public sealed record CasingModelSnapshot(CasingStat[] Forms);

public sealed record CasingStat(string Folded, string Surface, int Count);
