namespace GenesisNova.Tokenization;

/// <summary>
/// A learned detokenization spacing model. Tokenization is lossy about whitespace — "apple-orange" and
/// "apple - orange" both tokenize to ["apple","-","orange"] — so reconstructing text needs a model of which
/// adjacent token pairs were joined (no space between them) in the source. This learns that from the corpus:
/// every <see cref="Observe"/> during encoding records whether a pair was joined or separated, and
/// <see cref="TryDecide"/> returns the majority verdict at decode time.
///
/// Two levels, most-specific-wins: exact token-pair statistics, then a class-level backoff (word / digit /
/// each punctuation mark as its own class) so unseen pairs still generalize from how their *kinds* spaced.
/// When the model has seen neither, it reports no evidence and the caller falls back to a structural prior.
/// Nothing about the spacing is hardcoded — it is observed from whatever text flows through the tokenizer.
/// </summary>
public sealed class LearnedSpacingModel
{
    // (prev, curr) -> [joinedCount, separatedCount]. Tuple keys use ordinal string equality by default.
    private readonly Dictionary<(string Prev, string Curr), int[]> _pair = new();
    private readonly Dictionary<(string Prev, string Curr), int[]> _class = new();

    /// <summary>Record that <paramref name="curr"/> followed <paramref name="prev"/>, joined (no whitespace
    /// between them in the source) or separated.</summary>
    public void Observe(string prev, string curr, bool joined)
    {
        Bump(_pair, (prev, curr), joined);
        Bump(_class, (ClassOf(prev), ClassOf(curr)), joined);
    }

    /// <summary>If the model has evidence for this pair, set <paramref name="join"/> to the learned decision
    /// (true = concatenate, no space) and return true. Exact pair evidence wins; otherwise the class backoff;
    /// otherwise returns false (no evidence — let the caller apply its prior).</summary>
    public bool TryDecide(string prev, string curr, out bool join)
    {
        if (_pair.TryGetValue((prev, curr), out var p) && p[0] + p[1] > 0)
        {
            join = p[0] > p[1];
            return true;
        }

        if (_class.TryGetValue((ClassOf(prev), ClassOf(curr)), out var c) && c[0] + c[1] > 0)
        {
            join = c[0] > c[1];
            return true;
        }

        join = false;
        return false;
    }

    public SpacingModelSnapshot Export() => new(
        _pair.Select(kv => new SpacingStat(kv.Key.Prev, kv.Key.Curr, kv.Value[0], kv.Value[1])).ToArray(),
        _class.Select(kv => new SpacingStat(kv.Key.Prev, kv.Key.Curr, kv.Value[0], kv.Value[1])).ToArray());

    public void Import(SpacingModelSnapshot? snapshot)
    {
        _pair.Clear();
        _class.Clear();
        if (snapshot is null)
            return;
        foreach (var s in snapshot.Pairs)
            _pair[(s.Prev, s.Curr)] = [s.Joined, s.Separated];
        foreach (var s in snapshot.Classes)
            _class[(s.Prev, s.Curr)] = [s.Joined, s.Separated];
    }

    private static void Bump(Dictionary<(string, string), int[]> table, (string, string) key, bool joined)
    {
        if (!table.TryGetValue(key, out var counts))
        {
            counts = new int[2];
            table[key] = counts;
        }
        counts[joined ? 0 : 1]++;
    }

    // A token's spacing class: each punctuation/symbol mark is its own class (so "-" learns differently from
    // ","), all-digit and all-letter runs collapse to one class each, everything else is "other".
    private static string ClassOf(string token)
    {
        if (token.Length == 1 && (char.IsPunctuation(token[0]) || char.IsSymbol(token[0])))
            return token;
        if (token.Length > 0 && token.All(char.IsDigit))
            return "#D";
        if (token.Length > 0 && token.All(char.IsLetter))
            return "#W";
        return "#O";
    }
}

public sealed record SpacingModelSnapshot(SpacingStat[] Pairs, SpacingStat[] Classes);

public sealed record SpacingStat(string Prev, string Curr, int Joined, int Separated);
