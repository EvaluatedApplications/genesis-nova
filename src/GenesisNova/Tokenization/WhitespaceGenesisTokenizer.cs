namespace GenesisNova.Tokenization;

public sealed class WhitespaceGenesisTokenizer : IGenesisTokenizer
{
    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.Ordinal);
    private readonly List<string> _idToToken = [];

    public WhitespaceGenesisTokenizer()
    {
        AddToken("<pad>");
        AddToken("<bos>");
        AddToken("<eos>");
        AddDigitTokens();
    }

    public int PadTokenId => 0;
    public int BosTokenId => 1;
    public int EosTokenId => 2;
    public int VocabularySize => _idToToken.Count;
    public IReadOnlyList<string> Vocabulary => _idToToken;

    /// <summary>The learned detokenization spacing model. Trained from every <see cref="Encode"/> call (which
    /// sees the original whitespace) and consulted by <see cref="Decode"/>; persisted with the checkpoint.</summary>
    public LearnedSpacingModel SpacingModel { get; } = new();

    /// <summary>The learned casing model (folded token -> surface spelling). Trained from every
    /// <see cref="Encode"/> call (which sees the original case) and used by <see cref="Decode"/> to restore the
    /// true casing the case-folded vocab would otherwise lose; persisted with the checkpoint.</summary>
    public LearnedCasingModel CasingModel { get; } = new();

    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        // Pre-size to a rough token estimate (~1 token per 4 chars) + space for BOS/EOS to avoid list growth reallocs.
        var output = new List<int>((text?.Length ?? 0) / 4 + 2);
        if (addBos)
            output.Add(BosTokenId);

        if (!string.IsNullOrWhiteSpace(text))
        {
            // Tokenize preserves surface case; the vocab/embedding key is the CASE-FOLDED token so a concept is
            // shared across casings. The tokenizer is the only place that still sees the source case+whitespace,
            // so it's where the casing model (folded -> surface) and the spacing model (pair -> joined) learn.
            var parts = Tokenize(text);
            string? prevFolded = null;
            for (var i = 0; i < parts.Count; i++)
            {
                var surface = parts[i].Token;
                var folded = surface.ToLowerInvariant();
                CasingModel.Observe(surface);
                if (prevFolded is not null)
                    SpacingModel.Observe(prevFolded, folded, joined: !parts[i].SpaceBefore);
                output.Add(AddToken(folded));
                prevFolded = folded;
            }
        }

        if (addEos)
            output.Add(EosTokenId);

        return output.ToArray();
    }

    // Cached interned strings for single-char tokens (ASCII range) so per-digit/per-punct chars don't each
    // allocate a fresh string via ch.ToString(). char.ToString() always allocates; this reuses one instance
    // per character. Identical string content, fewer allocations.
    private static readonly string[] SingleCharStrings = BuildSingleCharStrings();

    private static string[] BuildSingleCharStrings()
    {
        var table = new string[128];
        for (var c = 0; c < table.Length; c++)
            table[c] = ((char)c).ToString();
        return table;
    }

    private static string CharToken(char ch)
        => ch < 128 ? SingleCharStrings[ch] : ch.ToString();

    private static IReadOnlyList<(string Token, bool SpaceBefore)> Tokenize(string text)
    {
        var tokens = new List<(string, bool)>(text.Length / 4 + 1);
        var normalized = text; // keep original case; folding happens per-token in Encode
        var buffer = new System.Text.StringBuilder();
        var sawSpace = false;        // whitespace seen since the last char added to a token
        var bufferSpaceBefore = false; // whether the current buffer's first char was preceded by whitespace

        for (var i = 0; i < normalized.Length; i++)
        {
            var ch = normalized[i];
            if (char.IsWhiteSpace(ch))
            {
                FlushBuffer();
                sawSpace = true;
                continue;
            }

            if (IsAsciiDigit(ch))
            {
                FlushBuffer();
                tokens.Add((CharToken(ch), sawSpace));
                sawSpace = false;
                continue;
            }

            if (char.IsLetterOrDigit(ch))
            {
                if (buffer.Length == 0)
                    bufferSpaceBefore = sawSpace;
                buffer.Append(ch);
                sawSpace = false;
                continue;
            }

            FlushBuffer();
            tokens.Add((CharToken(ch), sawSpace));
            sawSpace = false;
        }

        FlushBuffer();
        return tokens;

        void FlushBuffer()
        {
            if (buffer.Length == 0)
                return;

            tokens.Add((buffer.ToString(), bufferSpaceBefore));
            buffer.Clear();
        }
    }

    public string Decode(IReadOnlyList<int> tokens)
    {
        var words = new List<string>(tokens.Count);
        foreach (var token in tokens)
        {
            if (token < 0 || token >= _idToToken.Count)
                continue;

            if (token == PadTokenId || token == BosTokenId || token == EosTokenId)
                continue;

            words.Add(_idToToken[token]);
        }

        if (words.Count == 0)
            return string.Empty;

        // words are CASE-FOLDED vocab tokens; spacing is decided on the folded forms, casing restored for display.
        var output = new System.Text.StringBuilder(CasingModel.Restore(words[0]));
        for (var i = 1; i < words.Count; i++)
        {
            var current = words[i];
            var previous = words[i - 1];
            // Learned spacing wins where it has evidence; the structural heuristic is the cold-start prior.
            var join = SpacingModel.TryDecide(previous, current, out var learned)
                ? learned
                : HeuristicJoin(previous, current);
            var surface = CasingModel.Restore(current);
            if (join)
                output.Append(surface);
            else
                output.Append(' ').Append(surface);
        }

        return output.ToString();
    }

    public void ReplaceVocabulary(IReadOnlyList<string> tokens)
    {
        _tokenToId.Clear();
        _idToToken.Clear();

        foreach (var token in tokens)
            AddToken(token);

        AddDigitTokens();
    }

    private int AddToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out var existing))
            return existing;

        var id = _idToToken.Count;
        _tokenToId[token] = id;
        _idToToken.Add(token);
        return id;
    }

    private void AddDigitTokens()
    {
        for (var digit = '0'; digit <= '9'; digit++)
            AddToken(digit.ToString());
    }

    // Structural prior used only when the learned spacing model has no evidence for a pair.
    private static bool HeuristicJoin(string previous, string current)
    {
        if (IsDigitToken(previous) && IsDigitToken(current))
            return true;

        // Punctuation/operator glues to the token on its LEFT ("apple" + "-" => "apple-").
        if (current.Length == 1 && IsOperatorOrPunctuation(current[0]))
            return true;

        // ...and a word/number glues BACK onto a trailing connector, so hyphenated (and
        // snake_/path/decimal) tokens round-trip instead of splitting ("apple-" + "orange"
        // => "apple-orange", "5-" + "3" => "5-3", "3." + "14" => "3.14"). Without this the
        // left-only rule yields "apple- orange".
        if (previous.Length > 0 && IsConnector(previous[^1]) &&
            current.Length > 0 && char.IsLetterOrDigit(current[0]))
            return true;

        return false;
    }

    private static bool IsConnector(char ch)
        => ch is '-' or '+' or '_' or '/' or '.';

    private static bool IsDigitToken(string token)
        => token.Length == 1 && IsAsciiDigit(token[0]);

    private static bool IsAsciiDigit(char ch)
        => ch >= '0' && ch <= '9';

    private static bool IsOperatorOrPunctuation(char ch)
        => char.IsPunctuation(ch) || char.IsSymbol(ch);
}
