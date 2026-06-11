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

    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var output = new List<int>();
        if (addBos)
            output.Add(BosTokenId);

        if (!string.IsNullOrWhiteSpace(text))
        {
            foreach (var part in Tokenize(text))
                output.Add(AddToken(part));
        }

        if (addEos)
            output.Add(EosTokenId);

        return output.ToArray();
    }

    private static IReadOnlyList<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var normalized = text.ToLowerInvariant();
        var buffer = new System.Text.StringBuilder();

        for (var i = 0; i < normalized.Length; i++)
        {
            var ch = normalized[i];
            if (char.IsWhiteSpace(ch))
            {
                FlushBuffer();
                continue;
            }

            if (IsAsciiDigit(ch))
            {
                FlushBuffer();
                tokens.Add(ch.ToString());
                continue;
            }

            if (char.IsLetterOrDigit(ch))
            {
                buffer.Append(ch);
                continue;
            }

            FlushBuffer();
            tokens.Add(ch.ToString());
        }

        FlushBuffer();
        return tokens;

        void FlushBuffer()
        {
            if (buffer.Length == 0)
                return;

            tokens.Add(buffer.ToString());
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

        var output = new System.Text.StringBuilder(words[0]);
        for (var i = 1; i < words.Count; i++)
        {
            var current = words[i];
            var previous = words[i - 1];
            if (ShouldConcatenate(previous, current))
                output.Append(current);
            else
                output.Append(' ').Append(current);
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

    private static bool ShouldConcatenate(string previous, string current)
    {
        if (IsDigitToken(previous) && IsDigitToken(current))
            return true;

        if (current.Length == 1 && IsOperatorOrPunctuation(current[0]))
            return true;

        if ((previous == "-" || previous == "+") && current.All(char.IsDigit))
            return true;

        return false;
    }

    private static bool IsDigitToken(string token)
        => token.Length == 1 && IsAsciiDigit(token[0]);

    private static bool IsAsciiDigit(char ch)
        => ch >= '0' && ch <= '9';

    private static bool IsOperatorOrPunctuation(char ch)
        => char.IsPunctuation(ch) || char.IsSymbol(ch);
}
