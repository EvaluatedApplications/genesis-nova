using GenesisNova.Tokenization;

namespace GenesisNova.Tests;

public sealed class WhitespaceGenesisTokenizerTests
{
    [Fact]
    public void WhenEncodingThenDecoding_ThenNormalizedTextRoundTrips()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();

        var tokens = tokenizer.Encode("Hello   WORLD", addBos: true, addEos: true);
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("hello world", decoded);
        Assert.True(tokenizer.VocabularySize >= 5);
    }

    [Fact]
    public void WhenTokenizingExpression_ThenOperatorsArePreservedWithoutExtraSpaces()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();

        var tokens = tokenizer.Encode("12+3");
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("12+3", decoded);
        Assert.True(tokens.Length >= 3);
    }
}
