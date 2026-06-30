using System.Collections.Generic;
using System.Linq;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// BOUNDED-VOCAB + CHAR-FACE COMPOSITION (LLM-competitive token reading within a FIXED parameter budget).
///
/// The legacy model grows a learned ROW per token in _embT/_wOutT/_bOutT, so a large corpus balloons the
/// weights without bound. With <see cref="GenesisNovaConfig.CompositionalTokenEmbedding"/> on, the learned
/// per-token table is CAPPED at <see cref="GenesisNovaConfig.MaxModelVocab"/>; a token beyond the cap (OOV)
/// has its INPUT embedding COMPOSED from its deterministic spelling band via a small learned projection — NOT
/// a fresh row — so ANY token is readable within a bounded budget.
///
/// These pin the four guarantees: (1) an OOV token with NO row still produces a forward (read via the char
/// face) and the table does NOT grow; (2) two different OOV spellings get DIFFERENT embeddings (a genuine
/// read, not a shared UNK); (3) a flood of distinct tokens leaves the table bounded; (4) flag-off is the
/// legacy unbounded behaviour (the clamp is inert).
/// </summary>
public sealed class CompositionalTokenEmbeddingTests
{
    private readonly ITestOutputHelper _out;
    public CompositionalTokenEmbeddingTests(ITestOutputHelper o) => _out = o;

    private static void WireSpelling(GenesisNeuralModel model, WhitespaceGenesisTokenizer tok)
        => model.SetTokenSpelling(id => id >= 0 && id < tok.Vocabulary.Count ? tok.Vocabulary[id] : null);

    // Map i to a distinct lowercase-letter string (base-26), so each token is a single pure-letter word.
    private static string ToLetters(int i)
    {
        var s = string.Empty;
        do { s = (char)('a' + i % 26) + s; i = i / 26 - 1; } while (i >= 0);
        return s;
    }

    private static float[] Hidden(GenesisNeuralModel model, IReadOnlyList<int> tokens)
    {
        using var seed = model.EncodePromptState(tokens);
        return seed.detach().cpu().data<float>().ToArray();
    }

    [Fact]
    public void OovToken_WithNoRow_IsReadViaCharFace_AndTableStaysBounded()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var cap = tokenizer.VocabularySize; // base vocab (pad/bos/eos + digits) — any WORD added after this is OOV.
        var config = new GenesisNovaConfig(
            HiddenSize: ProductionDims.HiddenSize,
            CompositionalTokenEmbedding: true,
            MaxModelVocab: cap);
        var model = new GenesisNeuralModel(config);
        WireSpelling(model, tokenizer);

        // An out-of-cap WORD token: encoding adds it to the tokenizer (id >= cap), but the model must NOT grow a row.
        var oov = tokenizer.Encode("zarni").Single();
        Assert.True(oov >= cap, "the word token id must be beyond the bounded cap (OOV)");
        model.EnsureVocabularySize(tokenizer.VocabularySize); // would grow to vocab>cap legacy; here it clamps to cap

        // (a) the per-token table did NOT grow past the cap (VocabularySize == _embT.shape[0]).
        Assert.Equal(cap, model.VocabularySize);

        // (b) the model still produces a forward for the OOV token — composed from its char face, not a zero/UNK.
        var h = Hidden(model, new[] { oov });
        Assert.All(h, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        Assert.Contains(h, v => v != 0f);
        _out.WriteLine($"cap={cap} oovId={oov} VocabularySize={model.VocabularySize} ||h||0={h.Count(v => v != 0f)}");
    }

    [Fact]
    public void TwoDifferentOovSpellings_GetDifferentEmbeddings()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var cap = tokenizer.VocabularySize;
        var config = new GenesisNovaConfig(
            HiddenSize: ProductionDims.HiddenSize,
            CompositionalTokenEmbedding: true,
            MaxModelVocab: cap);
        var model = new GenesisNeuralModel(config);
        WireSpelling(model, tokenizer);

        var a = tokenizer.Encode("zarni").Single();
        var b = tokenizer.Encode("qwxplk").Single();
        model.EnsureVocabularySize(tokenizer.VocabularySize);
        Assert.True(a >= cap && b >= cap, "both word tokens must be OOV");

        var ha = Hidden(model, new[] { a });
        var hb = Hidden(model, new[] { b });

        // Different spellings → different deterministic bands → different composed embeddings → different hidden.
        // (A shared UNK row would make these identical.)
        var maxDiff = ha.Zip(hb, (x, y) => System.Math.Abs(x - y)).Max();
        _out.WriteLine($"maxDiff={maxDiff:F6}");
        Assert.True(maxDiff > 1e-4, $"two OOV spellings must produce different embeddings (maxDiff={maxDiff})");
    }

    [Fact]
    public void Flood_OfDistinctTokens_LeavesTableBounded()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var cap = tokenizer.VocabularySize;
        var config = new GenesisNovaConfig(
            HiddenSize: ProductionDims.HiddenSize,
            CompositionalTokenEmbedding: true,
            MaxModelVocab: cap);
        var model = new GenesisNeuralModel(config);
        WireSpelling(model, tokenizer);

        // Distinct LETTER-ONLY words (the tokenizer splits digits off, so digit suffixes would collapse to one token).
        for (var i = 0; i < 1000; i++)
            tokenizer.Encode("flood" + ToLetters(i));
        Assert.True(tokenizer.VocabularySize > cap + 500, "the tokenizer vocab must have ballooned");

        model.EnsureVocabularySize(tokenizer.VocabularySize);

        // The learned per-token table stays AT the cap no matter how many distinct tokens the corpus introduces.
        Assert.Equal(cap, model.VocabularySize);
        _out.WriteLine($"tokenizerVocab={tokenizer.VocabularySize} modelVocab={model.VocabularySize} (cap={cap})");
    }

    [Fact]
    public void GatedOff_GrowsTableAsBefore_ClampInert()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var config = new GenesisNovaConfig(
            HiddenSize: ProductionDims.HiddenSize,
            CompositionalTokenEmbedding: false, // OFF → byte-identical legacy growth
            MaxModelVocab: 8);                  // small cap present but must be IGNORED when the flag is off
        var model = new GenesisNeuralModel(config);

        for (var i = 0; i < 50; i++)
            tokenizer.Encode("word" + ToLetters(i));
        var fullVocab = tokenizer.VocabularySize;
        Assert.True(fullVocab > 8, "vocab must exceed the (ignored) cap to make the assertion meaningful");

        model.EnsureVocabularySize(fullVocab);

        // Flag off: the table grows to the FULL tokenizer vocab (the legacy unbounded behaviour) — the cap is inert.
        Assert.Equal(fullVocab, model.VocabularySize);
        _out.WriteLine($"flag-off vocab grew to {model.VocabularySize} (cap 8 ignored)");
    }
}
