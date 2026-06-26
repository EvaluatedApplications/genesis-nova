using System.Linq;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (#6, evidence-first): the whitespace tokenizer can't segment non-spaced scripts (CJK/kana) — with no
// spaces the characters accumulate into ONE token, so nothing downstream can work. Measure the gap, and the signal for
// a fix: CJK/kana characters are Unicode "OtherLetter" (caseless = ideographic/syllabic, each its own meaningful unit),
// so segmenting THEM per-character is correct, while Latin letter-runs stay whole words.
public sealed class TokenizerScriptExperiment
{
    private readonly ITestOutputHelper _out;
    public TokenizerScriptExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Measure_Current_Tokenizer_On_NonSpaced_Scripts()
    {
        var tok = new WhitespaceGenesisTokenizer();
        string[] Toks(string s) => tok.Encode(s).Select(id => tok.Decode(new[] { id })).Where(t => t.Length > 0).ToArray();

        var english = Toks("my name is stephen");
        var chinese = Toks("我叫史蒂芬");        // "my name (is) stephen"
        var japanese = Toks("こんにちは世界");    // "hello world" (kana + kanji)
        var mixed = Toks("hello 世界");

        _out.WriteLine($"english  ({english.Length}): {string.Join(" | ", english)}");
        _out.WriteLine($"chinese  ({chinese.Length}): {string.Join(" | ", chinese)}");
        _out.WriteLine($"japanese ({japanese.Length}): {string.Join(" | ", japanese)}");
        _out.WriteLine($"mixed    ({mixed.Length}): {string.Join(" | ", mixed)}");

        // The Unicode signal a fix would key on: each CJK/kana char is OtherLetter (caseless), Latin letters are cased.
        bool IsCaseless(char c) => System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c) == System.Globalization.UnicodeCategory.OtherLetter;
        _out.WriteLine($"caseless? 我={IsCaseless('我')} あ={IsCaseless('あ')} a={IsCaseless('a')} 5={IsCaseless('5')}");

        Assert.Equal(4, english.Length);     // Latin: word-segmented correctly (unchanged)
        Assert.Equal(5, chinese.Length);     // FIXED: 我|叫|史|蒂|芬 — each morpheme its own token
        Assert.Equal(7, japanese.Length);    // FIXED: こ|ん|に|ち|は|世|界
        Assert.Equal(new[] { "hello", "世", "界" }, mixed); // FIXED: Latin word kept whole, CJK split
        Assert.True(IsCaseless('我') && IsCaseless('あ') && !IsCaseless('a')); // the universal Unicode signal holds
    }
}
