using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// Unit-proves the LEARNED number-word lexicon in isolation (no model, no training loop): atoms learned by observation,
// scales learned COMPOSITIONALLY by solving the place-value equation, and universal base-10 compose/decompose both ways.
public sealed class NumberWordLexiconTests
{
    private readonly ITestOutputHelper _out;
    public NumberWordLexiconTests(ITestOutputHelper o) => _out = o;

    private static NumberWordLexicon TaughtAtoms()
    {
        var lex = new NumberWordLexicon();
        string[] ones = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
        for (var i = 0; i < ones.Length; i++) lex.LearnAtom(ones[i], i);
        string[] tens = { "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
        for (var i = 0; i < tens.Length; i++) lex.LearnAtom(tens[i], (i + 2) * 10);
        return lex;
    }

    [Fact]
    public void Atoms_Compose_And_Decompose_BothWays()
    {
        var lex = TaughtAtoms();
        // SCALE words are learned COMPOSITIONALLY from a digit->words example, not hardcoded.
        lex.LearnFromDigitWords(100, new[] { "one", "hundred" });           // residual "hundred" must be 100
        lex.LearnFromDigitWords(2000, new[] { "two", "thousand" });          // residual "thousand" must be 1000
        Assert.True(lex.KnowsAtom("hundred"));
        Assert.True(lex.KnowsAtom("thousand"));

        void Parse(string words, long want)
        {
            var ok = lex.TryParse(words.Split(' '), out var got);
            _out.WriteLine($"  parse '{words}' -> {got} (want {want}) {(ok && got == want ? "OK" : "MISS")}");
            Assert.True(ok && got == want);
        }
        Parse("forty seven", 47);
        Parse("one hundred thirteen", 113);
        Parse("two hundred fifty", 250);
        Parse("one thousand two hundred thirty four", 1234);
        Parse("nineteen", 19);
        Parse("five as a number", 5);          // framing words ignored

        void ToWords(long n, string want)
        {
            var ok = lex.TryToWords(n, out var got);
            _out.WriteLine($"  toWords {n} -> '{got}' (want '{want}') {(ok && got == want ? "OK" : "MISS")}");
            Assert.True(ok && got == want);
        }
        ToWords(7, "seven");
        ToWords(47, "forty seven");
        ToWords(113, "one hundred thirteen");
        ToWords(250, "two hundred fifty");
        ToWords(1234, "one thousand two hundred thirty four");
    }

    [Fact]
    public void Abstains_On_Untaught_Word_And_Missing_Scale()
    {
        var lex = TaughtAtoms(); // atoms only, NO scales taught
        Assert.False(lex.TryToWords(100, out _));            // "hundred" never learned -> abstain (honest), not a guess
        Assert.True(lex.TryParse("forty seven".Split(' '), out var v) && v == 47);
        Assert.False(lex.TryParse("glorp".Split(' '), out _)); // no number-word atom present at all
    }

    [Fact]
    public void Lexicon_Survives_Space_Snapshot_RoundTrip()
    {
        var a = new DialecticalSpace(256, seed: 7);
        string[] ones = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
        for (var i = 0; i < ones.Length; i++) a.NumberWords.LearnAtom(ones[i], i);
        a.NumberWords.LearnAtom("forty", 40); a.NumberWords.LearnAtom("hundred", 100);
        var n = a.NumberWords.AtomCount;

        var b = new DialecticalSpace(256, seed: 7);
        b.ImportSnapshot(a.ExportSnapshot()); // checkpoint round-trip

        Assert.Equal(n, b.NumberWords.AtomCount);                                         // atoms persisted
        Assert.True(b.NumberWords.TryParse("forty seven".Split(' '), out var v) && v == 47); // and still compose
        Assert.True(b.NumberWords.TryToWords(147, out var w) && w == "one hundred forty seven");
    }
}
