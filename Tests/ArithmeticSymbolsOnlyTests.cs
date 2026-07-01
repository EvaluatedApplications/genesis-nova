using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The hardcoded arithmetic may use ONLY universal MATH SYMBOLS (+ − × ÷ / and their ASCII compact forms) — a symbol IS
/// the operation, so it computes on a BLANK model like a number line does. Operator WORDS (plus/minus/times/over) are NOT
/// hardcoded: under the de-hardcoded config they resolve through the LEARNED op-cue, so a blank model ABSTAINS on a word
/// until it is taught, then computes any operands via the homomorphism. Symbols = structural; words = learned.
/// </summary>
public sealed class ArithmeticSymbolsOnlyTests
{
    private readonly ITestOutputHelper _out;
    public ArithmeticSymbolsOnlyTests(ITestOutputHelper o) => _out = o;

    private static GenesisInferenceEngine Engine()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        return new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true, LearnedCuesOnly = true };
    }

    private static string Ans(GenesisInferenceEngine e, string q) => (e.Generate(new GenerationRequest(q, 16)).Output ?? "").Trim();

    [Fact]
    public void UniversalSymbols_ComputeOnBlankModel_ButOperatorWordsMustBeLearned()
    {
        var e = Engine();

        // (1) UNIVERSAL SYMBOLS compute on a BLANK model — structural, no training (the codec is the number line).
        Assert.Equal("4", Ans(e, "2 + 2"));
        Assert.Equal("6", Ans(e, "10 - 4"));
        Assert.Equal("36", Ans(e, "12 x 3"));   // ASCII compact
        Assert.Equal("36", Ans(e, "12 × 3"));   // Unicode ×
        Assert.Equal("5", Ans(e, "20 / 4"));
        _out.WriteLine("SYMBOLS on blank: 2+2=4, 10-4=6, 12x3=36, 12×3=36, 20/4=5  (untrained, structural)");

        // (2) An operator WORD does NOT compute on a blank model — "plus" is not a symbol and no cue is learned yet.
        var coldPlus = Ans(e, "3 plus 4");
        Assert.NotEqual("7", coldPlus);
        _out.WriteLine($"WORD cold: '3 plus 4' → '{coldPlus}'  (NOT 7 — the word is not hardcoded)");

        // (3) After the op-cue is LEARNED (WarmOpCues teaches "plus"→add etc. via LearnArithmeticCue), the SAME words
        // compute on FRESH operands via the homomorphism — proving the word is learned, the computation still structural.
        GrammarWarmup.WarmOpCues(e);
        var plus = Ans(e, "3 plus 4"); var minus = Ans(e, "9 minus 5"); var times = Ans(e, "6 times 7");
        _out.WriteLine($"WORD warm: '3 plus 4'→'{plus}', '9 minus 5'→'{minus}', '6 times 7'→'{times}'  (learned cue)");
        Assert.Equal("7", plus);
        Assert.Equal("4", minus);
        Assert.Equal("42", times);
        // …and symbols still exact after warming.
        Assert.Equal("100", Ans(e, "50 + 50"));
    }
}
