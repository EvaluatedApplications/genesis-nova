using GenesisNova.Data.Creators;

namespace GenesisNova.Tests;

public sealed class LanguageCreatorTests
{
    [Fact]
    public void WhenDifficultyIsHigh_ThenLanguageUsesSentenceStructuredVariants()
    {
        var sut = LanguageDefaults.Greet;
        var examples = sut.Generate(count: 500, difficulty: 3, forTraining: true);

        Assert.Contains(examples, e => e.Input.StartsWith("respond in one short sentence:", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.StartsWith("for clarity, can you answer this question:", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.EndsWith("?", StringComparison.Ordinal));
    }

    [Fact]
    public void WhenDifficultyIsHigh_ThenLanguagePermutesSynonymKeywords()
    {
        var sut = LanguageDefaults.Greet;
        var examples = sut.Generate(count: 500, difficulty: 3, forTraining: true);

        Assert.Contains(examples, e => e.Input.StartsWith("answer in one concise sentence:", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.StartsWith("to be clear, can you answer this:", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.StartsWith("please respond:", StringComparison.OrdinalIgnoreCase));
    }
}
