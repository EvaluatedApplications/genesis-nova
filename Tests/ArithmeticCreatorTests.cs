using System.Text.RegularExpressions;
using GenesisNova.Data.Creators;

namespace GenesisNova.Tests;

public sealed class ArithmeticCreatorTests
{
    [Fact]
    public void WhenGeneratingAddExamples_ThenIncludesCompactOperatorSurface()
    {
        var sut = new ArithmeticCreator("add");
        var examples = sut.Generate(count: 30, difficulty: 0, forTraining: true);

        Assert.Contains(examples, e => Regex.IsMatch(e.Input, @"^-?\d+\+-?\d+$"));
    }

    [Fact]
    public void WhenGeneratingSubExamples_ThenIncludesCompactOperatorSurface()
    {
        var sut = new ArithmeticCreator("sub");
        var examples = sut.Generate(count: 30, difficulty: 0, forTraining: true);

        Assert.Contains(examples, e => Regex.IsMatch(e.Input, @"^-?\d+-\-?\d+$"));
    }

    [Fact]
    public void WhenDifficultyIsHigh_ThenArithmeticIncludesSentenceStylePrompts()
    {
        var sut = new ArithmeticCreator("add");
        var examples = sut.Generate(count: 800, difficulty: 3, forTraining: true);

        Assert.Contains(examples, e => e.Input.Contains("what is", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.Contains("tell me", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void WhenDifficultyIsHigh_ThenArithmeticPermutesSynonymKeywords()
    {
        var sut = new ArithmeticCreator("add");
        var examples = sut.Generate(count: 800, difficulty: 3, forTraining: true);

        Assert.Contains(examples, e => e.Input.Contains("plus", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.Contains("added to", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(examples, e => e.Input.Contains("sum", StringComparison.OrdinalIgnoreCase) || e.Input.Contains("total", StringComparison.OrdinalIgnoreCase));
    }
}
