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
}
