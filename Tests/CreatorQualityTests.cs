using System.Text.RegularExpressions;
using GenesisNova.Data.Creators;

namespace GenesisNova.Tests;

public sealed class CreatorQualityTests
{
    [Fact]
    public void WhenGeneratingDivisionAtDifficultyZero_ThenOutputsAreMostlyWholeNumbers()
    {
        var sut = new ArithmeticCreator("div");
        var examples = sut.Generate(count: 40, difficulty: 0, forTraining: true);

        var parsed = examples
            .Select(e => double.TryParse(e.Output, out var value) ? (ok: true, value) : (ok: false, value: 0.0))
            .Where(x => x.ok)
            .Select(x => x.value)
            .ToArray();

        Assert.NotEmpty(parsed);
        var wholeCount = parsed.Count(v => Math.Abs(v - Math.Round(v)) < 1e-9);
        Assert.True(wholeCount >= parsed.Length * 0.8, $"Expected mostly clean division targets, got {wholeCount}/{parsed.Length}");
    }

    [Fact]
    public void WhenGeneratingComparisonExamples_ThenContainsOrderingTargets()
    {
        var sut = new ComparisonCreator();
        var examples = sut.Generate(count: 30, difficulty: 1, forTraining: true);

        Assert.Contains(examples, e => e.Output is "greater" or "less" or "equal");
        Assert.Contains(examples, e => e.Output is "yes" or "no");
    }

    [Fact]
    public void WhenGeneratingSequenceExamples_ThenOutputsAreNumericNextTerms()
    {
        var sut = new SequenceCreator();
        var examples = sut.Generate(count: 25, difficulty: 1, forTraining: true);

        Assert.All(examples, e =>
        {
            Assert.Matches(@"^\d+$", e.Output);
            Assert.Matches(@"(next in sequence|continue|what comes next)", e.Input);
            Assert.True(Regex.Matches(e.Input, @"\d+").Count >= 3);
        });
    }
}
