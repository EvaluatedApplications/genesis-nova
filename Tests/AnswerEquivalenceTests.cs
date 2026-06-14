using GenesisNova.Core;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// The face-aware return grader (<see cref="AnswerEquivalence"/>): a digit and its number-word are the
/// same answer (the equivalence the interface learns), while non-numeric answers stay exact-match. The
/// grader is a ground-truth oracle (reference vocabulary), so it never lets the model self-grade.
/// </summary>
public sealed class AnswerEquivalenceTests
{
    [Theory]
    [InlineData("2", "two")]
    [InlineData("two", "2")]
    [InlineData("18", "eighteen")]
    [InlineData("1 8", "18")]          // digit-run tokenisation folds to the same value
    [InlineData("eighteen", "1 8")]
    [InlineData("0", "zero")]
    [InlineData("2", "2")]             // exact still works
    [InlineData("Paris", "Paris")]     // non-numeric exact
    public void Equivalent_AcceptsSameValueAcrossSurfaces(string predicted, string expected)
        => Assert.True(AnswerEquivalence.Equivalent(predicted, expected));

    [Theory]
    [InlineData("2", "3")]             // different values
    [InlineData("two", "3")]
    [InlineData("greater", "less")]    // non-numeric, different
    [InlineData("greater", "2")]       // one numeric, one not
    [InlineData("paris", "london")]
    [InlineData("3", "-3")]            // sign matters
    [InlineData("", "2")]             // empty is not a match
    public void Equivalent_RejectsDifferentAnswers(string predicted, string expected)
        => Assert.False(AnswerEquivalence.Equivalent(predicted, expected));
}
