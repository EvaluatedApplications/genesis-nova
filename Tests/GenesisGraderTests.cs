using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>Contextual, LLM-tolerant grading: the answer must OCCUR (value-aware), filler/personality is free,
/// and only OVER-GENERATION OF THE ANSWER'S TYPE (extra numbers) is penalized. The exact cases from the spec.</summary>
public class GenesisGraderTests
{
    private static double G(string output, string answer)
        => GenesisGrader.Quality(output, new[] { answer }, requiredDepth: 1, usedNeuralFallback: false, requirePlatonic: false);

    [Fact]
    public void AnswerOccurs_WithFillerOrPersonality_ScoresFull()
    {
        Assert.Equal(1.0, G("20", "20"), 3);
        Assert.Equal(1.0, G("its 20", "20"), 3);
        Assert.Equal(1.0, G("the answer is 20 actually", "20"), 3);
        Assert.Equal(1.0, G("the answer is twenty, am I right?", "20"), 3); // value-equiv + personality
    }

    [Fact]
    public void DuplicateOfRightNumber_IsMildOverGeneration()
    {
        var s = G("20 20", "20");
        Assert.True(s > 0.7 && s < 1.0, $"expected mild penalty for a duplicate number, got {s}");
    }

    [Fact]
    public void CompetingNumbers_AreHedging_AndFail()
    {
        Assert.True(G("20 30 40", "20") < 0.1, "digit hedging should fail");
        Assert.True(G("the answer is twenty thirty forty", "20") < 0.5, "word hedging should be penalized");
    }

    [Fact]
    public void WrongNumber_ScoresZero()
        => Assert.Equal(0.0, G("the answer is 30", "20"), 3);

    [Fact]
    public void RequirePlatonic_NeuralFallback_ScoresZero()
        => Assert.Equal(0.0, GenesisGrader.Quality("20", new[] { "20" }, 1, usedNeuralFallback: true, requirePlatonic: true), 3);

    private static double GS(string output, string[] allowed, string[] vocab)
        => GenesisGrader.Quality(output, allowed, requiredDepth: 1, usedNeuralFallback: false, requirePlatonic: false, answerVocabulary: vocab);

    [Fact]
    public void Category_AcceptsAnyValidGrouping_FillerFree_PenalizesWrongCategory()
    {
        var allowed = new[] { "fruit", "food" };               // apple → fruit OR food
        var vocab = new[] { "fruit", "food", "animal", "color", "vehicle" };
        Assert.Equal(1.0, GS("fruit", allowed, vocab), 3);
        Assert.Equal(1.0, GS("food", allowed, vocab), 3);                       // valid alternate
        Assert.Equal(1.0, GS("the answer is fruit", allowed, vocab), 3);        // filler free
        Assert.True(GS("its a fruit or animal", allowed, vocab) < 0.6, "competing category = hedging");
        Assert.Equal(0.0, GS("vehicle", allowed, vocab), 3);                    // no valid answer present
    }
}
