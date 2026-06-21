using System;
using System.Linq;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// The expanded personality trainer's grading contract: MULTIPLE correct rude replies, EMERGENT recombinations
/// that merely CONTAIN a valid rude phrase still count (not just verbatim trained replies), and OFF-PERSONA
/// polite output is penalized. Uses the SAME allowed/vocabulary the curriculum emits (intent replies ∪ the
/// universal rude markers; polite words as the competing vocabulary).
/// </summary>
public sealed class PersonalityGradingTests
{
    // A representative intent's replies (greeting), as the curriculum lists them.
    private static readonly string[] GreetReplies =
    {
        "what the fuck do you want now", "ugh, you again", "oh joy, it's you",
        "yeah yeah, what is it", "great, you're back", "what now", "spit it out",
    };

    private static string[] Allowed =>
        GreetReplies.Concat(PersonalityCurriculum.RudeMarkers).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();

    private static double G(string output) =>
        GenesisGrader.Quality(output, Allowed, requiredDepth: 1, usedNeuralFallback: false,
            requirePlatonic: false, answerVocabulary: PersonalityCurriculum.PoliteMarkers);

    [Fact] // A verbatim trained reply scores full.
    public void ExactRudeReply_ScoresFull()
    {
        Assert.True(G("ugh, you again") > 0.95);
        Assert.True(G("what the fuck do you want now") > 0.95);
    }

    [Fact] // MULTIPLE distinct valid replies all score — not one locked answer.
    public void MultipleDistinctReplies_AllScore()
    {
        foreach (var r in new[] { "spit it out", "what now", "great, you're back" })
            Assert.True(G(r) > 0.8, $"'{r}' should be a valid rude reply");
    }

    [Fact] // EMERGENT recombination — a natural rude line the model invents, NOT a verbatim reply, still hits
           // because it contains a valid rude marker.
    public void EmergentRudeRecombination_CountsAsHit()
    {
        Assert.True(G("ugh, just get lost then") > 0.8, "contains the rude marker 'get lost'");
        Assert.True(G("oh, fuck off already") > 0.8, "contains 'fuck off'");
        Assert.True(G("yeah whatever, who cares") > 0.8, "contains 'whatever' / 'who cares'");
    }

    [Fact] // OFF-PERSONA: a polite reply has no rude content and uses competing polite words → fails.
    public void PoliteReply_FailsThePersona()
    {
        Assert.True(G("happy to help, my pleasure!") < 0.2);
        Assert.True(G("of course, you're welcome") < 0.2);
    }

    [Fact] // MIXED: rude content present but polite hedging too → counts, but the competing polite word is taxed.
    public void RudeButHedgedPolitely_IsPenalisedNotZero()
    {
        var s = G("happy to help, now get lost"); // matches 'get lost', but 'happy to help' competes
        Assert.True(s > 0.2 && s < 0.9, $"mixed persona should be penalised, not full/zero; got {s:F2}");
    }
}
