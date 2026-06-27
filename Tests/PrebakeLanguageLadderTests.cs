using System.Collections.Generic;
using System.Linq;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

// FAST (no training): the leveled language prebake generates the right SHAPES — L1 is short function-word fragments,
// higher levels compose LONGER + MULTI-SENTENCE inputs, content is REAL words salted with held-out NONCE, and the
// relational/argument sides of the readiness signal are disjoint. Validates the schema ladder + the real-vs-nonce
// decision deterministically, without a slow warming run.
public sealed class PrebakeLanguageLadderTests
{
    [Fact]
    public void Level1_IsShortFragments_NoMultiSentence()
    {
        var c = new PrebakeLanguageCurriculum(trainPerCycle: 200, seed: 1);
        var inputs = c.NextTrainBatch().Select(t => t.Input).ToList();
        Assert.All(inputs, s => Assert.DoesNotContain(" . ", s));        // L1 never multi-sentence
        Assert.All(inputs, s => Assert.True(s.Split(' ').Length <= 4));  // short fragments (incl. a 3-item content list)
    }

    [Fact]
    public void HigherLevels_ComposeLongerAndMultiSentence()
    {
        var c = new PrebakeLanguageCurriculum(trainPerCycle: 400, seed: 2);
        for (var i = 0; i < 20; i++) c.RecordCycle(new CycleGrade(0.95, 1, 1)); // drive to the top of the ladder
        Assert.Equal(5, c.Difficulty);
        var inputs = c.NextTrainBatch().Select(t => t.Input).ToList();
        Assert.Contains(inputs, s => s.Contains(" . "));                 // multi-sentence composition appears
        Assert.Contains(inputs, s => s.Split(' ').Length >= 5);         // genuinely longer than L1
    }

    [Fact]
    public void Uses_RealWords_With_NonceSalt()
    {
        var c = new PrebakeLanguageCurriculum(trainPerCycle: 600, seed: 3);
        var toks = c.NextTrainBatch().SelectMany(t => (t.Input + " " + t.Output).Split(' ')).ToHashSet();
        Assert.Contains("the", toks);                                                  // real closed-class glue
        Assert.Contains(toks, t => new[] { "cat", "dog", "red", "blue", "sam" }.Contains(t)); // real vocabulary
        var nonce = c.NonceContent(24).ToHashSet();
        Assert.Contains(toks, t => nonce.Contains(t));                                  // nonce salt present
    }

    [Fact]
    public void ReadinessSets_Function_Vs_Argument_AreDisjoint()
    {
        var c = new PrebakeLanguageCurriculum(seed: 4);
        var glue = c.Glue.ToHashSet();
        var content = c.SampleContent(100).ToHashSet();
        Assert.Empty(glue.Intersect(content));   // a word is never both the skeleton AND content
    }
}
