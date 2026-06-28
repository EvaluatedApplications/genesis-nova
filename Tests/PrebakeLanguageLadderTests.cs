using System.Collections.Generic;
using System.Linq;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

// FAST (no training): the synthetic language prebake generates the right SHAPES — short function-word fragments, content
// is REAL words salted with held-out NONCE, and the relational/argument sides of the readiness signal are disjoint. Per
// option 4 the synthetic prebake is a FUNCTION-WORD FOUNDATION capped at L1 (a parsing prerequisite); longer + multi-
// sentence composition is the corpus warmer's + Merge parser's job, so the ladder no longer climbs. Validates the schema
// + the real-vs-nonce decision deterministically, without a slow warming run.
public sealed class PrebakeLanguageLadderTests
{
    [Fact]
    public void Level1_IsShortFragments_NoMultiSentence()
    {
        var c = new PrebakeLanguageCurriculum(trainPerCycle: 200, seed: 1);
        var inputs = c.NextTrainBatch().Select(t => t.Input).ToList();
        Assert.All(inputs, s => Assert.DoesNotContain(" . ", s));        // L1 never multi-sentence
        Assert.All(inputs, s => Assert.True(s.Split(' ').Length <= 5));  // short fragments (incl. coordinated glue: "my cat and your dog")
    }

    [Fact]
    public void Prebake_IsCappedAtL1_FunctionWordFoundationOnly()
    {
        // Option 4: the synthetic prebake is a function-word foundation (a parsing prerequisite), NOT a multi-sentence
        // composer — longer/multi-sentence composition is the corpus warmer's + Merge parser's job. So the ladder is
        // capped at L1: driving mastery cycles must never climb past it.
        var c = new PrebakeLanguageCurriculum(trainPerCycle: 400, seed: 2);
        for (var i = 0; i < 20; i++) c.RecordCycle(new CycleGrade(0.95, 1, 1));
        Assert.Equal(1, c.Difficulty);
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
