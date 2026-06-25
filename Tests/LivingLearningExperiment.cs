using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (no box-ticking) — is "losing learned state" a REAL obstacle in the field-cognition mind, or did the
// architecture already solve it? The homeostasis structure (defend a valued state against a degrading force) only
// has a purpose if learned knowledge actually erodes under continued learning. Measure it before integrating anything.
public sealed class LivingLearningExperiment
{
    private readonly ITestOutputHelper _out;
    public LivingLearningExperiment(ITestOutputHelper o) => _out = o;

    [Fact(Skip = SlowTests.BareSubjectWarmup)] // bare subjects ("alice is doctor") need a GRU-trained warm-up
    public void Does_LearnedKnowledge_Erode_UnderContinuedLearning()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        GrammarWarmup.WarmRoleHead(tok, model, mind); // LEARNED role parser (no hardcoded copula fallback) — warm as the gym does

        void Tell(string s) => mind.Generate(new GenerationRequest(s, 8));
        string Ask(string s) => mind.Generate(new GenerationRequest(s, 8)).Output?.Trim() ?? "";

        // The mind learns 10 ANCHOR facts (subject -> attribute).
        var anchors = new (string who, string what)[]
        {
            ("alice","doctor"), ("bob","teacher"), ("carol","pilot"), ("dave","chef"), ("eve","artist"),
            ("frank","lawyer"), ("grace","nurse"), ("henry","baker"), ("iris","painter"), ("jack","farmer"),
        };
        foreach (var (who, what) in anchors) Tell($"{who} is {what}");

        int RecallScore()
            => anchors.Count(a => Ask($"what is {a.who}").Equals(a.what, StringComparison.OrdinalIgnoreCase));

        var before = RecallScore();
        _out.WriteLine($"[anchors] recalled {before}/{anchors.Length} immediately after learning");

        // Now the mind keeps LEARNING — a flood of new facts (the entropy of continued experience).
        var rng = new Random(11);
        string[] names = Enumerable.Range(0, 60).Select(i => $"person{i}").ToArray();
        string[] jobs = { "miner","sailor","clerk","guard","scout","smith","weaver","hunter","mason","cook","tailor","scribe" };
        foreach (var n in names) Tell($"{n} is {jobs[rng.Next(jobs.Length)]}");

        var after = RecallScore();
        _out.WriteLine($"[anchors] recalled {after}/{anchors.Length} AFTER learning 60 more facts (erosion = {before - after})");

        // The field-cognition memory is robust: learned knowledge does NOT erode under continued learning (the old
        // gym's forgetting was the GRU CLASSIFIER eroding — bypassed here). So "losing learned state" is NOT the
        // obstacle the livingness must defend against; the substrate already solved it.
        Assert.Equal(anchors.Length, before);                 // it learns all of it
        Assert.True(after >= anchors.Length, $"robust memory — no erosion under load; {after}/{anchors.Length}");
    }

    [Fact(Skip = SlowTests.BareSubjectWarmup)] // bare subjects ("qab is red") — The OTHER candidate obstacle, at SCALE: does the mind stay COHERENT (genesis G2 non-contradiction) when
           // much of what it holds is later CONTRADICTED by new experience and buried under unrelated learning — or
           // does it accumulate conflicting truths and answer stale / at random?
    public void Does_The_Mind_Hold_Contradictions_AtScale()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        GrammarWarmup.WarmRoleHead(tok, model, mind); // LEARNED role parser (no hardcoded copula fallback) — warm as the gym does
        void Tell(string s) => mind.Generate(new GenerationRequest(s, 8));
        string Ask(string who) => mind.Generate(new GenerationRequest($"what is {who}", 8)).Output?.Trim() ?? "";

        // 20 beliefs (subject -> value). All-letter names (the learner needs all-letter content words). Current = v1.
        static string Nm(char p, int i) => $"{p}{(char)('a' + i / 26 % 26)}{(char)('a' + i % 26)}";
        var people = Enumerable.Range(0, 20).Select(i => Nm('q', i)).ToArray();
        var v1 = new[] { "red","blue","green","gold","black","white","brown","pink","cyan","lime","navy","teal","plum","ruby","jade","sand","rose","mint","grey","tan" };
        var v2 = new[] { "iron","zinc","lead","copper","steel","brass","nickel","silver","cobalt","chrome","bronze","pewter","mercury","platinum","aluminium","titanium","tungsten","carbon","cadmium","gallium" };
        var current = new System.Collections.Generic.Dictionary<string,string>();
        for (var i = 0; i < people.Length; i++) { Tell($"{people[i]} is {v1[i]}"); current[people[i]] = v1[i]; }

        // The world CHANGES: 12 of the 20 beliefs are contradicted by a single new observation (the current truth is now v2).
        for (var i = 0; i < 12; i++) { Tell($"{people[i]} is {v2[i]}"); current[people[i]] = v2[i]; }

        // ...then a flood of UNRELATED learning buries the updates (so recency alone can't carry them).
        var rng = new Random(5);
        var jobs = new[] { "miner","sailor","clerk","guard","scout","smith","weaver","hunter","mason","cook" };
        for (var i = 0; i < 50; i++) Tell($"{Nm('w', i)} is {jobs[rng.Next(jobs.Length)]}");

        // Probe: separate the CONTRADICTED beliefs (0..11) from the UNTOUCHED ones (12..19).
        int updHit = 0, updStale = 0, updAbstain = 0, stableHit = 0, stableAbstain = 0;
        for (var i = 0; i < people.Length; i++)
        {
            var ans = Ask(people[i]);
            var blank = string.IsNullOrWhiteSpace(ans);
            if (i < 12)
            {
                if (ans.Equals(v2[i], StringComparison.OrdinalIgnoreCase)) updHit++;
                else if (ans.Equals(v1[i], StringComparison.OrdinalIgnoreCase)) updStale++;
                else if (blank) updAbstain++;
            }
            else
            {
                if (ans.Equals(v1[i], StringComparison.OrdinalIgnoreCase)) stableHit++;
                else if (blank) stableAbstain++;
            }
        }
        _out.WriteLine($"[CONTRADICTED 0-11] current={updHit}  stale={updStale}  ABSTAIN={updAbstain}  (of 12)");
        _out.WriteLine($"[UNTOUCHED  12-19] recalled={stableHit}  abstain={stableAbstain}  (of 8)");

        // THE LIVING REQUIREMENT (genesis G2 / free-energy): when the world contradicts a belief, the mind must UPDATE
        // its model — answer with the CURRENT truth, not a stale one. Before belief-revision this was 2/12 (it clung
        // to the past); with it, the mind stays current. Untouched beliefs are unharmed.
        Assert.True(updHit >= 10, $"the living mind updates to the CURRENT truth, not stale; current={updHit}/12 stale={updStale}");
        Assert.Equal(8, stableHit);
    }
}
