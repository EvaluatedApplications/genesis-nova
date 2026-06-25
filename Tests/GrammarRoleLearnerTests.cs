using System;
using System.Collections.Generic;
using GenesisNova.Cognition;
using Xunit;
using Xunit.Abstractions;
using Role = GenesisNova.Cognition.GrammarRoleLearner.Role;

namespace GenesisNova.Tests;

// The alignment role-learner: learns copula/question/possessive/key/value ROLES from the ASSERT/RECALL structure
// alone — no word-order, no word-lists. The decisive checks: a NONCE query cue ("glarf") and NONCE copula ("ploo"),
// in no list anywhere, are classified by HOW THEY BEHAVE (query cues live in answer-absent frames; copulas don't).
public sealed class GrammarRoleLearnerTests
{
    private readonly ITestOutputHelper _out;
    public GrammarRoleLearnerTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Learns_StructuralRoles_FromAssertRecall_NoWordLists_IncludingNonce()
    {
        var learner = new GrammarRoleLearner();
        var world = new[]
        {
            ("my", "name", "sam"), ("your", "name", "rex"), ("his", "dog", "fido"),
            ("her", "car", "audi"), ("our", "job", "coder"), ("their", "city", "lisbon"),
        };
        var copulas = new[] { "is", "was", "ploo" };      // ploo = NONCE copula (in no list)
        var queries = new[] { "what", "whats", "who", "glarf" }; // glarf = NONCE query cue (in no list)
        var rng = new Random(3);
        for (var i = 0; i < 3000; i++)
        {
            var (p, n, v) = world[rng.Next(world.Length)];
            if (i % 2 == 0) learner.Observe(new[] { p, n, copulas[rng.Next(copulas.Length)], v }, v); // ASSERT: answer present
            else learner.Observe(new[] { queries[rng.Next(queries.Length)], p, n }, v);                // RECALL: answer absent
        }

        // FILLER detection is the space's separately-learned centrality signal (proven). Here we supply it so this test
        // isolates the ROLE logic; the integration uses DialecticalSpace.IsFunctionLike.
        var fillers = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { "my", "your", "his", "her", "our", "their", "is", "was", "ploo", "what", "whats", "who", "glarf" };
        Role R(string t) { var r = learner.Classify(t, x => fillers.Contains(x)); _out.WriteLine($"  {t,-7} -> {r}"); return r; }

        // Query cues — incl. the NONCE one — live ONLY in answer-absent (recall) inputs:
        Assert.Equal(Role.Query, R("what"));
        Assert.Equal(Role.Query, R("glarf"));   // classified by BEHAVIOUR, not a list
        // Copulas — incl. the NONCE one — are generic filler, NOT query cues (they sit in assertions):
        Assert.Equal(Role.Filler, R("is"));
        Assert.Equal(Role.Filler, R("ploo"));
        // Possessives appear in BOTH frames → generic filler (kept in the subject key elsewhere, by adjacency):
        Assert.Equal(Role.Filler, R("my"));
        Assert.Equal(Role.Filler, R("their"));
        // Content: a queried noun is a KEY; a thing that's ever an answer is a VALUE:
        Assert.Equal(Role.Key, R("name"));
        Assert.Equal(Role.Key, R("dog"));
        Assert.Equal(Role.Value, R("sam"));
        Assert.Equal(Role.Value, R("audi"));
        // A never-seen token abstains (warm-start honesty):
        Assert.Equal(Role.Unknown, R("zzzz"));
    }
}
