using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// DE-HARDCODING the last hardcoded cue — the IsQuestionCue wh-list (what/who/where/…) → a LEARNED interrogative cue
// (∘qst), the same ∘-anchor mechanism as the op/intent cues. A wh-question RETRIEVES its answer (output content absent
// from the input) and is FRONTED by a function-like token; that fronting word IS the interrogative, in ANY language.
// THE PROOF: a NONCE question marker ("kwa", in no list anywhere) routes as a question after a few examples, while a
// personal FACT does not — all with LearnedCuesOnly on (the wh-list disabled). This is what lets the corpus pay off:
// the routing rides the learned function-word signal the corpus warms, not an English word-list.
public sealed class QuestionCueDeHardcodingTest
{
    private readonly ITestOutputHelper _out;
    public QuestionCueDeHardcodingTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public void LearnedInterrogativeCue_RoutesQuestions_WithoutWhList()
    {
        var nova = new GenesisRuntimeState(new GenesisNovaConfig(HiddenSize: 96, Seed: 11).WithProductionMechanisms());
        var ds = (DialecticalSpace)nova.Memory;
        var eng = nova.Inference;

        // 1. WARM the learned function-word signal: content clusters (members co-occur with their own kind) plus bridging
        //    glue / question / NONCE words (co-occur with many clusters), exactly as windowed corpus text would produce.
        string[][] clusters =
        {
            new[]{"cat","dog","cow","pig","hen","fox","owl","bat","ant","elk"},
            new[]{"red","blue","green","pink","gray","black","white","brown","gold","teal"},
            new[]{"bob","sam","joe","amy","tom","kim","dan","liz","ben","eve"},
            new[]{"name","color","age","city","job","pet","car","book","song","food"},
            new[]{"rome","paris","tokyo","cairo","lima","oslo","delhi","perth","kyoto","nice"},
        };
        string[] bridges = { "what", "who", "where", "is", "the", "a", "my", "of", "kwa" }; // kwa = a NONCE interrogative
        var rng = new Random(11);
        for (var step = 0; step < 5000; step++)
        {
            var cl = clusters[rng.Next(clusters.Length)];
            var w1 = cl[rng.Next(cl.Length)]; var w2 = cl[rng.Next(cl.Length)];
            if (w1 != w2) ds.ObserveContradiction(w1, w2, 0.15);
            var any = clusters[rng.Next(clusters.Length)][rng.Next(10)];
            ds.ObserveContradiction(bridges[rng.Next(bridges.Length)], any, 0.2);
        }
        Assert.True(ds.IsFunctionLike("what") && ds.IsFunctionLike("kwa"), "warming did not make question/nonce words function-like");
        Assert.False(ds.IsFunctionLike("name"), "a content word wrongly read function-like");

        // 2. TEACH the interrogative cue from a few questions (incl. the nonce). Answer absent from the input ⇒ retrieval;
        //    the fronted glue token ⇒ the cue. NO wh-list is consulted — "kwa" can only be learned structurally.
        eng.LearnQuestionCue("what is my name", "bob");
        eng.LearnQuestionCue("where is the cat", "fox");
        eng.LearnQuestionCue("who is my pet", "dog");
        eng.LearnQuestionCue("kwa is my color", "blue");
        eng.LearnQuestionCue("kwa is my city", "rome");

        // 3. DE-HARDCODE: disable the wh-list; routing must now come purely from the learned ∘qst cue.
        eng.LearnedCuesOnly = true;
        bool Q(string s) { var r = eng.IsQueryOrRetrievalForTests(s); _out.WriteLine($"  '{s}' -> query? {r}"); return r; }

        Assert.True(Q("what is my name"), "a real wh-question did not route as a question");
        Assert.True(Q("kwa is my job"), "the NONCE interrogative did not route — still bound to a hardcoded list");
        Assert.False(Q("my name is bob"), "a personal FACT was wrongly routed as a question");
    }
}
