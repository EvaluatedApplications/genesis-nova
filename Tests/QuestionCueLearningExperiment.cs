using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (de-hardcoding, find-the-truth-first): the hardcoded IsQuestionCue wh-list (what/who/where/…) is the last
// cue not driven by the LEARNED function-word signal. The fact/question parse ALREADY reads ds.IsFunctionLike (which the
// corpus warms); the wh-list is the residue. Before de-hardcoding it we must KNOW the foundation supports it: does corpus-
// style co-occurrence make question words read FUNCTION-LIKE (they bridge many unrelated words) while content does not?
// If yes, IsFiller can drop the hardcoded OR. If no, that's the real blocker and we learn it here instead of shipping a regression.
public sealed class QuestionCueLearningExperiment
{
    private readonly ITestOutputHelper _out;
    public QuestionCueLearningExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Corpus_style_warming_makes_question_words_read_function_like()
    {
        var ds = new DialecticalSpace(faceDimension: 128);

        // Content lives in clusters (members co-occur with their OWN kind → high neighbour-coherence = content).
        string[][] clusters =
        {
            new[] { "cat", "dog", "cow", "pig", "hen", "fox", "owl", "bat", "ant", "elk" },
            new[] { "red", "blue", "green", "pink", "gray", "black", "white", "brown", "gold", "teal" },
            new[] { "bob", "sam", "joe", "amy", "tom", "kim", "dan", "liz", "ben", "eve" },
            new[] { "rome", "paris", "tokyo", "cairo", "lima", "oslo", "delhi", "perth", "kyoto", "nice" },
        };
        // Question words + plain function words: both BRIDGE — they precede/follow words from MANY clusters (diverse
        // neighbours that are not connected to each other → low coherence = function-like).
        string[] question = { "what", "who", "where" };
        string[] plainFn = { "is", "the", "a", "of" };

        // Warm: clustered content co-occurrence + bridging function/question co-occurrence, as windowed text would produce.
        var rng = new Random(7);
        for (var step = 0; step < 4000; step++)
        {
            var cl = clusters[rng.Next(clusters.Length)];
            var w1 = cl[rng.Next(cl.Length)];
            var w2 = cl[rng.Next(cl.Length)];
            if (w1 != w2) ds.ObserveContradiction(w1, w2, 0.15);                 // content clusters with its kind

            // a bridging word ties to content from ANY cluster (different clusters each time → diverse neighbourhood)
            var anyContent = clusters[rng.Next(clusters.Length)][rng.Next(10)];
            ds.ObserveContradiction(question[rng.Next(question.Length)], anyContent, 0.2);
            ds.ObserveContradiction(plainFn[rng.Next(plainFn.Length)], anyContent, 0.2);
        }

        double Frac(string[] ws) => ws.Count(ds.IsFunctionLike) / (double)ws.Length;
        var qFrac = Frac(question);
        var fnFrac = Frac(plainFn);
        var contentFrac = Frac(clusters.SelectMany(c => c).ToArray());
        _out.WriteLine($"function-like: question={qFrac:F2}  plainFn={fnFrac:F2}  content={contentFrac:F2}");
        foreach (var q in question) _out.WriteLine($"   {q}: fnLike={ds.IsFunctionLike(q)} degree={ds.GetRelationDegree(q)}");

        // The experiment's question: do question words read function-like (so IsFunctionLike alone could replace the wh-list)?
        Assert.True(qFrac >= 0.66, $"question words did NOT warm to function-like (qFrac={qFrac:F2}) — de-hardcoding IsQuestionCue via IsFunctionLike would regress");
        // And they must be SEPARABLE from content (else the signal is useless).
        Assert.True(qFrac - contentFrac >= 0.34, $"question vs content separation too small (q={qFrac:F2} content={contentFrac:F2})");
    }
}
