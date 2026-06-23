using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// DIAGNOSIS of the gym's NEGATIVE geometry separation (full-gym run: related 2.016 &gt; unrelated 1.719, sep −0.297).
/// Reproduces the gym's actual relation structure on DialecticalSpace — synonym/category STAR-HUBS, op→face routing
/// hubs (reserved), and digit↔word relations — and DECOMPOSES the separation to find the cause: is it a metric
/// artifact (numbers at the semantic origin + reserved op-hubs inflate "related"), or do related WORDS genuinely
/// fail to cluster? Fast, no NN.
/// </summary>
public sealed class GymGeometryDiagnosticTests
{
    private readonly ITestOutputHelper _out;
    public GymGeometryDiagnosticTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void DiagnoseGymSeparation()
    {
        var s = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        var dim = s.FaceDimension;
        var semStart = FaceLayout.WordFaceStart(dim);

        var syn = new Dictionary<string, string[]>
        {
            ["big"] = new[] { "large", "huge", "giant" },
            ["small"] = new[] { "tiny", "little", "mini" },
            ["fast"] = new[] { "quick", "rapid", "swift" },
        };
        var cat = new Dictionary<string, string[]>
        {
            ["fruit"] = new[] { "apple", "banana", "grape" },
            ["vegetable"] = new[] { "carrot", "potato", "onion" },
        };
        var nums = new[] { ("five", "5"), ("six", "6"), ("seven", "7") };

        for (var ep = 0; ep < 40; ep++)
        {
            foreach (var (hub, items) in syn) foreach (var it in items) s.ObserveContradiction(hub, it, 0.1);
            foreach (var (hub, items) in cat) foreach (var it in items) s.ObserveContradiction(it, hub, 0.1);
            foreach (var (w, d) in nums) s.ObserveContradiction(w, d, 0.1);
            foreach (var op in new[] { "add", "sub" }) s.ObserveContradiction(op, "face:poly", 0.1);
            foreach (var op in new[] { "mul", "div" }) s.ObserveContradiction(op, "face:log", 0.1);
        }

        double D(string a, string b)
        {
            if (!s.TryGetConceptFace(a, out var fa) || !s.TryGetConceptFace(b, out var fb)) return double.NaN;
            var sum = 0.0; for (var i = semStart; i < fa.Length && i < fb.Length; i++) { var d = fa[i] - fb[i]; sum += d * d; }
            return Math.Sqrt(sum);
        }

        var g = s.SummarizePushPullGeometry();
        _out.WriteLine($"OVERALL  separation {g.Separation:F3}  (related {g.RelatedMean:F3}  unrelated {g.UnrelatedMean:F3})");

        // (1) Do related WORD pairs cluster vs unrelated word pairs? — the real positioning question.
        var relWord = new[] { ("big", "large"), ("big", "huge"), ("small", "tiny"), ("fast", "quick"), ("apple", "fruit"), ("carrot", "vegetable") }.Select(p => D(p.Item1, p.Item2)).ToArray();
        var unrelWord = new[] { ("big", "apple"), ("large", "carrot"), ("fast", "banana"), ("tiny", "potato"), ("huge", "onion"), ("quick", "grape") }.Select(p => D(p.Item1, p.Item2)).ToArray();
        _out.WriteLine($"WORDS    related {relWord.Average():F3}  unrelated {unrelWord.Average():F3}  → word-separation {unrelWord.Average() - relWord.Average():F3}");

        // (2) Related pairs INVOLVING NUMBERS (numbers sit at semantic origin → likely inflate 'related').
        var relNum = nums.Select(p => D(p.Item1, p.Item2)).ToArray();
        _out.WriteLine($"NUMBERS  word↔number related {relNum.Average():F3}  (e.g. five↔5)");

        // (3) Related pairs involving RESERVED op→face hubs.
        var relOp = new[] { ("add", "face:poly"), ("sub", "face:poly"), ("mul", "face:log") }.Select(p => D(p.Item1, p.Item2)).ToArray();
        _out.WriteLine($"OP-HUBS  op↔face related {relOp.Average():F3}  (reserved routing hubs)");

        // (4) Nearest-neighbour retrieval sanity: is a synonym actually the nearest concept to its cue?
        var nnBig = s.GetNearestConcepts("big", null, 3).Select(n => n.Symbol).ToArray();
        var nnApple = s.GetNearestConcepts("apple", null, 3).Select(n => n.Symbol).ToArray();
        _out.WriteLine($"NN(big)={string.Join(",", nnBig)}   NN(apple)={string.Join(",", nnApple)}");

        // REGRESSION GUARD (the gym found this; the pure separation metric missed it): NUMBERS must not pollute word
        // retrieval. Numbers carry a semantic orbital that settles toward their word, so a word's nearest concept is
        // its RELATED concept — not a number squatting at the semantic origin.
        Assert.True(unrelWord.Average() - relWord.Average() > 0.05, "related words must cluster closer than unrelated");
        Assert.Equal(new[] { "large", "huge", "giant" }.OrderBy(x => x), nnBig.OrderBy(x => x)); // big's neighbours are its synonyms
        Assert.Equal("fruit", nnApple[0]);                                                       // apple's nearest is its category
        Assert.DoesNotContain(nnApple, c => int.TryParse(c, out _) && c == nnApple[0]);           // not a number first
    }
}
