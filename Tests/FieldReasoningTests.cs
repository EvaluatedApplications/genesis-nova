using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Relaxation-as-reasoning on the REAL substrate (PLATONIC_MIND.md): build concepts via ObserveContradiction (real
/// distributional clouds), then DialecticalSpace.Reason relaxes a query over them. Tests the three predictions on
/// the actual representation, not a toy: categorisation/recall, disambiguation of an ambiguous word by context, and
/// abstention on an unknown query. Production dims.
/// </summary>
public sealed class FieldReasoningTests
{
    private readonly ITestOutputHelper _out;
    public FieldReasoningTests(ITestOutputHelper o) => _out = o;

    private static DialecticalSpace BuildWorld()
    {
        var s = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        void Rel(string a, params string[] ctx) { foreach (var c in ctx) s.ObserveContradiction(a, c, 0.05); }
        for (var ep = 0; ep < 10; ep++)
        {
            Rel("cat", "animal", "pet", "fur");
            Rel("dog", "animal", "pet", "bark");
            Rel("lion", "animal", "wild", "roar");
            Rel("car", "vehicle", "road", "engine");
            Rel("stream", "river", "water", "fish");   // river-sense concept
            Rel("fund", "money", "loan", "cash");       // money-sense concept
            Rel("bank", "river", "water", "money", "loan"); // AMBIGUOUS: both senses
        }
        return s;
    }

    [Fact]
    public void Reason_Categorises_Disambiguates_Abstains()
    {
        var s = BuildWorld();

        // (1) CATEGORISE / recall: "what is animal + pet?" → a member of that cluster, and it settles.
        var t1 = s.Reason(new[] { "animal", "pet" });
        _out.WriteLine($"animal+pet → {t1.Symbol} (conf {t1.Confidence:F2}, settled {t1.Settled})");
        Assert.Contains(t1.Symbol, new[] { "cat", "dog", "lion" });
        Assert.True(t1.Settled);

        // (2) DISAMBIGUATE: the same ambiguous word 'bank' resolves to the right SENSE REGION by context — river
        // context settles into the river-sense cluster, money context into the money-sense cluster.
        var river = new[] { "stream", "water", "fish" };
        var money = new[] { "fund", "loan", "cash" };
        var tr = s.Reason(new[] { "bank", "river" });
        var tm = s.Reason(new[] { "bank", "money" });
        _out.WriteLine($"bank+river → {tr.Symbol}   bank+money → {tm.Symbol}");
        Assert.Contains(tr.Symbol, river); // river context → a river-sense concept
        Assert.Contains(tm.Symbol, money); // money context → a money-sense concept

        // (3) ABSTAIN: an unknown query has no near basin → the field does not settle (and does not invent).
        var tu = s.Reason(new[] { "zxqv", "wmkp" });
        _out.WriteLine($"unknown → settled {tu.Settled} (conf {tu.Confidence:F2})");
        Assert.False(tu.Settled);
    }
}
