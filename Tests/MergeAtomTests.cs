using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;
using ElementKind = GenesisNova.Cognition.Platonic.ElementKind;

namespace GenesisNova.Tests;

// M0 of the Merge foundation ([[nova-merge-substrate-plan]]): the ATOM. Merge(a, b | label) must produce a FIRST-CLASS
// element — kind Relation, holding a and b as ▷ Components, positioned at the BLEND of their meanings — so it is
// COMPOSABLE (an element), RECURSIVE (a Merge output is a valid endpoint of the next Merge), and idempotent. This is
// Chomsky's binary recursive composition realised natively on the substrate; the copula-pivot is one labelled Merge.
public sealed class MergeAtomTests
{
    private readonly ITestOutputHelper _out;
    public MergeAtomTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Merge_IsAComposable_Recursive_PositionedRelationElement()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        // MERGE two atoms → a new element.
        var m1 = space.Merge("favorite", "color", "mod");
        var e1 = space.GetElement(m1);
        Assert.NotNull(e1);
        Assert.Equal(ElementKind.Relation, e1!.Kind);                 // it IS an element (a Relation), not a side-table row

        // ▷ COMPONENTS: it holds both endpoints (the structural slot Element.Components, finally used for relations).
        var fav = space.GetElement("favorite"); var col = space.GetElement("color");
        Assert.NotNull(fav); Assert.NotNull(col);
        Assert.Equal(2, e1.Components.Length);
        Assert.Contains(fav!.Id, e1.Components);
        Assert.Contains(col!.Id, e1.Components);

        // POSITION = the blend (centroid) of the two meanings — m carries a composed MEANING, not just a link.
        var ca = space.SemanticVectorOf("favorite"); var cb = space.SemanticVectorOf("color");
        Assert.NotNull(ca); Assert.NotNull(cb);
        for (var i = 0; i < ca!.Length; i++)
            Assert.Equal(0.5 * (ca[i] + cb![i]), e1.SemanticFace[i], 6);

        // RECURSION: a Merge output is a valid endpoint of the next Merge — hierarchy from one operation.
        var m2 = space.Merge("my", m1, "poss");
        var e2 = space.GetElement(m2);
        Assert.NotNull(e2);
        Assert.Equal(2, e2!.Components.Length);
        Assert.Contains(e1.Id, e2.Components);                        // the FIRST merge is a COMPONENT of the second
        var my = space.GetElement("my");
        Assert.Contains(my!.Id, e2.Components);

        // Three-level deep — proving unbounded nesting (not a fixed two-token rule).
        var m3 = space.Merge(m2, "blue", "is");
        var e3 = space.GetElement(m3);
        Assert.NotNull(e3);
        Assert.Contains(e2.Id, e3!.Components);

        // IDEMPOTENT: the same (a, label, b) is the same element (content-addressed) — composition is deterministic.
        Assert.Equal(m1, space.Merge("favorite", "color", "mod"));

        _out.WriteLine($"m1={m1}\nm2={m2}\nm3={m3}");
    }
}
