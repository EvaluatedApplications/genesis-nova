using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// M1 of the Merge foundation ([[nova-merge-substrate-plan]]): fact memory IS a Merge typed "is", and recall is structural
// tree-traversal (key → fact → the non-key component) — no RetrievalFrame/∘ret/filler gate. The copula-pivot collapses
// into one labelled Merge. Proves store + recall for novel tokens, a multi-token key that is itself a Merge SUBTREE
// (recursion paying off), belief revision (the world changed), and honest abstention on an unknown key.
public sealed class MergeFactMemoryTests
{
    private readonly ITestOutputHelper _out;
    public MergeFactMemoryTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Merge_FactMemory_StoresAndRecalls()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        space.LearnFact("name", "sam");
        Assert.True(space.TryRecallFact("name", out var v)); Assert.Equal("sam", v);

        // NOVEL tokens — generalises structurally, nothing memorised.
        space.LearnFact("quux", "florp");
        Assert.True(space.TryRecallFact("quux", out var v2)); Assert.Equal("florp", v2);

        // MULTI-TOKEN key as a Merge SUBTREE — the key is itself a structure (recursion), recalled by the same structure.
        var key = space.Merge("my", space.Merge("favorite", "color", "mod"), "poss");
        space.LearnFact(key, "blue");
        Assert.True(space.TryRecallFact(key, out var v3)); Assert.Equal("blue", v3);

        // BELIEF REVISION — a fresh assertion supersedes the old (G2 / free-energy); recall returns the CURRENT truth.
        space.LearnFact("name", "rex");
        Assert.True(space.TryRecallFact("name", out var v4)); Assert.Equal("rex", v4);

        // Honest abstention on an unknown key.
        Assert.False(space.TryRecallFact("zzzznever", out _));
        _out.WriteLine("merge fact memory: store/recall/novel/multi-token/belief-revision OK");
    }
}
