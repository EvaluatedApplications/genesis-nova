using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M1 — COMPOSITIONAL ▷-HUBS (PLATONIC_THEORY.md Laws C/S; the scaling fix). Composites are hubs over REUSED
/// components: a word over its char atoms, a text over its word elements. The atom set is bounded by the alphabet,
/// so a novel word costs ~0 new atoms and a novel sentence ~0 new words — the direct test of "can it get bigger".
/// EMPIRICAL: the probe builds a vocabulary and measures the atom/composite growth, never assumes it.
/// </summary>
public sealed class DialecticalSpaceM1Tests
{
    private readonly ITestOutputHelper _out;
    public DialecticalSpaceM1Tests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Composition_ReusesComponents_BoundedGrowth()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        // A vocabulary over the lowercase alphabet (anchored to one concept so the words get created).
        var words = new[] { "cat", "car", "cart", "care", "dog", "dig", "cog", "rat", "tar", "art", "dart" };
        foreach (var w in words) space.ObserveContradiction(w, "thing", 0.2);

        // (1) Law S — char atoms are bounded by the ALPHABET, not the vocabulary size.
        _out.WriteLine($"vocab={words.Length} composites={space.CompositeCount} atoms={space.AtomCount}");
        Assert.True(space.AtomCount <= 26, $"char atoms must stay bounded by the alphabet; got {space.AtomCount}");

        // (2) ▷ hierarchy — a word is a hub over its char atoms, and the SAME atom is reused across words.
        Assert.Equal(new[] { "atom:c", "atom:a", "atom:t" }, space.ComponentSymbolsOf("cat"));
        foreach (var w in new[] { "cat", "car", "rat", "tar", "art" })
            Assert.Contains("atom:a", space.ComponentSymbolsOf(w)); // one shared 'a' atom referenced by all

        // (3) Law S — a NOVEL word over already-seen chars adds ZERO new atoms (only a hub + edges = O(1)).
        var atomsBefore = space.AtomCount;
        space.ObserveContradiction("toad", "thing", 0.2); // t,o,a,d all already present
        _out.WriteLine($"after novel word 'toad': atoms {atomsBefore} -> {space.AtomCount}");
        Assert.Equal(atomsBefore, space.AtomCount);                 // 0 new atoms
        Assert.True(space.CompositeCount > words.Length);            // but a new composite hub exists

        // (4) Law C/S — a NOVEL SENTENCE over EXISTING words adds ZERO new word concepts and ZERO atoms: it is a
        // text hub whose ▷ components are the reused word elements.
        var atomsBefore2 = space.AtomCount;
        var compositesBefore = space.CompositeCount;
        space.ObserveContradiction("cat dog", "dog cat", 0.2);      // both sentences over the existing words cat, dog
        Assert.Equal(new[] { "cat", "dog" }, space.ComponentSymbolsOf("cat dog"));
        Assert.Equal(new[] { "dog", "cat" }, space.ComponentSymbolsOf("dog cat"));
        Assert.Equal(atomsBefore2, space.AtomCount);                // 0 new atoms
        Assert.Equal(compositesBefore + 2, space.CompositeCount);   // exactly the 2 new sentence hubs; words reused
        _out.WriteLine($"after 2 sentences over existing words: atoms unchanged={space.AtomCount}, composites +2={space.CompositeCount}");
    }

    [Fact] // Scaling witness: atoms grow sublinearly (flat) as vocabulary scales — the genesis "frontier grows
           // unbounded" wall, removed. Many random words over a fixed alphabet keep the atom set ≈ constant.
    public void AtomSet_StaysFlat_AsVocabularyScales()
    {
        var space = new DialecticalSpace(ProductionDims.FaceDimension, seed: 7);
        const string alphabet = "abcdefghijklmnopqrstuvwxyz";
        var rng = new System.Random(7);
        for (var n = 0; n < 300; n++)
        {
            var len = 3 + rng.Next(6);
            var w = new string(Enumerable.Range(0, len).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray());
            space.ObserveContradiction(w, "thing", 0.2);
        }
        _out.WriteLine($"composites={space.CompositeCount} atoms={space.AtomCount} (alphabet=26)");
        Assert.True(space.AtomCount <= 26, $"atoms must stay flat at ~alphabet size; got {space.AtomCount}");
        Assert.True(space.CompositeCount > 100, "vocabulary actually grew");
    }
}
