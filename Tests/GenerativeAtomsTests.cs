using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// First brick of the generative substrate (genesis create→decompose→select→store). A token is stored as its OWN biggest-
// chunk atom, then DECOMPOSED on demand into candidate sub-atoms (chars + short n-grams) that COMPETE via relevance-decay:
// a morpheme that recurs across tokens AND contributes to correct answers (e.g. "hel") survives; a one-off chunk discharges.
// So useful granularity is DISCOVERED, not fixed at single-char — and it's language-agnostic (a CJK char is its own atom).
// Gated by GenerativeAtoms; default OFF is the byte-identical legacy eager char-composition.
public sealed class GenerativeAtomsTests
{
    private readonly ITestOutputHelper _out;
    public GenerativeAtomsTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Token_is_its_own_atom_and_useful_morphemes_win_the_competition()
    {
        var space = new DialecticalSpace(faceDimension: 64)
        { GenerativeAtoms = true, DischargeStalenessWindow = 800, DischargeInterval = 500 };

        // A token is stored as its OWN atom (the biggest chunk) — NOT eagerly split into a char-composition.
        space.ObserveContradiction("hello", "greeting", 0.2);
        Assert.Equal(ElementKind.Object, space.GetElement("hello")!.Kind);

        // DECOMPOSE on demand → candidate sub-atoms. "hel" recurs across hello + help; "zeb" is a one-off from zebra.
        space.Decompose("hello"); space.Decompose("help"); space.Decompose("zebra");
        Assert.True(space.ContainsConcept("hel"), "decompose did not surface the candidate morpheme 'hel'");
        Assert.True(space.ContainsConcept("zeb"), "decompose did not surface the one-off chunk 'zeb'");

        // "hel" CONTRIBUTES to correct answers (it helped recognise words — credited via ReinforceEvidence); "zeb" never does.
        for (var i = 0; i < 15; i++) space.ReinforceEvidence(new[] { new PlatonicEvidence("hello", "hel", 1.0, 1) }, success: true);

        // Run the clock far past the bare staleness (unrelated noise) — the sweep discharges what hasn't earned its grace.
        for (var step = 0; step < 4000; step++) space.ObserveContradiction("f" + step, "ctx", 0.2);

        Assert.True(space.ContainsConcept("hel"), "the morpheme that EARNED correct answers was wrongly discharged");
        Assert.False(space.ContainsConcept("zeb"), "the useless one-off chunk was not discharged");
        _out.WriteLine($"nodes={space.NodeCount}  (hel kept by utility, zeb decayed)");
    }

    [Fact]
    public void Novel_word_is_recognised_and_positioned_via_its_known_morpheme()
    {
        // THE PAYOFF — generalisation. Learn a stem from two words, then recognise a word that was NEVER seen, via the
        // morpheme it shares: it lands near its family in meaning space instead of nowhere. That's the point of decompose.
        var space = new DialecticalSpace(faceDimension: 96) { GenerativeAtoms = true, DischargeInterval = 100_000 };
        for (var i = 0; i < 5; i++) { space.ObserveContradiction("hello", "greet", 0.1); space.ObserveContradiction("help", "greet", 0.1); }
        space.Decompose("hello"); space.Decompose("help");          // "hel" learned + related to hello/help
        Assert.True(space.ContainsConcept("hel"));

        var rec = space.Recognize("helix");                          // a NOVEL word, never observed
        Assert.True(rec.KnownComponents > 0, "novel 'helix' shared no known morpheme");
        Assert.False(rec.WholeHit, "a novel word should be COMPOSED from parts, not remembered whole");

        var near = space.GetNearestConcepts("helix", maxNeighbors: 6).Select(n => n.Symbol).ToList();
        _out.WriteLine($"helix: known {rec.KnownComponents}/{rec.TotalComponents}  nearest=[{string.Join(",", near)}]");
        Assert.True(near.Contains("hello") || near.Contains("help") || near.Contains("hel"),
            $"novel 'helix' did not land near its morpheme family — got: [{string.Join(",", near)}]");
    }

    [Fact]
    public void Default_off_is_legacy_eager_char_composition()
    {
        var space = new DialecticalSpace(faceDimension: 64); // GenerativeAtoms default OFF
        space.ObserveContradiction("hello", "greeting", 0.2);
        Assert.Equal(ElementKind.Composition, space.GetElement("hello")!.Kind); // single word → composition over char atoms
        Assert.True(space.AtomCount > 0, "legacy mode should create the char atoms");
    }
}
