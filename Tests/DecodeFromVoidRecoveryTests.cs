using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// DECODE-FROM-THE-VOID RECOVERY — the materialised space as a CACHE over the conserved decodable void. Identity lives
/// in the FROZEN face bands [0,416) (a deterministic, invertible codec), so an EVICTED concept's identity is re-derivable
/// from its coordinate; eviction frees only the learned orbital + the store slot, not the recoverable identity (G6 via
/// the latent address). These tests pin <see cref="DialecticalSpace.RecoverFromCoordinate"/>: it re-materialises an
/// evicted concept from its coordinate with the SAME identity + a family-reconstructed orbital, REJECTS noise (the
/// critical guard against materialising junk), is gated off by default, and is reachable through the navigator's landing.
/// </summary>
public sealed class DecodeFromVoidRecoveryTests
{
    private const int Dim = 1024; // production face: address-space layout active (dim >= 512), frozen codec [0,416)
    private readonly ITestOutputHelper _out;
    public DecodeFromVoidRecoveryTests(ITestOutputHelper o) => _out = o;

    // Build a space with a small morpheme FAMILY around "hel" (so a recovered "helix" can land near it), an UNRELATED
    // cluster ("ocean"), and an active target word "helix" whose orbital composes from the known morpheme.
    private static DialecticalSpace BuildFamilySpace()
    {
        var space = new DialecticalSpace(Dim)
        {
            RecoverFromVoid = true,
            GenerativeAtoms = true,          // single words are Object atoms (exact spelling decode) + morpheme composition
            DischargeStalenessWindow = 1_000_000_000, // nothing goes stale on this short clock — eviction is by the cap only
            DischargeInterval = 1_000_000_000,         // never auto-sweep; we drive maintenance explicitly
            MaxActiveConcepts = 0,           // start uncapped
        };
        for (var i = 0; i < 3; i++)
        {
            space.ObserveContradiction("hel", "helium", 0.15);   // the morpheme cluster
            space.ObserveContradiction("helium", "element", 0.15);
            space.ObserveContradiction("ocean", "water", 0.15);  // an unrelated cluster (the far comparison)
            space.ObserveContradiction("water", "sea", 0.15);
        }
        space.Recognize("helix"); // realise the target; ObservationCount 0 ⇒ uniquely lowest utility ⇒ the one the cap evicts
        return space;
    }

    private static double Cosine(double[] a, double[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length && i < b.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    // ── 1. RECOVERY WORKS: evict a word, then re-materialise it from its coordinate with the SAME identity + family orbital.
    [Fact]
    public void RecoverFromCoordinate_rematerialises_evicted_word_with_same_identity_and_family_orbital()
    {
        var space = BuildFamilySpace();
        Assert.True(space.ContainsConcept("helix"), "precondition: target must be active");

        // The recovery COORDINATE = the symbol's deterministic FROZEN identity (pure codec, independent of the live orbital).
        var coord = PlatonicFaceComposer.GetFreshEmbedding("helix", Dim);

        // EVICT it: with the cap one below the active count, the lowest-utility unreferenced concept (helix, obs 0) is dropped.
        space.MaxActiveConcepts = space.NodeCount - 1;
        space.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());
        Assert.False(space.ContainsConcept("helix"), "the target must be evicted from the active store");
        Assert.True(space.ContainsConcept("hel"), "the morpheme family must survive (it is reinforced) to reconstruct from");

        // RECOVER: decode the identity off the coordinate and re-materialise it.
        space.MaxActiveConcepts = 0; // lift the cap so the recovered element isn't instantly re-evicted
        var rec = space.RecoverFromCoordinate(coord);

        Assert.NotNull(rec);
        Assert.Equal("helix", rec!.Symbol);                       // SAME identity recovered
        Assert.True(space.ContainsConcept("helix"), "the recovered concept must be back in the active store");
        Assert.True(space.TryDecodeCoordinate(coord, out var kind, out var decoded, out _) && decoded == "helix",
            "the recovered identity must decode back to the same symbol");
        Assert.True(kind is PlatonicKind.Object or PlatonicKind.Atom or PlatonicKind.None, $"unexpected recovered kind {kind}");

        // ORBITAL reconstructed from the KNOWN morpheme ⇒ a non-empty cloud that lands NEAR its family, not the unrelated cluster.
        var vHelix = space.SemanticVectorOf("helix");
        var vHel = space.SemanticVectorOf("hel");
        var vOcean = space.SemanticVectorOf("ocean");
        Assert.NotNull(vHelix);
        Assert.True(vHelix!.Any(x => Math.Abs(x) > 1e-9), "recovered orbital must be non-empty");
        var nearFamily = Cosine(vHelix, vHel!);
        var nearUnrelated = Cosine(vHelix, vOcean!);
        _out.WriteLine($"recovered 'helix' cos(family 'hel')={nearFamily:F3}  cos(unrelated 'ocean')={nearUnrelated:F3}");
        Assert.True(nearFamily > nearUnrelated, $"recovered orbital not near its family ({nearFamily:F3} ≤ {nearUnrelated:F3})");
    }

    // ── 2. THE GUARD: a NOISE coordinate that does not cleanly decode to a valid identity recovers NOTHING (no junk in the store).
    [Fact]
    public void RecoverFromCoordinate_rejects_noise_and_materialises_nothing()
    {
        var space = new DialecticalSpace(Dim) { RecoverFromVoid = true };
        // seed a couple of real concepts so "materialises nothing" is measured against a non-empty store
        space.ObserveContradiction("apple", "fruit", 0.15);
        var before = space.NodeCount;

        var rng = new Random(20260630);
        for (var trial = 0; trial < 5; trial++)
        {
            var noise = new double[Dim];
            for (var i = 0; i < Dim; i++) noise[i] = (rng.NextDouble() - 0.5) * 0.6; // random, off every clean address
            var rec = space.RecoverFromCoordinate(noise);
            Assert.Null(rec); // the coordinate does not decode to a confident valid identity ⇒ recover nothing
        }
        Assert.Equal(before, space.NodeCount); // the store is untouched — no junk concept was materialised
    }

    // ── 3. GATED OFF by default = byte-identical: even a perfectly valid coordinate recovers nothing when the flag is off.
    [Fact]
    public void RecoverFromCoordinate_is_a_noop_when_the_flag_is_off()
    {
        var space = new DialecticalSpace(Dim); // RecoverFromVoid default false
        var coord = PlatonicFaceComposer.GetFreshEmbedding("zephyrium", Dim); // a clean, decodable latent identity
        Assert.True(space.TryDecodeCoordinate(coord, out _, out var dec, out _) && dec == "zephyrium",
            "sanity: the coordinate IS a clean valid identity");

        var rec = space.RecoverFromCoordinate(coord);
        Assert.Null(rec);                                         // default-off ⇒ no recovery
        Assert.False(space.ContainsConcept("zephyrium"), "nothing must be materialised when recovery is off");
    }

    // ── 4. NAVIGATOR LANDING: a TryLand onto an evicted concept's coordinate recovers + lands on the re-materialised element.
    [Fact]
    public void TryLand_recovers_and_lands_on_an_evicted_concept()
    {
        var space = BuildFamilySpace();
        var coord = PlatonicFaceComposer.GetFreshEmbedding("helix", Dim);

        space.MaxActiveConcepts = space.NodeCount - 1;
        space.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest());
        Assert.False(space.ContainsConcept("helix"), "the target must be evicted before the walk reaches its address");
        space.MaxActiveConcepts = 0;

        var landed = space.TryLand(coord, out var symbol, out _, out _, out _);

        Assert.True(landed, "the walk must land on the decodable coordinate");
        Assert.Equal("helix", symbol);
        Assert.True(space.ContainsConcept("helix"), "landing on the evicted coordinate must have RECOVERED the concept");
    }
}
