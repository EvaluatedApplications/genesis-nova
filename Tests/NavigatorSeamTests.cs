using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// STEP 1 of the NN-navigator (PLATONIC_NAVIGATOR.md §5.1/§5.2/§9): the SUBSTRATE MOTION PRIMITIVES a future walker
/// steps with — proven in isolation, NO NN / policy / training. Each test exercises one seam on a fresh
/// <see cref="DialecticalSpace"/> at the address-space dim (512):
///   • TryLand        — "the lattice lands the step": decode-first on a clean address, else snap to the nearest real coord.
///   • Materialise    — the genesis-tick write-path: a passed-through latent coordinate becomes a realised element.
/// </summary>
public sealed class NavigatorSeamTests
{
    private const int Dim = 1024; // production face: address-space layout active (dim >= 512), orbital [416,1024)=608
    private readonly ITestOutputHelper _out;
    public NavigatorSeamTests(ITestOutputHelper o) => _out = o;

    // 1. TryLand DECODE-FIRST onto a LATENT number — a COMPUTE-jump to GetFreshNumericEmbedding(141) lands on "141"
    //    with zero index cost, in an EMPTY space, WITHOUT storing it.
    [Fact]
    public void TryLand_DecodeFirst_LatentNumber_LandsWithoutStoring()
    {
        var space = new DialecticalSpace(Dim);
        var target = PlatonicFaceComposer.GetFreshNumericEmbedding(141, Dim);

        var ok = space.TryLand(target, out var symbol, out var kind, out var landed, out var conf);
        _out.WriteLine($"[1] TryLand(141): ok={ok} symbol='{symbol}' kind={kind} conf={conf:F4} landedLen={landed.Length}");

        Assert.True(ok, "a clean numeric address must land");
        Assert.Equal(141.0, double.Parse(symbol, System.Globalization.CultureInfo.InvariantCulture), 3);
        Assert.True(conf > 0.55, $"decode-first confidence should be high, got {conf:F4}");
        Assert.NotEmpty(landed);
        Assert.False(space.ContainsConcept("141"), "TryLand must NOT store the landing (pure read)");
    }

    // 2. TryLand SNAP — a coordinate perturbed OFF the frozen address of a stored word snaps back to that word via the
    //    lattice harvest + FrozenIdentityDistance rescore.
    [Fact]
    public void TryLand_Snap_PerturbedWord_SnapsBackToStoredWord()
    {
        var space = new DialecticalSpace(Dim);
        space.Materialise("apple");
        space.Materialise("banana");
        space.Materialise("cherry");

        Assert.True(space.TryGetConceptFace("apple", out var appleFace));
        var target = (double[])appleFace.Clone();
        // Knock the coordinate OFF-address by perturbing the FROZEN spelling band (within [KindStart,OrbitalStart)):
        // decode-first confidence drops below the land threshold, forcing the SNAP branch.
        for (var i = 60; i < 180; i++) target[i] += 0.12;

        var ok = space.TryLand(target, out var symbol, out var kind, out var landed, out var conf);
        _out.WriteLine($"[2] TryLand(apple+noise): ok={ok} symbol='{symbol}' kind={kind} conf={conf:F4}");

        Assert.True(ok, "an off-address coordinate must snap to a real concept");
        Assert.Equal("apple", symbol);
        Assert.True(conf is > 0.0 and < 1.0, $"snap confidence is a distance score in (0,1), got {conf:F4}");
        Assert.NotEmpty(landed);
    }

    // 3. Materialise — the genesis-tick write-path turns a LATENT word coordinate into a REALISED element, and it then
    //    shows up in the navigation sensor of a frozen-near sibling.
    [Fact]
    public void Materialise_LatentCoordinate_BecomesRealised_AndVisibleToNeighbour()
    {
        var space = new DialecticalSpace(Dim);
        space.Materialise("wig");          // a frozen-near sibling (one char from "wug")
        space.Materialise("elephant");     // a frozen-far decoy

        var latent = PlatonicFaceComposer.GetFreshEmbedding("wug", Dim);
        Assert.False(space.ContainsConcept("wug"), "precondition: 'wug' must be latent (never stored)");

        space.Materialise(latent);
        _out.WriteLine($"[3] after Materialise(wug): ContainsConcept(wug)={space.ContainsConcept("wug")}");
        Assert.True(space.ContainsConcept("wug"), "Materialise must realise the decoded symbol");

        var near = space.Neighborhood("wig", 8);
        _out.WriteLine($"[3] Neighborhood(wig)=[{string.Join(", ", near.Select(n => $"{n.Symbol}:{n.Distance:F3}"))}]");
        Assert.Contains(near, n => n.Symbol == "wug");
    }
}
