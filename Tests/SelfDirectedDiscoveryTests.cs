using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE SELF STEERS DISCOVERY — the integration point. The searcher (DiscoverComposition) is directed by SALIENCE = what
/// the self attends to. Two functionally-complete primitives (NAND, NOR) can BOTH build AND; the self decides which
/// derivation surfaces. Ablate the self's salience → a different valid derivation → the self is LOAD-BEARING in thinking,
/// not decorative. This is the hook that makes the self the σ of generative thought (feed _selfField's salient concepts
/// in as `salience`); proven the same way LivingSelfExperiment proved the self conditions retrieval.
/// </summary>
public sealed class SelfDirectedDiscoveryTests
{
    private readonly ITestOutputHelper _out;
    public SelfDirectedDiscoveryTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void SelfSalience_DecidesWhichDerivationIsDiscovered()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var ds = new DialecticalSpace(config.FaceDimension, seed: 7);

        // two realised complete primitives (multilinear coeffs c0,c1,c2,c3):
        //   NAND = 1 − a·b ;  NOR = 1 − a − b + a·b
        var primitives = new (string, double[])[]
        {
            ("nand", new double[] { 1, 0, 0, -1 }),
            ("nor",  new double[] { 1, -1, -1, 1 }),
        };
        int[] AND = { 0, 0, 0, 1 };   // target truth table over (0,0),(0,1),(1,0),(1,1)

        var viaNand = ds.DiscoverComposition(primitives, AND, salience: new[] { "nand" });
        var viaNor = ds.DiscoverComposition(primitives, AND, salience: new[] { "nor" });
        _out.WriteLine($"self attends NAND → AND discovered as: {viaNand}");
        _out.WriteLine($"self attends NOR  → AND discovered as: {viaNor}");

        Assert.NotNull(viaNand);
        Assert.NotNull(viaNor);
        Assert.Contains("nand", viaNand!);
        Assert.DoesNotContain("nor", viaNand!);
        Assert.Contains("nor", viaNor!);
        // THE PROOF: the self's salience CHANGES the discovered derivation → the self is load-bearing in thinking.
        Assert.NotEqual(viaNand, viaNor);
        _out.WriteLine("=> the self's salience steered discovery to different derivations (load-bearing).");
    }
}
