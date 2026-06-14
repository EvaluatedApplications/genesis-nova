using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// STEP 1 of moving the relation graph INTO the substrate (user: "build the graph inside the platonic
/// space"). A relation is represented as a first-class positioned element (RelationElement: endpoints +
/// strength + centroid embedding). This proves the element-graph retrieval is at PARITY with the legacy
/// dict-backed Relational tier (same neighbour, both directions) and that each relation sits at the
/// centroid of its endpoints — the prerequisite for retiring the _relations dictionary and making
/// relations composable (higher-order). Keeps the relation graph's proven retrieval, just changes its home.
/// </summary>
public sealed class RelationElementTests
{
    private readonly ITestOutputHelper _out;
    public RelationElementTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void RelationElement_RetrievesAtParityWithDict_AndSitsAtCentroid()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);

        var pairs = new[] { ("one", "1"), ("apple", "fruit"), ("dog", "animal") };
        foreach (var (a, b) in pairs)
            for (var i = 0; i < 50; i++)
            {
                memory.ObserveContradiction(a, b, 0.05);
                memory.ObserveContradiction(b, a, 0.05);
            }

        // PARITY: element-graph traversal returns the SAME neighbour as the dict-backed Relational tier,
        // in BOTH directions — so retrieval behaviour is preserved when the graph moves into the substrate.
        foreach (var (a, b) in pairs)
            foreach (var q in new[] { a, b })
            {
                var dict = memory.GetNeighbors(q, PlatonicNeighborhoodType.Relational, maxNeighbors: 1, minConfidence: 0.0);
                Assert.True(dict.Count > 0, $"dict found no relation for '{q}'");
                Assert.True(memory.TryRelationElementNeighbour(q, out var elemNeighbour, out var strength),
                    $"element graph found no relation for '{q}'");
                _out.WriteLine($"'{q}': dict='{dict[0].Concept}'({dict[0].Confidence:F2})  element='{elemNeighbour}'({strength:F2})");
                Assert.Equal(dict[0].Concept, elemNeighbour);
            }

        // POSITION: each relation-element sits at the centroid of its endpoints' faces (a positioned
        // object between them — the basis for composability).
        var rels = memory.GetRelationElements();
        Assert.NotEmpty(rels);
        var oneRel = rels.First(r =>
            (string.Equals(r.Left, "one", StringComparison.OrdinalIgnoreCase) && r.Right == "1") ||
            (r.Left == "1" && string.Equals(r.Right, "one", StringComparison.OrdinalIgnoreCase)));
        Assert.True(memory.TryGetConceptFace(oneRel.Left, out var lf));
        Assert.True(memory.TryGetConceptFace(oneRel.Right, out var rf));
        Assert.Equal(lf.Length, oneRel.Embedding.Length);
        for (var i = 0; i < lf.Length; i++)
            Assert.True(Math.Abs(oneRel.Embedding[i] - 0.5 * (lf[i] + rf[i])) < 1e-9,
                $"relation-element not at centroid at dim {i}");
    }
}
