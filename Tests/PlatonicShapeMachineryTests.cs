using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Fast (no-NN) tests for the ELEMENT-NATIVE shape machinery behind the Seq and Ref glider skills — the parts
/// not covered by the block-level <see cref="PlatonicGliderDemoTests"/> or the (slow) composer-emergence tests:
///  • the CHUNK-ELEMENT STORE (Seq scaffolds mined from graded-correct outputs) and its persistence,
///  • the FUNCTION-ELEMENT registry (shapes as <see cref="ElementKind.Function"/> elements of the space),
///  • the <see cref="PlatonicShapeRegistry"/> and the interpreter resolving Ref recursively against it.
/// These run on the real substrate at production face dimension; deterministic; fast.
/// </summary>
public sealed class PlatonicShapeMachineryTests
{
    private readonly ITestOutputHelper _out;
    public PlatonicShapeMachineryTests(ITestOutputHelper o) => _out = o;

    private static PlatonicSpaceMemory Space()
        => new(faceDimension: ProductionDims.FaceDimension, seed: 7);

    // ── Chunk-element store (Seq scaffold mining) ───────────────────────────────────────────────────────

    [Fact] // MineChunk reinforces; TryGetTopChunk returns the most-reinforced scaffold (competition by count).
    public void ChunkStore_MinesAndRetrieves_MostReinforcedScaffold()
    {
        var space = Space();
        Assert.False(space.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out _)); // nothing mined yet → abstain

        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "the answer is");
        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "the answer is");
        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "result");

        Assert.True(space.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out var top));
        Assert.Equal("the answer is", top); // 2 reinforcements beats 1

        // A different tag is isolated — the store is keyed.
        Assert.False(space.TryGetTopChunk("⟨unrelated⟩", out _));
    }

    [Fact] // The mined chunk store survives an ExportSnapshot → ImportSnapshot round-trip (checkpoint persistence).
    public void ChunkStore_PersistsThrough_SnapshotRoundTrip()
    {
        var space = Space();
        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "the answer is");
        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "the answer is");

        var snapshot = space.ExportSnapshot();
        Assert.Contains(snapshot.Chunks ?? Array.Empty<PlatonicChunkSnapshot>(),
            c => c.Tag.Contains("seq", StringComparison.OrdinalIgnoreCase) && c.Chunk == "the answer is" && c.Count == 2);

        var restored = Space();
        restored.ImportSnapshot(snapshot);
        Assert.True(restored.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out var top));
        Assert.Equal("the answer is", top);
    }

    [Fact] // Import is ADDITIVE: a chunk-less snapshot (what ApplyMaintenance rebuilds from) never wipes chunks.
    public void ChunkStore_Import_IsAdditive_SoMaintenanceNeverWipesIt()
    {
        var space = Space();
        space.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, "the answer is");

        // A maintenance-style snapshot carries nodes/relations but NO chunks.
        var maintenanceSnapshot = new PlatonicMemorySnapshot(
            FaceDimension: space.FaceDimension,
            Nodes: Array.Empty<PlatonicNodeSnapshot>(),
            Relations: Array.Empty<PlatonicRelationSnapshot>());
        space.ImportSnapshot(maintenanceSnapshot);

        Assert.True(space.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out var top));
        Assert.Equal("the answer is", top); // preserved, not wiped
    }

    // ── Function-element registry (shapes as ElementKind.Function) ──────────────────────────────────────

    [Fact] // A registered shape is a positioned Kind=Function element whose RelatedTo points at its sub-shapes.
    public void FunctionElement_Registers_AsFunctionKind_WithReferences_AndIsIdempotent()
    {
        var space = Space();
        var nodesBefore = space.NodeCount;

        var larger = space.RegisterFunctionElement("larger");
        var twice = space.RegisterFunctionElement("twicelarger", new[] { "larger" });

        Assert.Equal(ElementKind.Function, larger.Kind);
        Assert.Equal(ElementKind.Function, twice.Kind);
        Assert.Equal("shape:function", twice.GenerationPath);
        Assert.Contains(larger.Id, twice.RelatedTo); // the Ref shape references the sub-shape element

        Assert.True(space.TryGetFunctionElement("twicelarger", out var fetched));
        Assert.Equal(twice.Id, fetched.Id);

        // Idempotent: re-registering returns the same element, adds nothing.
        var count = space.FunctionElements.Count;
        var again = space.RegisterFunctionElement("larger");
        Assert.Equal(larger.Id, again.Id);
        Assert.Equal(count, space.FunctionElements.Count);

        // Zero concept-node pollution: Function elements live in their own index, not _nodes.
        Assert.Equal(nodesBefore, space.NodeCount);
    }

    // ── PlatonicShapeRegistry + interpreter Ref resolution (the Ref skill end-to-end on the substrate) ───

    [Fact] // The registry seeds the named shapes AND materializes them as Function elements in the space.
    public void ShapeRegistry_SeedsShapes_AndMaterializesFunctionElements()
    {
        var space = Space();
        var registry = new PlatonicShapeRegistry(space);

        Assert.True(registry.TryGet(PlatonicShapeRegistry.LargerShape, out _));
        Assert.True(registry.TryGet(PlatonicShapeRegistry.TwiceLargerShape, out _));

        Assert.True(space.TryGetFunctionElement(PlatonicShapeRegistry.LargerShape, out var lg));
        Assert.True(space.TryGetFunctionElement(PlatonicShapeRegistry.TwiceLargerShape, out var tl));
        Assert.Equal(ElementKind.Function, tl.Kind);
        Assert.Contains(lg.Id, tl.RelatedTo); // twicelarger references larger
    }

    [Fact] // Ref skill end-to-end: the interpreter resolves Ref against the registry library and computes
           // 2·max(a,b) on the substrate — generalising to held-out operands (no manual library, no NN).
    public void ShapeRegistry_Interpreter_ResolvesRef_And_ComputesTwiceLarger()
    {
        var space = Space();
        var registry = new PlatonicShapeRegistry(space);
        var interp = new PlatonicGliderInterpreter(space, registry.Library);

        Assert.True(registry.TryGet(PlatonicShapeRegistry.TwiceLargerShape, out var shape));

        Assert.Equal("14", interp.Execute(shape, new[] { "3", "7" })); // 2·max(3,7)
        Assert.Equal("16", interp.Execute(shape, new[] { "8", "5" })); // 2·max(8,5) — held-out
        Assert.Equal("18", interp.Execute(shape, new[] { "2", "9" })); // 2·max(2,9) — held-out

        _out.WriteLine("Ref via registry: twicelarger -> 14/16/18 on the substrate (recursive shape-of-shapes).");
    }
}
