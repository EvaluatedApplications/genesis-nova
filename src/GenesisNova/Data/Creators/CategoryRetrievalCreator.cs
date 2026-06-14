using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

/// <summary>
/// FOUNDATIONAL bootstrap lesson: single-answer retrieval (item → category), emitted as BARE pairs
/// ("apple" → "fruit") with NO filler prose.
///
/// The demonstration (Tests/CoreBootstrapTests.cs::Retrieval_CanEmerge_AndProbesDecodeTermination)
/// established WHY bare matters: RelationCreator's filler prompts ("apple is a", "category of apple")
/// diluted the coupling and over-clustered the categories so the relation carrier failed (0/16),
/// whereas the bare regime makes the carrier and first-token retrieval emerge cleanly (16/16) — the
/// same way the bare bidirectional number-word lesson did. This creator IS that proven regime.
///
/// (The remaining defect the demonstration isolates — the decoder emitting the right answer then
/// over-generating sibling members instead of terminating — is a decode-layer issue, NOT a property
/// of this lesson.)
/// </summary>
public sealed class CategoryRetrievalCreator : IExampleCreator
{
    // Item → category reference table (reference data, like RelationCreator's; the NN still LEARNS the
    // mapping from the pairs). Ordered so difficulty-0 covers four members each of four categories.
    private static readonly (string Item, string Category)[] Table =
    [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"),
        ("dog", "animal"), ("cat", "animal"), ("wolf", "animal"), ("bear", "animal"),
        ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
        ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"), ("boat", "vehicle"),
        ("piano", "instrument"), ("drum", "instrument"), ("violin", "instrument"), ("flute", "instrument"),
        ("oak", "tree"), ("pine", "tree"), ("cedar", "tree"), ("maple", "tree"),
    ];

    // "corenova:" prefix marks a CORE tool-training lesson (teaches USING the platonic space — here
    // single-answer retrieval), not an outcome-/answer-producing creator. See NumberWordCreator.
    public string Name => "corenova:retrieval-category";

    // Foundational tier, just above number-word equivalence.
    public int EstimatedComplexity => 10;

    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var slice = SliceForLevel(Math.Max(0, difficulty));
        if (slice.Length == 0 || count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (item, category) = slice[i % slice.Length];
            examples.Add((item, category)); // BARE — item → category, no filler.
        }

        return examples.ToImmutable();
    }

    private static (string Item, string Category)[] SliceForLevel(int level)
    {
        // d0 = 16 (four categories × four members), d1 = 20, d2+ = full table.
        var take = level switch
        {
            0 => 16,
            1 => 20,
            _ => Table.Length,
        };
        return Table.Take(Math.Min(take, Table.Length)).ToArray();
    }
}
