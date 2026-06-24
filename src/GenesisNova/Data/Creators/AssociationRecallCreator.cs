using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

/// <summary>
/// LEARNABLE benchmark material — a pure associative-recall task that, unlike arithmetic (computed) and number-word
/// (codec), can ONLY be answered by LEARNING. Each entity has an ARBITRARY fixed attribute (no model has a prior for
/// "otter → amber"), taught through MANY varied phrasings. Because a bench splits by unique input string, different
/// phrasings of the SAME entity land in train vs held-out — so held-out is "a SEEN entity in a NEW phrasing", which
/// a relation/cloud retriever genuinely generalises to (vs CategoryRetrievalCreator, whose bare one-phrasing-per-item
/// split puts whole UNSEEN members in held-out → forced abstention, 0% for BOTH models).
///
/// NOT in ExampleCreatorRegistry.All (the production curriculum is untouched) — this is shared so the RaceBench
/// benchmark and the diagnostic tests can both use it.
/// </summary>
public sealed class AssociationRecallCreator : IExampleCreator
{
    // Arbitrary, fixed entity → attribute map. Entities are uncommon animals; attributes a small shared pool
    // (each attribute is a hub of three entities), so retrieval must DISAMBIGUATE by the entity, not guess a prior.
    private static readonly string[] Entities =
    {
        "otter","walrus","badger","heron","marten","lynx","raven","finch","perch","gecko","newt","toad",
        "hare","mole","vole","stoat","weasel","ferret","beaver","marmot","bison","moose","quail","crane",
    };
    private static readonly string[] Attributes = { "amber", "indigo", "copper", "jade", "onyx", "coral", "ivory", "slate" };
    private static string Attr(int i) => Attributes[i % Attributes.Length];

    // Varied phrasings around the entity. Framing words (what/is/the/of/about) are filtered by the field, so the
    // entity is the discriminative anchor in every one — the attribute is retrieved by its learned relation.
    private static readonly System.Func<string, string>[] Phrasings =
    {
        e => $"what is {e}",
        e => $"{e} is",
        e => $"tell me about {e}",
        e => $"describe {e}",
        e => $"{e} goes with",
        e => $"the trait of {e}",
        e => $"what about {e}",
        e => $"{e} pairs with",
    };

    public string Name => "race:association-recall";
    public int EstimatedComplexity => 12;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    private static readonly GradingPolicy GradingPolicyValue = new(RequirePlatonic: true, AnswerVocabulary: Attributes);
    public GradingPolicy Grading => GradingPolicyValue;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var nEntities = System.Math.Max(0, difficulty) switch { 0 => 12, 1 => 18, _ => Entities.Length };
        nEntities = System.Math.Min(nEntities, Entities.Length);
        if (count <= 0 || nEntities == 0) return ImmutableArray<(string, string)>.Empty;

        // Emit EVERY (entity, phrasing) → attribute. A bench dedups by input and splits, distributing phrasings of
        // each entity across train/held. (forTraining is intentionally ignored — the bench owns the split.)
        var b = ImmutableArray.CreateBuilder<(string, string)>();
        for (var e = 0; e < nEntities; e++)
            foreach (var phrase in Phrasings)
                b.Add((phrase(Entities[e]), Attr(e)));

        if (b.Count == 0) return ImmutableArray<(string, string)>.Empty;
        if (b.Count >= count) return b.ToImmutable();
        var outb = ImmutableArray.CreateBuilder<(string, string)>(count);
        for (var i = 0; i < count; i++) outb.Add(b[i % b.Count]);
        return outb.ToImmutable();
    }
}
