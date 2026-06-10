using GenesisNova.Data;

namespace GenesisNova.Tests;

public sealed class ExampleCreatorRegistryParityTests
{
    [Fact]
    public void WhenRegistryLoaded_ThenDefaultSetContainsPublicCorpusCreatorsOnly()
    {
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:fineweb-edu");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:slimpajama");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:gutenberg");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:openwebmath");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:gsm8k");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "public:wikidata-triples");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "math:fractions");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "math:percent");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "math:ratio");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "math:algebra-solve");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "math:geometry");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "logic:boolean");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "logic:implication");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "logic:quantifiers");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "logic:ordering");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "logic:syllogism");

        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "arithmetic:add");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "numeric:compare");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "sequence:next");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "relation:category");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "language:words");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "say:word");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "say:exact");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "jeopardy:trivia");
    }

    [Fact]
    public void WhenRegistryLoaded_ThenLegacySetRetainsClassicCreatorsButStaysHidden()
    {
        Assert.Contains(ExampleCreatorRegistry.Legacy, c => c.Name == "arithmetic:add");
        Assert.Contains(ExampleCreatorRegistry.Legacy, c => c.Name == "numeric:compare");
        Assert.Contains(ExampleCreatorRegistry.Legacy, c => c.Name == "sequence:next");
        Assert.Contains(ExampleCreatorRegistry.Legacy, c => c.Name == "relation:category");
        Assert.DoesNotContain(ExampleCreatorRegistry.Legacy, c => c.Name == "public:fineweb-edu");
    }
}
