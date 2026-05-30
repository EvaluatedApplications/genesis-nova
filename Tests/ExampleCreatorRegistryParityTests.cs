using GenesisNova.Data;

namespace GenesisNova.Tests;

public sealed class ExampleCreatorRegistryParityTests
{
    [Fact]
    public void WhenRegistryLoaded_ThenContainsProductionCreatorsAndExcludesLegacyEchoCreators()
    {
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "arithmetic:add");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "numeric:compare");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "sequence:next");
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "relation:category");

        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "language:words");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "say:word");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "say:exact");
        Assert.DoesNotContain(ExampleCreatorRegistry.All, c => c.Name == "jeopardy:trivia");
    }
}
