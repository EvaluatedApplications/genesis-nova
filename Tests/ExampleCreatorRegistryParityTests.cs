using GenesisNova.Data;

namespace GenesisNova.Tests;

public sealed class ExampleCreatorRegistryParityTests
{
    [Fact]
    public void WhenRegistryLoaded_ThenContainsLegacySayExactCreator()
    {
        Assert.Contains(ExampleCreatorRegistry.All, c => c.Name == "say:exact");
    }
}
