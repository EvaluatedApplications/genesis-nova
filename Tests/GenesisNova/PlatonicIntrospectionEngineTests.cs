using GenesisNova.Cognition;

namespace GenesisNova.Tests;

public sealed class PlatonicIntrospectionEngineTests
{
    [Fact]
    public void WhenTigerIntroduced_ThenCatDogSynthesisRemainsHigh()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 16, seed: 123);
        var cognition = new PlatonicIntrospectionEngine(memory);

        cognition.ObserveDirectContradiction("cat", "dog", 0.95);
        cognition.ObserveDirectContradiction("cat", "tiger", 0.10);
        cognition.ObserveDirectContradiction("dog", "tiger", 0.85);

        var catDog = memory.GetContradiction("cat", "dog");
        var catTiger = memory.GetContradiction("cat", "tiger");
        var dogTiger = memory.GetContradiction("dog", "tiger");

        Assert.True(catTiger < dogTiger);
        Assert.True(catDog > 0.70);
    }

    [Fact]
    public void WhenInferenceQueued_ThenIntrospectionProcessesFromQueue()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 12, seed: 42);
        var cognition = new PlatonicIntrospectionEngine(memory);

        cognition.QueueInference("cat and dog are different", "they are opposites", routeId: 1, confidence: 0.8);
        cognition.QueueInference("tiger is similar to cat", "close family", routeId: 2, confidence: 0.7);

        var processed = cognition.RunCycles(2);

        Assert.Equal(2, processed);
        Assert.True(cognition.QueueSize == 0);
        Assert.True(memory.NodeCount >= 3);
        Assert.True(memory.RelationCount >= 2);
    }
}

