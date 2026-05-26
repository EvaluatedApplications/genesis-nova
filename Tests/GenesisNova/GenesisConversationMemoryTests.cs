using GenesisNova.Runtime;

namespace GenesisNova.Tests;

public sealed class GenesisConversationMemoryTests
{
    [Fact]
    public void WhenTurnsExceedCapacity_ThenConversationCompacts()
    {
        var memory = new GenesisConversationMemory();

        for (var i = 0; i < 20; i++)
            memory.ObserveTurn("user", $"turn {i}");

        Assert.True(memory.CompactionCount > 0);
        Assert.True(memory.RecentTurns.Count <= 12);
        Assert.Contains("turn 0", memory.Summary);
    }

    [Fact]
    public void WhenResetSignalObserved_ThenTrustDropsAndResetIsTracked()
    {
        var memory = new GenesisConversationMemory();

        memory.ObserveTurn("user", "this branch is wrong", resetSignal: true, note: "user reset");

        Assert.Equal(1, memory.ResetCount);
        Assert.True(memory.BranchTrust < 1.0);
        Assert.NotNull(memory.LastResetAtUtc);
        Assert.Contains("reset signal", memory.Summary);
    }

    [Fact]
    public void WhenSnapshotRoundTrips_ThenRecentTurnsAndSummaryPersist()
    {
        var memory = new GenesisConversationMemory();
        memory.ObserveTurn("user", "hello there");
        memory.ObserveTurn("assistant", "general kenobi");
        memory.Compact();

        var snapshot = memory.ExportSnapshot();
        var restored = new GenesisConversationMemory();
        restored.ImportSnapshot(snapshot);

        Assert.Equal(memory.ResetCount, restored.ResetCount);
        Assert.Equal(memory.CompactionCount, restored.CompactionCount);
        Assert.Equal(memory.Summary, restored.Summary);
        Assert.Equal(memory.RecentTurns.Count, restored.RecentTurns.Count);
    }
}
