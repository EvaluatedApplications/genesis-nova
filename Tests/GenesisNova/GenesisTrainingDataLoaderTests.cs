using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class GenesisTrainingDataLoaderTests
{
    [Fact]
    public void WhenLoadingFile_ThenParsesExamplesAndRouteLabels()
    {
        var path = Path.Combine(Path.GetTempPath(), $"genesis-train-{Guid.NewGuid():N}.txt");
        File.WriteAllLines(path,
        [
            "# comment",
            "what is 2+2 => 4 | route=1",
            "say hi => hello"
        ]);

        try
        {
            var examples = GenesisTrainingDataLoader.LoadFromFile(path);
            Assert.Equal(2, examples.Count);
            Assert.Equal(1, examples[0].RouteLabel);
            Assert.Equal("hello", examples[1].Output);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}

