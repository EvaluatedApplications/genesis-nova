using System;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// Confirms the corpus warmer actually streams REAL text (hydrates Wikipedia from HuggingFace, caches to disk), so we
// know the warming source works before pointing the gym at it. Hydration runs in the background, so we poll a few
// rounds: early rounds may show the fallback sentences, later rounds should show real Wikipedia windows.
public sealed class CorpusWarmSmoke
{
    private readonly ITestOutputHelper _out;
    public CorpusWarmSmoke(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task CorpusWarmer_StreamsRealText()
    {
        var c = new CorpusWarmCurriculum(trainPerCycle: 32);
        for (var round = 1; round <= 8; round++)
        {
            var batch = c.NextTrainBatch();
            _out.WriteLine($"-- round {round}: {batch.Count} windows --");
            foreach (var (input, output) in batch.Take(3))
                _out.WriteLine($"   in: \"{input[..Math.Min(90, input.Length)]}\"  =>  out: \"{output[..Math.Min(40, output.Length)]}\"");
            await Task.Delay(5000); // let background hydration progress
        }
    }
}
