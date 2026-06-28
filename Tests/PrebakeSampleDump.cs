using System.Linq;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// Print sample generated frames per level so the synthetic English can be eyeballed for parsability/plausibility.
// Fast (no training). Run with detailed verbosity to read the output.
public sealed class PrebakeSampleDump
{
    private readonly ITestOutputHelper _out;
    public PrebakeSampleDump(ITestOutputHelper o) => _out = o;

    [Fact]
    public void DumpSamplesPerLevel()
    {
        var c = new PrebakeLanguageCurriculum(seed: 3);
        var names = new[] { "", "L1 function-words", "L2 predication/SVO", "L3 questions", "L4 modification", "L5 discourse" };
        for (var lvl = 1; lvl <= 5; lvl++)
        {
            _out.WriteLine($"\n===== {names[lvl]} =====");
            foreach (var (input, output) in c.SampleLevel(lvl, 12))
                _out.WriteLine($"   \"{input}\"   => {output}");
        }
    }
}
