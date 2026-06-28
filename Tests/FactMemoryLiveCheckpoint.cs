using System;
using System.IO;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// Does the live warmed checkpoint remember a name? Loads the gym checkpoint READ-ONLY (so there is NO concurrent gym
// training to erode the fact), teaches + recalls through the real PredictAsync path, and prints the DecisionPath so we
// can see WHERE it breaks: if recall works here but not in the live REPL, the gym's concurrent training is eroding the
// just-taught fact; if it fails here too, the parse/recall itself is broken despite the clean function-word signal.
public sealed class FactMemoryLiveCheckpoint
{
    private readonly ITestOutputHelper _out;
    public FactMemoryLiveCheckpoint(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task DoesItRememberNames_OnTheLiveWarmedCheckpoint()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var gym = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "GenesisNova", "gym");
        Assert.True(File.Exists(Path.Combine(gym, "genesis-nova.autosave.checkpoint.json")), $"no gym checkpoint at {gym}");

        var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 2048, FaceDimensionOverride: 512,
            AutoPersist: false, AutoResume: true, LocalStateDirectory: gym).WithProductionMechanisms();
        var rt = new GenesisEvalAppRuntime(config);

        async Task<(string Out, string Path)> P(string s)
        {
            var r = (await rt.PredictAsync(s, 12)).Result;
            var o = r?.Output?.Trim() ?? ""; var path = r?.DecisionPath ?? "";
            _out.WriteLine($"   '{s}'  ->  '{o}'   [{path}]");
            return (o, path);
        }

        _out.WriteLine("== teach + recall on the live warmed checkpoint (NO concurrent gym training) ==");
        await P("my name is bob");
        var name = await P("what is my name");
        await P("my favorite color is red");
        var color = await P("what is my favorite color");
        _out.WriteLine($"\nVERDICT: name='{name.Out}' [{name.Path}]   color='{color.Out}' [{color.Path}]");
    }
}
