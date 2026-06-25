using System;
using System.IO;
using System.Threading.Tasks;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Runtime;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// GENERAL fact memory — NO hardcoding. Teach a fact with an ordinary assertion ("my name is X") and recall it with an
// ordinary question ("what is my name") through the substrate's native learn (TryFieldLearn) + retrieve, exactly the
// same path as any other association. If this works, "remember my name" is just a special case of general fact memory,
// learned, not a coded name routine.
public sealed class FactRecallExperiment
{
    private readonly ITestOutputHelper _out;
    public FactRecallExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_LearnsThenRecalls_AnAssertedFact()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true };
        string Say(string s) { var r = mind.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' → '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        Say("my name is stephen");        // learn it (general assertion path)
        Say("my dog is rex");
        var name = Say("what is my name"); // recall it (general retrieval)
        var dog = Say("what is my dog");

        Assert.Contains("stephen", name, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("rex", dog, StringComparison.OrdinalIgnoreCase);
    }

    // THE DEMO: name the BOT and remember the USER — two facts that share the noun "name" but differ in POSSESSOR
    // ("my name" vs "your name"). They must NOT collide: naming the bot must not erase the user's name. The possessor
    // distinguishes them with NO hardcoded possessive list — the determiner is part of the subject phrase.
    [Fact]
    public void Field_Distinguishes_MyName_From_YourName()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true };
        string Say(string s) { var r = mind.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' → '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        Say("my name is stephen");          // the USER's name
        Say("your name is rex");            // NAME THE BOT (must not overwrite the user's name)
        var mine = Say("what is my name");  // → stephen
        var yours = Say("what is your name"); // → rex

        Assert.Contains("stephen", mine, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("rex", yours, StringComparison.OrdinalIgnoreCase);
    }

    // The actual REPL path: through GenesisEvalAppRuntime.PredictAsync (what the app's REPL calls), production
    // mechanisms on. Confirms it works there too, and probes which PHRASINGS the general path covers.
    [Fact]
    public async Task ReplRuntimePath_LearnsThenRecalls()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-fact-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: ComputeBackend.Cpu, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);
            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 8)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' → '{o}' [{r?.DecisionPath}]"); return o; }

            await P("my name is stephen");                 // learn (copula assertion)
            var q1 = await P("what is my name");           // recall — should be stephen
            var q2 = await P("whats my name");             // contraction-ish phrasing — coverage probe
            var q3 = await P("do you know my name");        // another phrasing — coverage probe

            Assert.Contains("stephen", q1, StringComparison.OrdinalIgnoreCase); // the core path works on the REPL runtime
            _out.WriteLine($"coverage: 'what is my name'={(q1.Contains("stephen", StringComparison.OrdinalIgnoreCase))}  'whats my name'={(q2.Contains("stephen", StringComparison.OrdinalIgnoreCase))}  'do you know my name'={(q3.Contains("stephen", StringComparison.OrdinalIgnoreCase))}");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    // Does a REPL-learned fact SURVIVE a reload? It's written as a RELATION into the model's substrate (not a session
    // variable), so it should be saved in the checkpoint and reloaded. Learn → save → FRESH runtime resumes → recall.
    [Fact]
    public async Task ReplLearnedFact_SurvivesSaveAndReload()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-fact-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            GenesisNovaConfig Cfg() => new GenesisNovaConfig(Backend: ComputeBackend.Cpu, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: true, AutoResume: true, LocalStateDirectory: dir).WithProductionMechanisms();

            var rt1 = new GenesisEvalAppRuntime(Cfg());
            await rt1.PredictAsync("my name is stephen", 8);          // learn it in "session 1"
            await rt1.SaveAsync(rt1.AutoCheckpointPath);              // as an autosave / clean close would

            var rt2 = new GenesisEvalAppRuntime(Cfg());               // "restart" — fresh runtime resumes from disk
            var name = (await rt2.PredictAsync("what is my name", 8)).Result?.Output?.Trim() ?? "";
            _out.WriteLine($"  after reload: 'what is my name' → '{name}'");
            Assert.Contains("stephen", name, StringComparison.OrdinalIgnoreCase); // it PERSISTED across the reload
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
