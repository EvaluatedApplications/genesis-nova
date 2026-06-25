using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// DIAGNOSTIC: name-recall works COLD (FactRecallExperiment) but ABSTAINS in a warm grammar-trained space. Reproduce
// cheaply (grammar-only warm-up), assert a fact, then DUMP exactly what is stored ("my name" relations) vs what recall
// does — to find why a just-learned relation does not come back. No hard gate; this prints the evidence.
public sealed class NameRecallDiagnostic
{
    private readonly ITestOutputHelper _out;
    public NameRecallDiagnostic(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task WhereDoesWarmNameRecallBreak()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-nrd-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            var curriculum = new FocusedCurriculum(new ITrainingCurriculum[] { new GrammarCurriculum(trainPerCycle: 128) }.ToList(), masteryBar: 0.9, focusBudget: 1);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = false, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("NRD_SECONDS"), out var ss) ? ss : 120.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 8)).Result; var o = r?.Output?.Trim() ?? ""; _out.WriteLine($"  '{s}' -> '{o}' [{r?.DecisionPath}]"); return o; }
            void DumpRel(string c)
            {
                var (exists, rels) = rt.ProbeRelations(c);
                _out.WriteLine($"  '{c}' exists={exists} relations: " + string.Join(", ", rels.Take(10).Select(r => $"{r.Concept}:{r.Confidence:F2}(obs{r.Obs})")));
            }

            string[] RN = { "NONE", "SUBJECT", "VALUE", "QUERY" };
            _out.WriteLine("── role parse ──");
            _out.WriteLine("  my name is stephen -> " + string.Join(" ", rt.ProbeRoles("my name is stephen").Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));
            _out.WriteLine("  what is my name    -> " + string.Join(" ", rt.ProbeRoles("what is my name").Select(t => $"{t.Token}:{(t.Role is >= 0 and < 4 ? RN[t.Role] : "?")}")));

            _out.WriteLine("── BEFORE assert: what curriculum relations exist on the phrase/noun ──");
            DumpRel("my name"); DumpRel("name");

            _out.WriteLine("── assert + recall ──");
            await P("my name is stephen");

            _out.WriteLine("── AFTER assert: relations on the phrase/noun ──");
            DumpRel("my name"); DumpRel("name");

            var recall = await P("what is my name");
            _out.WriteLine($"\nRESULT: recall='{recall}' (want stephen)");

            // The DEMO end-to-end (nonce-noun curriculum = no pollution; AsCopula = "what is" parses): name the bot AND
            // remember the user, distinct, with the user's facts winning recall.
            await P("your name is rex");
            var mineAgain = await P("what is my name");
            var yours = await P("what is your name");
            _out.WriteLine($"FINAL: my name='{mineAgain}' (want stephen)  your name='{yours}' (want rex)");
            Assert.Contains("stephen", recall, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("stephen", mineAgain, StringComparison.OrdinalIgnoreCase); // naming the bot didn't erase mine
            Assert.Contains("rex", yours, StringComparison.OrdinalIgnoreCase);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
