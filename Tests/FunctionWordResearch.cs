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

// RESEARCH (de-hardcoding the function-word list): degree alone fails (it conflates "the" with popular content like
// "copper"). Hypothesis: a function word's meaning-CLOUD sits near the global centroid (it co-occurs with everything),
// while content words point somewhere specific. This trains a real space and DUMPS degree + cloud-centrality for known
// function vs content words, to see which signal actually separates them. Diagnostic first — measure before building.
public sealed class FunctionWordResearch
{
    private readonly ITestOutputHelper _out;
    public FunctionWordResearch(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task WhichLearnedSignal_SeparatesFunctionFromContent()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-fw-" + DateTime.Now.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(dir);
        var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
            AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
        var rt = new GenesisEvalAppRuntime(config);

        var skills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Multiply, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord, GymSkill.Predicate, GymSkill.WordedAdd };
        var children = skills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 }).ToList();
        var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 6);
        var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
        var seconds = double.TryParse(Environment.GetEnvironmentVariable("FW_SECONDS"), out var ss) ? ss : 90.0;
        using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
            try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

        // Function words appear across the gym's frames + the diversified cruft; content words are the actual
        // cues/answers (incl. POPULAR ones, which degree wrongly flags as hubs).
        var fns = new[] { "what", "is", "a", "of", "the", "for", "kind", "thing", "to", "in",
                          "can", "you", "do", "me", "i", "if", "with", "about", "could", "any" };
        var content = new[] { "apple", "fruit", "big", "large", "huge", "dog", "animal", "greater", "vehicle", "color" };

        var stats = rt.ProbeTokenSignals(fns.Concat(content).ToArray())
            .ToDictionary(s => s.Token, s => s, StringComparer.OrdinalIgnoreCase);
        void Dump(string label, string[] toks)
        {
            _out.WriteLine($"── {label} ──  (token: known degree centrality)");
            foreach (var t in toks)
                if (stats.TryGetValue(t, out var s))
                    _out.WriteLine($"   {t,-10} known={s.Known,-5} deg={s.Degree,-4} centrality={s.Centrality,7:F3}");
        }
        Dump("FUNCTION words", fns);
        Dump("CONTENT words", content);

        var fnKnown = fns.Where(t => stats.TryGetValue(t, out var s) && s.Known).ToArray();
        var coKnown = content.Where(t => stats.TryGetValue(t, out var s) && s.Known).ToArray();
        if (fnKnown.Length > 0 && coKnown.Length > 0)
        {
            _out.WriteLine($"\nDEGREE      function avg {fnKnown.Average(t => stats[t].Degree):F1}   content avg {coKnown.Average(t => stats[t].Degree):F1}");
            _out.WriteLine($"CENTRALITY  function avg {fnKnown.Average(t => stats[t].Centrality):F3}   content avg {coKnown.Average(t => stats[t].Centrality):F3}");
        }
        Assert.True(true); // diagnostic — the numbers above are the result
    }
}
