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

// EXPERIMENT (de-hardcoding #5, evidence-first): the engine's number<->word answer is a hardcoded codec
// (NumberWordVocabulary). Number RELATION EDGES are forbidden (they pollute), so a learned replacement must live in the
// MODEL (decoder / face geometry). Before building anything: TRAIN the NumberWord skill, then turn the codec OFF
// (LearnedNumberWordsOnly) and MEASURE what the trained substrate recovers on its own — split ATOMS (0-19, tens,
// hundred) from COMPOSED (47, 147, "forty seven") and both DIRECTIONS. Where it breaks tells us the mechanism:
//   - atoms recover, composition fails  -> build a universal place-value COMPOSER over learned atoms (path B)
//   - atoms fail                         -> the decoder/faces don't hold it; need more training or a face mechanism
public sealed class NumberWordLearningExperiment
{
    private readonly ITestOutputHelper _out;
    public NumberWordLearningExperiment(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public async Task LearnedNumberWords_RecoveryWithoutCodec()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-nw-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // Train number<->word (both directions) with a couple of skills for GRU feature richness. The gym/grader use
            // the codec as GROUND TRUTH (allowed); only the ENGINE's inference codec is what we measure-without.
            var children = new[] { GymSkill.NumberWord, GymSkill.Add, GymSkill.Synonym }
                .Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 }).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 3);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("NW_SECONDS"), out var ss) ? ss : 200.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            // CODEC OFF — number<->word now must come from what the MODEL learned.
            rt.State.Inference.LearnedNumberWordsOnly = true;

            async Task<string> P(string s) { var r = (await rt.PredictAsync(s, 12)).Result; return r?.Output?.Trim() ?? ""; }
            bool Eq(string g, string w) => AnswerEquivalence.Equivalent(g, w);

            async Task<double> Rate(string label, IEnumerable<(string Prompt, string Want)> probes)
            {
                int hit = 0, n = 0; var shown = 0;
                foreach (var (prompt, want) in probes)
                {
                    var got = await P(prompt); n++; var ok = Eq(got, want); if (ok) hit++;
                    if (shown++ < 4) _out.WriteLine($"   [{label}] '{prompt}' -> '{got}' (want '{want}') {(ok ? "OK" : "MISS")}");
                }
                var r = n == 0 ? 0 : hit / (double)n;
                _out.WriteLine($"  {label,-22} {r,6:P0}  ({hit}/{n})");
                return r;
            }

            string W(long v) => NumberWordVocabulary.ToWords(v); // ground truth for grading only
            var atomVals = new long[] { 0, 1, 5, 9, 12, 15, 20, 40, 70, 90, 100 };
            var compVals = new long[] { 21, 47, 88, 113, 147, 250, 999, 1234 };

            _out.WriteLine("== DIGIT -> WORD (decoder generates the word; codec off) ==");
            var atomTW = await Rate("atom digit->word", atomVals.Select(v => (v.ToString() + " in words", W(v))));
            var compTW = await Rate("composed digit->word", compVals.Select(v => (v.ToString() + " in words", W(v))));

            _out.WriteLine("== WORD -> DIGIT (model parses the word to a value; codec off) ==");
            var atomWD = await Rate("atom word->digit", atomVals.Select(v => (W(v) + " as a number", v.ToString())));
            var compWD = await Rate("composed word->digit", compVals.Select(v => (W(v) + " as a number", v.ToString())));

            _out.WriteLine($"\nSUMMARY  digit->word atom={atomTW:P0} composed={compTW:P0}  |  word->digit atom={atomWD:P0} composed={compWD:P0}");
            _out.WriteLine("READ: atoms-recover + composition-fails => build a place-value composer over learned atoms;");
            _out.WriteLine("      atoms-fail => the decoder/faces don't hold it (need a face mechanism / more training).");
            Assert.True(true); // diagnostic — the rates are the result
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
