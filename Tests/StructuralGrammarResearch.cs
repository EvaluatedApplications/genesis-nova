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

// RESEARCH (de-hardcoding the GRAMMAR: copula/question/possessive word-lists → LEARNED roles). The hypothesis the
// project rests on: a token's ROLE is distributional. A copula ("is"), a question cue ("what"), a possessive ("my")
// are FUNCTION words that, across many frames, co-occur with everything → their cloud collapses toward the centroid,
// exactly like the filler we already learned (FunctionWordResearch). The DECISIVE test of "learned, not listed": a
// NONCE copula ("ploo") and NONCE possessive ("blorp"), seen ONLY in the structural slot across varied vocabulary and
// in NO code list, must land in the SAME filler band — i.e. the model captures the ROLE, not the specific word.
//
// This trains a small STRUCTURAL-GRAMMAR curriculum (words live in the DATA, not src/) and dumps centrality for the
// structural tokens vs content. Measure before building the learned parser.
public sealed class StructuralGrammarResearch
{
    private readonly ITestOutputHelper _out;
    public StructuralGrammarResearch(ITestOutputHelper o) => _out = o;

    // A curriculum of assert/recall FRAMES with deliberately VARIED grammar tokens — so no single copula/possessive/
    // query word is a constant correlate (the bare>consistent-filler lesson). Includes NONCE structural tokens that
    // exist nowhere in the code, to test whether the ROLE (not the word) is what's learned.
    private sealed class StructuralGrammarCurriculum : ITrainingCurriculum
    {
        private readonly Random _rng = new(11);
        private static readonly string[] Possessives = { "my", "your", "his", "her", "our", "their", "blorp" }; // blorp = NONCE
        private static readonly string[] Copulas = { "is", "was", "ploo" };                                      // ploo = NONCE
        private static readonly string[] QueryCues = { "what", "whats", "tell me", "remind me of", "who is" };
        private static readonly string[] Nouns = { "name", "dog", "car", "job", "city", "color", "food", "book", "song", "friend", "team", "drink" };
        private static readonly string[] Values = { "alex", "rex", "blue", "pizza", "lisbon", "coder", "tea", "novel", "anthem", "sam", "rovers", "cola" };

        public string Name => "grammar";
        public int Difficulty => 1;
        private string P() => Possessives[_rng.Next(Possessives.Length)];
        private string N() => Nouns[_rng.Next(Nouns.Length)];
        private string V() => Values[_rng.Next(Values.Length)];

        public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
        {
            var b = new List<(string, string)>(64);
            for (var i = 0; i < 64; i++)
            {
                var poss = P(); var noun = N(); var value = V();
                if (i % 2 == 0) // ASSERTION: "<poss> <noun> <copula> <value>"  → the value is the last token
                    b.Add(($"{poss} {noun} {Copulas[_rng.Next(Copulas.Length)]} {value}", value));
                else            // RECALL: "<querycue> <poss> <noun>"  → its value (warming the query frame)
                    b.Add(($"{QueryCues[_rng.Next(QueryCues.Length)]} {poss} {noun}", value));
            }
            return b;
        }

        public IReadOnlyList<TrainingProbe> NextProbes()
            => Enumerable.Range(0, 12).Select(_ =>
                new TrainingProbe($"what {P()} {N()}", new[] { V() }, RequiredDepth: 1, RequirePlatonic: false)).ToList();

        public void RecordCycle(CycleGrade grade) { }
    }

    [SlowFact]
    public async Task DoStructuralRoles_EmergeDistributionally_IncludingNonce()
    {
        var backend = Environment.GetEnvironmentVariable("GYM_GPU") == "1" ? ComputeBackend.Gpu : ComputeBackend.Cpu;
        var dir = Path.Combine(Path.GetTempPath(), "gn-gram-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(Backend: backend, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoPersist: false, AutoResume: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var rt = new GenesisEvalAppRuntime(config);

            // MIX the grammar curriculum with the DIVERSE gym (like the real app) — centrality needs context diversity
            // for a filler word to collapse toward the global centroid (a grammar-only space is too narrow, measured).
            var gymSkills = new[] { GymSkill.Add, GymSkill.Subtract, GymSkill.Synonym, GymSkill.Category, GymSkill.NumberWord, GymSkill.Predicate, GymSkill.WordedAdd };
            var children = gymSkills.Select(s => (ITrainingCurriculum)new GymTrainer(1, 7, new[] { s }) { MasteryBar = 0.9, TrainPerCycle = 48 })
                .Append(new StructuralGrammarCurriculum()).ToList();
            var curriculum = new FocusedCurriculum(children, masteryBar: 0.9, focusBudget: 6);
            var opt = new GenesisModularTrainingOrchestrator.Options { MasteryBar = 0.9, WorkDir = dir, AutosaveSeconds = 0, TrainOnFailureOnly = true, ThrottlePercent = () => 0 };
            var seconds = double.TryParse(Environment.GetEnvironmentVariable("GRAM_SECONDS"), out var ss) ? ss : 90.0;
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(seconds)))
                try { await new GenesisModularTrainingOrchestrator().RunAsync(rt, curriculum, opt, _ => { }, cts.Token); } catch (OperationCanceledException) { }

            // Structural tokens (incl. NONCE ploo/blorp) vs content. If the role is distributional, structural tokens
            // — known AND nonce — read HIGH centrality (filler band), content reads LOW.
            var structural = new[] { "is", "was", "ploo", "my", "your", "blorp", "what", "whats", "who" };
            var content = new[] { "name", "dog", "city", "color", "alex", "rex", "blue", "pizza" };
            var stats = rt.ProbeTokenSignals(structural.Concat(content).ToArray())
                .ToDictionary(s => s.Token, s => s, StringComparer.OrdinalIgnoreCase);

            void Dump(string label, string[] toks)
            {
                _out.WriteLine($"── {label} ──");
                foreach (var t in toks)
                    if (stats.TryGetValue(t, out var s))
                        _out.WriteLine($"   {t,-8} known={s.Known,-5} deg={s.Degree,-4} centrality={s.Centrality,7:F3}");
            }
            Dump("STRUCTURAL (is/was/ploo copulas, my/your/blorp possessives, what/whats/who queries)", structural);
            Dump("CONTENT (nouns + values)", content);

            var sk = structural.Where(t => stats.TryGetValue(t, out var s) && s.Known).ToArray();
            var ck = content.Where(t => stats.TryGetValue(t, out var s) && s.Known).ToArray();
            if (sk.Length > 0 && ck.Length > 0)
                _out.WriteLine($"\nCENTRALITY  structural avg {sk.Average(t => stats[t].Centrality):F3}   content avg {ck.Average(t => stats[t].Centrality):F3}");
            _out.WriteLine($"NONCE check: ploo known={(stats.TryGetValue("ploo", out var pp) && pp.Known)} cen={(stats.TryGetValue("ploo", out var p2) ? p2.Centrality : 0):F3}   blorp known={(stats.TryGetValue("blorp", out var bb) && bb.Known)} cen={(stats.TryGetValue("blorp", out var b2) ? b2.Centrality : 0):F3}");
            Assert.True(true); // diagnostic — the numbers are the result
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
