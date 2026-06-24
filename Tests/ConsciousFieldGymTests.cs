using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE HONEST MEASUREMENT: conscious-field cognition against the REAL gym — the gym's OWN 13 skills, its OWN
/// held-out probe generator (<see cref="GymTrainer.NextProbes"/>), and its OWN value-aware grader
/// (<see cref="GenesisGrader"/>). No hand-picked easy examples. The arithmetic / predicate / number-word skills
/// need NO training (the homomorphism + the linguistic codec); only the two retrieval skills (Synonym, Category)
/// are populated by observing the gym's facts (no GRU training — the field bypasses the classifier). Prints a
/// per-skill breakdown so we see exactly what the field can and cannot do. Production face dim.
/// </summary>
public sealed class ConsciousFieldGymTests
{
    private readonly ITestOutputHelper _out;
    public ConsciousFieldGymTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_OnRealGym_PerSkillBreakdown()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };

        static IReadOnlyList<string> Words(string s) =>
            s.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var retrieval = new HashSet<GymSkill> { GymSkill.Synonym, GymSkill.Category };
        var results = new List<(GymSkill Skill, double Acc, int N)>();
        var totHit = 0; var totN = 0;

        foreach (var skill in Enum.GetValues<GymSkill>())
        {
            var gym = new GymTrainer(startLevel: 1, seed: 12345, skills: new[] { skill }) { ProbeCount = 24 };

            // Retrieval skills learn their facts by OBSERVATION of the CLEAN knowledge (member↔member, item↔category)
            // — NOT framed sentences (relating framing words to answers drowns the content signal). This is the
            // knowledge the gym's discriminative coupling learns; the field then reasons over it. No GRU training.
            if (skill == GymSkill.Synonym)
                for (var k = 0; k < 4; k++)
                    foreach (var group in GymLanguageFacts.SynonymGroups)
                        for (var a = 0; a < group.Length; a++)
                            for (var b = a + 1; b < group.Length; b++)
                                space.FineEditFromExample(new[] { group[a] }, new[] { group[b] }, isNegativeExample: false);
            if (skill == GymSkill.Category)
                for (var k = 0; k < 4; k++)
                    foreach (var (item, cat) in GymLanguageFacts.Categories)
                        space.FineEditFromExample(new[] { item }, new[] { cat }, isNegativeExample: false);

            var probes = gym.NextProbes();
            var hit = 0;
            var shown = 0;
            foreach (var p in probes)
            {
                var res = infer.Generate(new GenerationRequest(p.Query, 16));
                var q = GenesisGrader.Quality(res.Output ?? string.Empty, p.Allowed, p.RequiredDepth,
                    res.UsedNeuralFallback, requirePlatonic: false, p.AnswerVocabulary, p.SurfaceStrict);
                if (q >= 0.5) hit++;
                if (shown++ < 2)
                    _out.WriteLine($"   [{skill}] '{p.Query}' -> '{res.Output?.Trim()}' (want {string.Join("/", p.Allowed)}) q={q:F2} {res.DecisionPath}");
            }
            results.Add((skill, hit / (double)probes.Count, probes.Count));
            totHit += hit; totN += probes.Count;
        }

        _out.WriteLine("");
        _out.WriteLine("=== conscious-field on the real gym (per skill) ===");
        foreach (var (skill, acc, n) in results.OrderByDescending(r => r.Acc))
            _out.WriteLine($"  {skill,-18} {acc,6:P0}  ({n})");
        var overall = totHit / (double)totN;
        _out.WriteLine($"  {"OVERALL",-18} {overall,6:P0}  ({totN})");

        // The pure-homomorphism skills must be ~perfect (no training, generalises to any operands).
        double Acc(GymSkill s) => results.First(r => r.Skill == s).Acc;
        Assert.True(Acc(GymSkill.Add) >= 0.9 && Acc(GymSkill.Subtract) >= 0.9 && Acc(GymSkill.Multiply) >= 0.9,
            "single-op arithmetic must be exact via the homomorphism");
        Assert.True(Acc(GymSkill.FoldAdd) >= 0.8 && Acc(GymSkill.FoldMultiply) >= 0.8 && Acc(GymSkill.Expression) >= 0.8,
            "compositional arithmetic must reason multi-step (the whole point)");
        Assert.True(Acc(GymSkill.Predicate) >= 0.8, "predicate must compare-by-sign");
        Assert.True(Acc(GymSkill.WordedAdd) >= 0.8, "worded arithmetic must parse + compute");
        Assert.True(Acc(GymSkill.FunctionInduction) >= 0.8, "few-shot induction must derive the rule from the demos");
        // Overall across all 13: only retrieval (Synonym/Category) sits below 100%, capped by frames that contain
        // vocabulary words (a training-distribution gap, not a field bug). The breakdown above shows the split.
        Assert.True(overall >= 0.9, $"field must handle nearly all of the real gym; overall={overall:P0}");
    }

    [Fact] // The honest smoke probe (GenesisProbeSet) — every item is a deterministic gym skill; the field must
           // produce the EXACT surface string (surface-strict), so the journal headline reflects real capability.
    public void Field_PassesHonestSmokeProbe_SurfaceStrict()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var infer = new GenesisInferenceEngine(new WhitespaceGenesisTokenizer(), new GenesisNeuralModel(config),
            new DialecticalSpace(config.FaceDimension, seed: 7), null) { ConsciousField = true };
        var items = new (string Prompt, string Expected)[]
        {
            ("1 + 1", "2"), ("9 - 2", "7"), ("3 x 4", "12"), ("2 + 5 + 3", "10"), ("3 x 4 + 2", "14"),
            ("7 compared to 4", "greater"), ("what is 3 plus 4", "7"), ("5 in words", "five"),
            ("3 + 4 in words", "seven"), ("fn 2 is 4 fn 3 is 6 fn 5 is", "10"),
        };
        foreach (var (prompt, expected) in items)
        {
            var got = (infer.Generate(new GenerationRequest(prompt, 16)).Output ?? "").Trim();
            _out.WriteLine($"  '{prompt}' -> '{got}' (want '{expected}')");
            Assert.Equal(expected, got, ignoreCase: true);
        }
    }
}
