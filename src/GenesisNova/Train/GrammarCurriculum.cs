using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// GRAMMAR curriculum — teaches the ASSERT/RECALL frame ("&lt;poss&gt; &lt;noun&gt; &lt;copula&gt; &lt;value&gt;" →
/// remember; "&lt;query&gt; &lt;poss&gt; &lt;noun&gt;" → recall) so the structural roles (copula / question cue /
/// possessive) are LEARNED, not hardcoded word-lists. The mechanism: a structural token (is/was/my/your/what…) is a
/// FUNCTION word — across these many frames it co-occurs with everything, so its meaning-cloud collapses toward the
/// global centroid and the field can recognise it as filler distributionally (the same learned signal as
/// <c>DialecticalSpace.IsFunctionLike</c>; see the nova-function-word-learned / nova-learned-grammar-roles notes).
///
/// Crucially the grammar tokens are deliberately VARIED and ROTATED — and include NONCE ones ("ploo" copula, "blorp"
/// possessive) that exist in NO code list — so the model learns the ROLE, not the specific word (proven: a nonce
/// copula reads in the same filler band as "is"). The words live HERE in the DATA; the field stays general.
///
/// Bindings are a small FIXED mini-world (so recall is consistently gradeable) with varied POSSESSORS (my/your/his/
/// her/our/their) — that's what warms the possessives apart from one another. Bounded difficulty: masters once it
/// reliably recalls the world through any framing.
/// </summary>
public sealed class GrammarCurriculum : ITrainingCurriculum
{
    // A fixed mini-world: (possessive phrase) → value. Varied possessors so each warms as its own token.
    private static readonly (string Subject, string Value)[] World =
    {
        ("my name", "sam"), ("your name", "rex"), ("his name", "leo"), ("her name", "mia"),
        ("my dog", "fido"), ("your car", "audi"), ("his job", "coder"), ("her city", "lisbon"),
        ("our team", "rovers"), ("their drink", "cola"), ("my color", "teal"), ("your food", "ramen"),
    };

    // ROTATED grammar tokens — varied so none is a constant correlate, and NONCE ones so the ROLE generalises.
    private static readonly string[] Copulas = { "is", "was", "is", "ploo" };           // ploo = NONCE copula
    private static readonly string[] QueryCues = { "what is", "whats", "tell me", "remind me of", "who is" };
    private static readonly string[] LeadIns = { "", "", "", "hey, ", "ok so, ", "quick one, " };

    private readonly Random _rng = new();
    private readonly int _trainPerCycle;
    private readonly int _probeCount;

    public GrammarCurriculum(int trainPerCycle = 64, int probeCount = 24)
    {
        _trainPerCycle = Math.Max(16, trainPerCycle);
        _probeCount = Math.Max(8, probeCount);
    }

    public string Name => "grammar";
    public int Difficulty => Level;
    public int Level { get; private set; } = 1;
    public double MasteryBar { get; init; } = 0.80;
    public int StableCyclesToAdvance { get; init; } = 3;
    private const int MaxLevel = 3;
    public int MasteryDepth => MaxLevel;
    private int _streak;
    private bool _mastered;
    public bool IsMastered => _mastered;

    private string Lead() => LeadIns[_rng.Next(LeadIns.Length)];

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++)
        {
            var (subject, value) = World[_rng.Next(World.Length)];
            if (i % 2 == 0) // ASSERTION — the value is the final token after the (rotating) copula
                batch.Add(($"{Lead()}{subject} {Copulas[_rng.Next(Copulas.Length)]} {value}", value));
            else            // RECALL — a (rotating) query cue over the possessive phrase
                batch.Add(($"{Lead()}{QueryCues[_rng.Next(QueryCues.Length)]} {subject}", value));
        }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount; i++)
        {
            var (subject, value) = World[_rng.Next(World.Length)];
            // RECALL through a rotating query frame — graded route-agnostic (it's a field retrieval, not a route).
            probes.Add(new TrainingProbe($"{QueryCues[_rng.Next(QueryCues.Length)]} {subject}", new[] { value }, RequiredDepth: 1, RequirePlatonic: false));
        }
        return probes;
    }

    public void RecordCycle(CycleGrade grade)
    {
        if (grade.Accuracy >= MasteryBar)
        {
            if (++_streak >= StableCyclesToAdvance)
            {
                _streak = 0;
                if (Level < MaxLevel) Level++;
                else _mastered = true;
            }
        }
        else { _streak = 0; if (grade.Accuracy < MasteryBar - 0.15) _mastered = false; }
    }
}
