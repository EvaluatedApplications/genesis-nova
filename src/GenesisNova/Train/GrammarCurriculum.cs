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
    // VARIED, EPHEMERAL bindings — the curriculum teaches the assert/recall STRUCTURE (warm the structural tokens +
    // their KEY/VALUE/QUERY roles), NOT specific facts. Possessors and nouns warm as KEYS; the large value pool warms
    // as VALUES; bindings are RANDOM each example so no "my name"->X fact sticks (a fixed world would plant a strong
    // relation that fights the user's real runtime fact). The user's own assertions are what carry actual bindings.
    private static readonly string[] Possessives = { "my", "your", "his", "her", "our", "their", "blorp" }; // blorp = NONCE
    private static readonly string[] Nouns = { "name", "dog", "car", "job", "city", "color", "food", "book", "song", "friend", "team", "drink", "hobby", "boss" };
    private static readonly string[] Values =
    {
        "sam", "rex", "leo", "mia", "fido", "audi", "coder", "lisbon", "rovers", "cola", "teal", "ramen",
        "alex", "kim", "max", "nova", "pixel", "echo", "blue", "jade", "ace", "milo", "luna", "ziggy",
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
    private string Subject() => $"{Possessives[_rng.Next(Possessives.Length)]} {Nouns[_rng.Next(Nouns.Length)]}";
    private string Val() => Values[_rng.Next(Values.Length)];

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++)
        {
            var subject = Subject(); var value = Val(); // RANDOM binding — structure, not a sticky fact
            if (i % 2 == 0) // ASSERTION — the value is the final token after the (rotating) copula (answer PRESENT)
                batch.Add(($"{Lead()}{subject} {Copulas[_rng.Next(Copulas.Length)]} {value}", value));
            else            // RECALL — a (rotating) query cue over the possessive phrase (answer ABSENT)
                batch.Add(($"{Lead()}{QueryCues[_rng.Next(QueryCues.Length)]} {subject}", value));
        }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount; i++)
            // STRUCTURAL grading: bindings are ephemeral, so accept ANY value-pool token — the lesson is producing a
            // value-shaped recall through the frame, not memorising a specific fact (the user's assertions carry those).
            probes.Add(new TrainingProbe($"{QueryCues[_rng.Next(QueryCues.Length)]} {Subject()}", Values, RequiredDepth: 1, RequirePlatonic: false));
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
