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
    // Possessives are REAL (a small closed set users actually say) — they don't pollute (no "my"->value relation forms).
    private static readonly string[] Possessives = { "my", "your", "his", "her", "our", "their", "blorp" }; // blorp = NONCE
    // Nouns + values are NONCE — the role head GENERALISES to unseen nouns (proven 8/8), so it learns the STRUCTURE
    // from made-up nouns and tags real nouns ("name") SUBJECT by position, WITHOUT planting "my name"->value relations
    // that would pollute the user's real "my name"->stephen. This is what decouples "teach grammar" from "plant facts".
    private static readonly string[] Nouns =
        { "zibble", "quax", "florp", "glim", "wozit", "tarn", "vmoo", "skree", "drelb", "fnug", "gorm", "tweel", "plost", "brindle" };
    // NONCE adjectives so a SUBJECT can be a 3-word span ("my favorite color"): the role head learns the SUBJECT is the
    // whole contiguous noun phrase before the copula, not just the determiner+noun. Without these it tagged only 2-word
    // subjects and a 3-word phrase ("my favorite color") parsed to the wrong span -> retrieved the wrong fact.
    private static readonly string[] Adjectives =
        { "snorpy", "blimmy", "trabid", "quogish", "drelby", "fnordic", "wozzy", "glimmal", "tarnic", "vexil" };
    // MANY varied values — VALUE is the hardest role to generalise because (unlike the noun, which appears in BOTH
    // assert and recall frames) a value appears ONLY in asserts, so it gets ~half the exposure. A large, varied pool
    // forces the head to learn "the post-copula content is the VALUE" by POSITION rather than memorising a few tokens,
    // so it generalises to a real never-seen value like "stephen".
    private static readonly string[] Values =
        { "zorp", "quil", "fnordle", "blivet", "zarn", "morblo", "drav", "skell", "trisk", "vunk", "phlim", "grottle",
          "wexil", "drupe", "snarl", "quonk", "blive", "torvil", "klem", "spond", "vrang", "multo", "freel", "gazz",
          "crint", "jommer", "wlexa", "borzo", "quell", "stannic", "yorvel", "mibble", "drasq", "flonk", "treb", "zellan" };
    // ROTATED grammar tokens — varied so none is a constant correlate, and NONCE ones so the ROLE generalises.
    // MANY, mostly-NONCE copulas — the point is to stop the NN memorising specific copula tokens and FORCE it to learn
    // the copula POSITION (the filler between subject and value) as NONE, so it generalises to an unseen copula. With
    // only {is,was,ploo} it learned those tokens; a held-out copula then read SUBJECT (measured). Variety breaks that.
    private static readonly string[] Copulas =
        { "is", "was", "are", "be", "ploo", "vex", "glip", "borp", "zud", "plim", "krof", "snil", "drask", "wis" };
    // REALISTIC query phrasings — how people actually ask. Multi-word cues ("what is") are fine now: the copula "is"
    // is labelled NONE by its value-adjacent POSITION in assertions (GrammarRoleLearner.AsCopula), not mislabelled
    // SUBJECT for also appearing in these recall frames. The cue words themselves appear ONLY in recalls -> QUERY.
    private static readonly string[] QueryCues = { "what is", "whats", "tell me", "who is", "do you know", "remind me of" };
    private static readonly string[] LeadIns = { "" }; // keep frames clean so the per-token role labels stay clean

    // HELD-OUT pools (DISJOINT from the training tokens above) for HONEST grading: the probe asserts a fact with
    // never-seen noun/copula/value then recalls it, so accuracy reflects GENERALISATION (can the role head parse a
    // novel frame), not memorisation of the trained set. A unit that can only recall what it trained on reports LOW
    // here and the weakest-first scheduler prioritises it. A small fixed pool → bounded, nonce → no real-phrase collision.
    private static readonly string[] HeldNouns = { "blixnar", "gizmo", "wodget", "plonk", "trizzle", "quomp", "snarf", "vlim" };
    private static readonly string[] HeldValues = { "zorptron", "quxil", "fnord", "plimth", "dworb", "skav", "trint", "vooble" };
    private static readonly string[] HeldCopulas = { "vumple", "zib", "kront", "blarg", "frot", "splim" };

    private readonly Random _rng;
    private readonly int _trainPerCycle;
    private readonly int _probeCount;

    public GrammarCurriculum(int trainPerCycle = 64, int probeCount = 24, int? seed = null)
    {
        _rng = seed is int s ? new Random(s) : new Random(); // seedable so a warm-up (GrammarWarmup) converges deterministically
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
    // VARY the DETERMINER so the role head learns SUBJECT by POSITION (the noun before the copula), not "a subject is
    // always possessive+noun". Real facts come bare ("alice is a doctor"), "the"-determined ("the password is plum"),
    // or possessive ("my name is X"). Without this variety the head tagged only possessive subjects and missed the rest.
    private string Subject()
    {
        var noun = Nouns[_rng.Next(Nouns.Length)];
        var adj = Adjectives[_rng.Next(Adjectives.Length)];
        return _rng.Next(6) switch
        {
            0 or 1 => $"{Possessives[_rng.Next(Possessives.Length)]} {noun}",            // "my zibble" — possessive (distinguishes my/your)
            2 => $"the {noun}",                                                          // "the password"
            3 => noun,                                                                   // "alice" — bare subject, no determiner
            // ~1/3 are 3-word spans so the head reliably tags the WHOLE noun phrase (a single nonce-adj example was flaky)
            _ => $"{Possessives[_rng.Next(Possessives.Length)]} {adj} {noun}",           // "my favorite color" — 3-word subject span
        };
    }
    private string Val() => Values[_rng.Next(Values.Length)];

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++)
        {
            var subject = Subject(); var value = Val(); // RANDOM binding — structure, not a sticky fact
            // ~70% ASSERTIONS so VALUE (which only appears in asserts) gets exposure comparable to the noun (which
            // appears in both frames); ~30% RECALLS is enough to train the QUERY role. Without this the value class
            // saw half the data and failed to generalise to a real never-seen value.
            if (i % 10 < 7) // ASSERTION — the value is the final token after the (rotating) copula (answer PRESENT)
                batch.Add(($"{Lead()}{subject} {Copulas[_rng.Next(Copulas.Length)]} {value}", value));
            else            // RECALL — a (rotating) query cue over the possessive phrase (answer ABSENT)
                batch.Add(($"{Lead()}{QueryCues[_rng.Next(QueryCues.Length)]} {subject}", value));
        }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        // HONEST held-out grading: each pair ASSERTS a fact with NEVER-SEEN tokens (the assert runs through the
        // inference path, so it actually learns it) then RECALLS it. Correct only if the role head GENERALISES the
        // novel frame (parses subject/value/copula by position) AND retrieval returns it. So this unit's accuracy
        // tracks GENERALISATION, giving the weakest-first scheduler a truthful weakness signal. The held-out pools are
        // nonce + small, so the probe asserts cannot collide with real user phrases and grow the space unboundedly.
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount / 2; i++)
        {
            var poss = Possessives[_rng.Next(Possessives.Length)];
            var noun = HeldNouns[_rng.Next(HeldNouns.Length)];
            var value = HeldValues[_rng.Next(HeldValues.Length)];
            var cop = HeldCopulas[_rng.Next(HeldCopulas.Length)];
            var query = QueryCues[_rng.Next(QueryCues.Length)];
            probes.Add(new TrainingProbe($"{poss} {noun} {cop} {value}", new[] { value }, RequiredDepth: 1, RequirePlatonic: false)); // assert (held-out) → echoes value if it parses
            probes.Add(new TrainingProbe($"{query} {poss} {noun}", new[] { value }, RequiredDepth: 1, RequirePlatonic: false));        // recall it right after → tests generalise + retrieve
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
