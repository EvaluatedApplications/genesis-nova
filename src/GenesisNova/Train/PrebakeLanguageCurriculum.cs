using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// PREBAKE — the run-first LANGUAGE curriculum. Warms the substrate with the SCHEMAS of English as DISTRIBUTIONAL
/// TRAINING DATA (not a dispatch list), climbing a ladder of Merge shapes from function-word recognition up to
/// multi-sentence composition. Each level adds one schema and gets LONGER inputs:
///   L1 function-words (fragments) → L2 predication / SVO (clauses) → L3 questions (Q&amp;A) →
///   L4 modification / nesting (phrases) → L5 discourse (multi-sentence, coreference).
/// CONTENT is REAL words in REAL semantic clusters (a usable vocabulary out-of-the-box, and tight clusters that SHARPEN
/// the function-word signal), SALTED with held-out NONCE tokens (~15%) as the clean anchor — free of frequency/polysemy
/// confounds — AND the generalisation PROOF (parsing a nonce sentence ⇒ the SHAPE was learned, not the vocab). The
/// closed-class GLUE (the/is/my/what…) is REAL: it's the skeleton the user actually types. Graded by <see cref="SelfAssess"/>
/// — a property of the SPACE (relational tokens introduced up to the current level separate from argument tokens),
/// scoped to the current level so it never deadlocks on a not-yet-introduced schema. NOTE: this warms the SUBSTRATE for
/// the whole ladder; cleanly PARSING each new schema (verb-as-label, wh-as-query) is the per-rung follow-on — same
/// empirical method that landed L1 (see [[nova-merge-substrate-plan]]).
/// </summary>
public sealed class PrebakeLanguageCurriculum : ITrainingCurriculum
{
    // ── Closed-class GLUE (REAL — the structural skeleton we WANT recognised as function) ───────────────────
    private static readonly string[] Det = { "the", "a", "an" };
    private static readonly string[] Poss = { "my", "your", "his", "her", "its", "our", "their" };
    private static readonly string[] Prep = { "in", "on", "at", "by", "near", "with", "under", "over", "to", "from" };
    private static readonly string[] Cop = { "is", "are", "was", "were" };
    private static readonly string[] Conj = { "and", "or", "but" };
    private static readonly string[] Wh = { "what", "where", "who", "which" };          // query markers (L3)
    private static readonly string[] Aux = { "does", "did", "can", "will" };            // auxiliaries (L3)

    // ── REAL content in REAL semantic clusters (DATA, not dispatch) ─────────────────────────────────────────
    // Concrete nouns only (no broad words like thing/do/make that distribute like function words and blur the line).
    private static readonly string[] People = { "sam", "joe", "mia", "ben", "ana", "leo", "kai", "zoe" };
    // VOCABULARY DIVERSITY is what the clustering signal runs on — a function word separates only because it bridges
    // MANY distinct, weakly-interconnected content words. A tiny lexicon (~50 words) stays densely interconnected →
    // high clustering → NO separation at reachable scale (VerbClusteringDiagnostic: 72 words → function 0.88, stuck L1).
    // So the real content pool is broad (≈100 real words across many domains) + a large NONCE long-tail (below). People
    // is one CLUSTER among many — subjects draw from the WHOLE pool (see Subj()) so verbs bridge diversely.
    private static readonly string[][] NounClusters =
    {
        new[] { "cat", "dog", "bird", "fish", "horse", "cow", "frog", "bee" },         // animals
        new[] { "red", "blue", "green", "yellow", "black", "white", "pink", "gray" },  // colours
        new[] { "apple", "bread", "rice", "soup", "cake", "egg", "milk", "pear" },     // food
        new[] { "park", "house", "school", "shop", "farm", "lake", "road", "hill" },   // places
        new[] { "car", "ball", "book", "cup", "hat", "key", "box", "door" },           // objects
        new[] { "hand", "foot", "head", "eye", "ear", "nose", "arm", "leg" },          // body
        new[] { "shirt", "coat", "shoe", "sock", "dress", "glove", "scarf", "belt" },  // clothes
        new[] { "bus", "van", "bike", "boat", "train", "plane", "truck", "cart" },     // vehicles
        new[] { "saw", "hammer", "nail", "rope", "brush", "spoon", "fork", "knife" },  // tools
        new[] { "tree", "leaf", "rock", "sand", "wind", "rain", "snow", "cloud" },     // nature
        new[] { "tea", "water", "juice", "wine", "beer", "cola", "broth", "cream" },   // drinks
        People,                                                                         // people
    };
    private static readonly string[] Verbs = { "likes", "sees", "has", "wants", "eats", "finds", "knows", "makes", "holds", "needs" };
    private static readonly string[] Adjs = { "big", "small", "old", "new", "warm", "cold", "fast", "slow", "happy", "tall" };

    // ── NONCE salt: held-out, never real — the clean anchor + the generalisation proof ──────────────────────
    private readonly string[] _nonceNouns, _nonceVerbs, _nonceAdjs;
    // ~0.45 so the large nonce long-tail actually PARTICIPATES in neighbourhoods (at 0.15 each nonce word was ~10× rarer
    // than a real word → it barely diversified anything). Real words still appear the majority (~55%) → usable + grounded.
    private const double NonceRate = 0.45;

    private static string[] BuildNonce(int count, Random r, HashSet<string> used)
    {
        const string Cons = "bdfgklmnprstvwz", Vow = "aeiou";
        string Syl() => $"{Cons[r.Next(Cons.Length)]}{Vow[r.Next(Vow.Length)]}";
        var outp = new string[count];
        for (var i = 0; i < count; i++) { string w; do { w = Syl() + Syl() + (r.Next(2) == 0 ? Syl() : ""); } while (!used.Add(w)); outp[i] = w; }
        return outp;
    }

    private readonly Random _rng;
    private readonly int _trainPerCycle;
    private const int MaxLevel = 5;
    private int _level = 1;
    private int _streak;
    private bool _mastered;

    public PrebakeLanguageCurriculum(int trainPerCycle = 128, int probeCount = 16, int? seed = null)
    {
        _rng = seed is int s ? new Random(s) : new Random();
        _trainPerCycle = Math.Max(16, trainPerCycle);
        var nr = new Random(seed ?? 12345); var used = new HashSet<string>(StringComparer.Ordinal);
        _nonceNouns = BuildNonce(160, nr, used);  // a large long-tail — the diversity the clustering signal runs on
        _nonceVerbs = BuildNonce(24, nr, used);
        _nonceAdjs = BuildNonce(24, nr, used);
    }

    public string Name => "prebake:language";
    public int Difficulty => _level;
    public bool IsMastered => _mastered;

    // ── Word pickers (REAL with a NONCE-salt chance, so every slot proves generalisation) ───────────────────
    private string Pick(string[] a) => a[_rng.Next(a.Length)];
    private string Noun() => _rng.NextDouble() < NonceRate ? Pick(_nonceNouns)
        : NounClusters[_rng.Next(NounClusters.Length)] is var cl ? cl[_rng.Next(cl.Length)] : "";
    // CRITICAL for L2: subjects must be as DIVERSE as objects. Drawing them only from the 8-person pool made every verb
    // co-occur with a tight people-hub → highly interconnected neighbourhood → verbs clustered like NOUNS (~0.6 vs
    // function ~0.19, VerbClusteringDiagnostic) → SVO never crossed the relational threshold (stuck L2 ~66%). Drawing
    // from the FULL noun vocabulary makes a verb bridge many clusters → low clustering → relational, like the copula.
    private string Subj() => Noun();
    private string Verb() => _rng.NextDouble() < NonceRate ? Pick(_nonceVerbs) : Pick(Verbs);
    private string Adj() => _rng.NextDouble() < NonceRate ? Pick(_nonceAdjs) : Pick(Adjs);
    private (string A, string B) TwoNouns() { var a = Noun(); string b; do { b = Noun(); } while (b == a); return (a, b); }

    // ── The schema ladder. Each returns (Input, Output=the salient content token to recover) ────────────────
    private (string Input, string Output) Level1() // function-words / fragments
    {
        var n = Noun();
        return _rng.Next(5) switch
        {
            0 => ($"{Pick(Det)} {n}", n),
            1 => ($"{Pick(Poss)} {n}", n),
            2 => ($"{Pick(Prep)} {Pick(Det)} {n}", n),
            3 => (TwoNouns() is var (x, y) ? $"{x} {Pick(Conj)} {y}" : n, n),
            _ => ($"{Pick(Cop)} {n}", n),
        };
    }
    private (string Input, string Output) Predication() // L2
    {
        switch (_rng.Next(4))
        {
            case 0: { var a = Adj(); return ($"{Subj()} is {a}", a); }                       // copula: sam is happy
            case 1: { var (n, n2) = TwoNouns(); return ($"{Pick(Poss)} {n} is {n2}", n2); }  // possessive copula
            case 2: { var o = Noun(); return ($"{Subj()} {Verb()} {Pick(Det)} {o}", o); }    // SVO + det
            default: { var o = Noun(); return ($"{Subj()} {Verb()} {o}", o); }               // bare SVO
        }
    }
    private (string Input, string Output) Questions() // L3 — teach + ask in one (multi-clause)
    {
        switch (_rng.Next(3))
        {
            case 0: { var s = Subj(); var a = Adj(); return ($"{s} is {a} what is {s}", a); }
            case 1: { var s = Subj(); var p = Noun(); return ($"{s} is {Pick(Prep)} the {p} where is {s}", p); }
            default: { var s = Subj(); var v = Verb(); var o = Noun(); return ($"{s} {v} {o} what does {s} {v}", o); }
        }
    }
    private (string Input, string Output) Modification() // L4 — nested phrases
    {
        switch (_rng.Next(3))
        {
            case 0: { var n = Noun(); return ($"{Adj()} {Adj()} {n}", n); }
            case 1: { var n = Noun(); return ($"{Pick(Poss)} {Adj()} {n}", n); }
            default: { var (n, n2) = TwoNouns(); return ($"the {Adj()} {n} {Pick(Prep)} the {n2}", n2); }
        }
    }
    private (string Input, string Output) Discourse() // L5 — multi-sentence / coreference / paragraphs
    {
        switch (_rng.Next(3))
        {
            case 0: { var o = Noun(); var a = Adj(); return ($"{Subj()} {Verb()} {Pick(Det)} {o} . it is {a}", a); }         // coreference
            case 1: { var o2 = Noun(); return ($"{Subj()} {Verb()} {Noun()} . {Subj()} {Verb()} {o2}", o2); }                // two sentences
            default: { var s = Subj(); var o = Noun(); return ($"{s} is {Adj()} . {s} {Verb()} {o} . {s} is {Adj()}", o); }  // 3-clause paragraph
        }
    }

    private (string Input, string Output) ByLevel(int lvl) => lvl switch
    {
        1 => Level1(), 2 => Predication(), 3 => Questions(), 4 => Modification(), _ => Discourse(),
    };

    // A REHEARSAL mix: bias to the current level, but keep replaying every lower level so warmed schemas don't erode.
    private (string Input, string Output) Frame()
    {
        var lvl = _rng.NextDouble() < 0.55 ? _level : 1 + _rng.Next(_level);
        return ByLevel(lvl);
    }

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string, string)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++) batch.Add(Frame());
        return batch;
    }

    // No surface probes — graded by SelfAssess (a property of the space), like L1. The production field doesn't echo a
    // content word, so a surface echo would read 0% even when fully warm.
    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>();

    // ── Readiness sets, SCOPED to the current level (so it grades what should be warm by now, never deadlocks) ─
    private IReadOnlyList<string> FunctionWords => Det.Concat(Poss).Concat(Prep).Concat(Cop).Concat(Conj).ToArray();
    private IReadOnlyList<string> RelationalSet(int level)
    {
        var s = new List<string>(FunctionWords);
        if (level >= 2) s.AddRange(Verbs);              // predicates are relational (bridge subject↔object)
        if (level >= 3) { s.AddRange(Wh); s.AddRange(Aux); } // query markers are relational
        return s;
    }
    private IReadOnlyList<string> ArgumentSet() => NounClusters.SelectMany(c => c).Distinct().ToArray();

    /// <summary>READINESS (property of the space): do the RELATIONAL tokens introduced up to the current level read
    /// function-like (low neighbourhood clustering — they bridge) while ARGUMENT tokens (entities/nouns) do NOT? Score =
    /// frac(relational fn-like) − frac(argument fn-like), 1.0 = clean structural separation. Scoped to <see cref="_level"/>
    /// so a not-yet-introduced schema can't drag it to a deadlock.</summary>
    public double? SelfAssess(GenesisEvalAppRuntime runtime)
    {
        if (runtime.State.Memory is not DialecticalSpace ds) return null;
        var rel = RelationalSet(_level); var arg = ArgumentSet();
        if (rel.Count == 0) return 0.0;
        var relFn = rel.Count(ds.IsFunctionLike) / (double)rel.Count;
        var argFn = arg.Count == 0 ? 0.0 : arg.Count(ds.IsFunctionLike) / (double)arg.Count;
        return Math.Max(0.0, relFn - argFn);
    }

    public void RecordCycle(CycleGrade grade)
    {
        if (grade.Accuracy >= 0.80) { _streak++; if (_streak >= 3) { _mastered = true; if (_level < MaxLevel) { _level++; _streak = 0; } } }
        else { _streak = 0; if (grade.Accuracy < 0.55) _mastered = false; }
    }

    // ── DIAGNOSTICS (consumed by the smoke / compatible with the prior function-word curriculum) ─────────────
    /// <summary>The closed-class function-word GLUE (should read function-like once warm).</summary>
    public IReadOnlyList<string> Glue => FunctionWords;
    /// <summary>Sample REAL argument words (should NOT read function-like) — the content side of the separation.</summary>
    public IReadOnlyList<string> SampleContent(int n) => ArgumentSet().Where((_, i) => i % 3 == 0).Take(n).ToList();
    /// <summary>The held-out NONCE argument tokens (generalisation set — never seen as real words).</summary>
    public IReadOnlyList<string> NonceContent(int n) => _nonceNouns.Take(n).ToList();
    /// <summary>The PREDICATES (verbs) — the L2 relational tokens whose clustering we measure to see if SVO warms.</summary>
    public IReadOnlyList<string> Predicates => Verbs;
    /// <summary>The query markers (wh + aux) introduced at L3.</summary>
    public IReadOnlyList<string> QueryMarkers => Wh.Concat(Aux).ToArray();
}
