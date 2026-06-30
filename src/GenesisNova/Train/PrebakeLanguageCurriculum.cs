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
    private static readonly string[] Prep = { "in", "on", "at", "by", "near", "under", "over" };  // spatial only, so "X is {prep} the place" reads right
    private static readonly string[] Cop = { "is", "are", "was", "were" };
    private static readonly string[] Conj = { "and", "or", "but" };
    private static readonly string[] Wh = { "what", "where", "who" };    // query markers actually used at L3
    private static readonly string[] Aux = { "does" };                  // do-support (L3)

    // ── CONTENT as a TYPED LEXICON (the clever data structure). Every word carries a semantic CATEGORY; every verb
    // carries a CASE-FRAME (subject-category, object-category). The generator fills typed slots from this, so plausibility
    // ("cat eats fish", never "cat eats screw") lives in the DATA. That keeps co-occurrence STRUCTURED — a content word
    // keeps a distinctive in-domain neighbourhood instead of smearing across everything, which is what averaged the
    // clustering signal to death. Function words and verbs stay universal; they SHOULD bridge.
    // CONTENT exemplars must be UNAMBIGUOUS concrete nouns: words that ALSO double as common verbs (fish, duck, park,
    // shop, …) legitimately co-occur broadly and read function-like, so the learned classifier is RIGHT to flag them —
    // using them as "clean content" makes the function-word-separation probe lie. (These clusters feed only the inspect
    // probe / SelfAssess now that the prebake trains on 100% real corpus, so this is a probe-validity fix, not training.)
    private static readonly string[] Animals = { "cat", "dog", "bird", "owl", "horse", "cow", "frog", "bee", "swan", "goat" };
    private static readonly string[] Colours = { "red", "blue", "green", "yellow", "black", "white", "pink", "gray", "brown", "gold" };
    private static readonly string[] Food = { "apple", "bread", "rice", "soup", "cake", "egg", "milk", "pear", "plum", "jam" };
    private static readonly string[] Places = { "barn", "house", "school", "cabin", "farm", "lake", "road", "hill", "town", "beach" };
    private static readonly string[] Objects = { "ball", "book", "cup", "hat", "key", "box", "lamp", "clock", "bag", "toy" };
    private static readonly string[] Body = { "hand", "foot", "head", "eye", "ear", "nose", "arm", "leg", "hair", "tooth" };
    private static readonly string[] Clothes = { "shirt", "coat", "shoe", "sock", "dress", "glove", "scarf", "belt", "cap", "boot" };
    private static readonly string[] Vehicles = { "bus", "van", "bike", "boat", "train", "plane", "truck", "cart", "ship", "jet" };
    private static readonly string[] Tools = { "saw", "hammer", "nail", "rope", "brush", "spoon", "fork", "knife", "drill", "screw" };
    private static readonly string[] Nature = { "tree", "leaf", "rock", "sand", "wind", "rain", "snow", "cloud", "star", "moon" };
    private static readonly string[] Drinks = { "tea", "water", "juice", "wine", "ale", "cola", "broth", "cream", "soda", "cider" };
    private static readonly string[] Furniture = { "chair", "table", "bed", "desk", "shelf", "sofa", "stool", "bench", "crib", "rack" };
    private static readonly string[] Instruments = { "drum", "flute", "harp", "horn", "bell", "pipe", "lute", "gong", "reed", "chime" };
    private static readonly string[] Plants = { "rose", "fern", "oak", "vine", "moss", "weed", "bush", "palm", "ivy", "herb" };
    private static readonly string[] Jobs = { "chef", "nurse", "pilot", "clerk", "guard", "maid", "smith", "monk", "scout", "cook" };
    private static readonly string[] Weather = { "sun", "fog", "ice", "heat", "storm", "frost", "mist", "hail", "dew", "gust" };
    private static readonly string[] People = { "sam", "joe", "mia", "ben", "ana", "leo", "kai", "zoe", "max", "eve" };
    private static readonly string[][] NounClusters = { Animals, Colours, Food, Places, Objects, Body, Clothes, Vehicles, Tools, Nature, Drinks, Furniture, Instruments, Plants, Jobs, Weather, People };
    private static readonly string[] AllNouns = NounClusters.SelectMany(c => c).Distinct().ToArray();
    private static readonly HashSet<string> Proper = new(People, StringComparer.Ordinal);  // proper names take NO article
    private static readonly string[][] CommonClusters = NounClusters.Where(c => c != People).ToArray();  // clusters that take articles/adjectives

    // category -> words: base clusters + UNION categories that span clusters (agent, edible, handleable, …)
    private static readonly Dictionary<string, string[]> Cat = new(StringComparer.Ordinal)
    {
        ["animal"] = Animals, ["colour"] = Colours, ["food"] = Food, ["place"] = Places, ["object"] = Objects,
        ["body"] = Body, ["clothes"] = Clothes, ["vehicle"] = Vehicles, ["tool"] = Tools, ["nature"] = Nature,
        ["drink"] = Drinks, ["furniture"] = Furniture, ["instrument"] = Instruments, ["plant"] = Plants,
        ["job"] = Jobs, ["weather"] = Weather, ["person"] = People,
        ["agent"] = People.Concat(Jobs).Concat(Animals).ToArray(),
        ["human"] = People.Concat(Jobs).ToArray(),
        ["edible"] = Food.Concat(Drinks).ToArray(),
        ["thing"] = Objects.Concat(Tools).Concat(Furniture).Concat(Instruments).Concat(Clothes).ToArray(),
        ["handleable"] = Objects.Concat(Tools).Concat(Furniture).Concat(Instruments).Concat(Clothes).Concat(Food).Concat(Vehicles).ToArray(),
        ["any"] = AllNouns,
        ["common"] = AllNouns.Where(w => !Proper.Contains(w)).ToArray(),  // non-proper: takes articles / possessives / adjectives
        // things you can be in/on/near/under — MANY clusters, so a preposition bridges domains (and reads relational)
        // instead of only ever seeing places and clustering with them.
        ["locative"] = Places.Concat(Furniture).Concat(Objects).Concat(Nature).Concat(Vehicles).Concat(Body).Concat(Tools).Concat(Instruments).ToArray(),
    };
    // VERB CASE-FRAMES: (verb, subject-category, object-category) — selectional restrictions AS DATA.
    private static readonly (string V, string Subj, string Obj)[] VerbFrames =
    {
        ("eat", "agent", "edible"),
        ("read", "human", "object"), ("make", "human", "handleable"), ("build", "human", "handleable"),
        ("hold", "human", "handleable"), ("open", "human", "handleable"), ("break", "human", "handleable"),
        ("draw", "human", "thing"), ("paint", "human", "thing"), ("take", "human", "handleable"),
        ("give", "human", "handleable"), ("keep", "human", "handleable"), ("send", "human", "handleable"),
        ("bring", "human", "handleable"),
        ("find", "agent", "thing"), ("have", "agent", "handleable"), ("want", "agent", "handleable"), ("need", "agent", "handleable"),
        ("like", "agent", "any"), ("see", "agent", "any"), ("know", "agent", "any"),
        ("feel", "agent", "any"), ("love", "agent", "any"), ("hear", "agent", "any"),
    };
    private static string Sg(string v) => v == "have" ? "has" : v + "s";   // 3rd-person-singular for statements (base form used in questions)
    private static readonly string[] Verbs = VerbFrames.Select(f => Sg(f.V)).Distinct().ToArray();  // statement-form verbs = the relational set for SelfAssess
    private static readonly string[] Adjs = { "big", "small", "old", "new", "warm", "cold", "fast", "slow", "happy", "tall", "short", "long", "wide", "soft", "hard", "loud", "quiet", "bright", "dark", "clean", "sharp", "heavy", "light", "smooth" };

    // ── NONCE salt: held-out, never real — the clean anchor + the generalisation proof ──────────────────────
    private readonly string[] _nonceNouns, _nonceVerbs, _nonceAdjs;
    // Low: structured REAL words carry the signal now; nonce is just a light generalisation salt (parse a never-seen word).
    private const double NonceRate = 0.12;

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
    // CAPPED AT L1 (option 4): the prebake's ONE job is to warm function-word recognition, which stays clean in the
    // simple-frame regime. Climbing into multi-clause frames (L2-L5) DENSIFIES the co-occurrence graph and erodes that
    // signal no matter how structured the data is (function 0.31 vs nouns 0.53 at L5, still drifting). So composition —
    // SVO, questions, multi-sentence — is handled as a PARSING task by the inference engine (Merge over the function-word
    // skeleton), NOT warmed here. The L2-L5 templates are kept (re-enable by raising MaxLevel) but are not trained.
    private const int MaxLevel = 1;
    private const double AdvanceBar = 0.65;   // "warm enough" to ramp the next skill in; lower levels keep rehearsing to perfect
    private const int LevelPatience = 25;     // backstop: never let a slow level stall the whole ladder
    private int _level = 1;
    private int _streak;
    private int _cyclesAtLevel;
    private bool _mastered;

    public PrebakeLanguageCurriculum(int trainPerCycle = 128, int probeCount = 16, int? seed = null, int startLevel = 1)
    {
        _rng = seed is int s ? new Random(s) : new Random();
        _trainPerCycle = Math.Max(16, trainPerCycle);
        _level = Math.Clamp(startLevel, 1, MaxLevel);  // resume the difficulty reached on a prior run (persisted by the gym)
        var nr = new Random(seed ?? 12345); var used = new HashSet<string>(StringComparer.Ordinal);
        _nonceNouns = BuildNonce(160, nr, used);  // a large long-tail — the diversity the clustering signal runs on
        _nonceVerbs = BuildNonce(24, nr, used);
        _nonceAdjs = BuildNonce(24, nr, used);
    }

    public string Name => "prebake:language";
    public int Difficulty => _level;
    public bool IsMastered => _mastered;

    // ── Typed pickers, driven by the lexicon + case-frames ──────────────────────────────────────────────────
    private string Pick(string[] a) => a[_rng.Next(a.Length)];
    private string[] WordsOf(string cat) => Cat.TryGetValue(cat, out var w) ? w : AllNouns;
    // a BARE content word of a category, with occasional NONCE salt (generalisation proof)
    private string Word(string cat) => _rng.NextDouble() < NonceRate ? Pick(_nonceNouns) : Pick(WordsOf(cat));
    private string Adj() => _rng.NextDouble() < NonceRate ? Pick(_nonceAdjs) : Pick(Adjs);
    // a/an agreement, and a "the / a / an" article picker that agrees with the word it precedes.
    private static string AnA(string w) => w.Length > 0 && "aeiou".IndexOf(char.ToLowerInvariant(w[0])) >= 0 ? "an" : "a";
    private string Art(string w) => _rng.Next(2) == 0 ? "the" : AnA(w);
    // an argument NOUN PHRASE: proper names take no article, common nouns get an agreeing one. Returns (surface, bare word).
    private (string Surface, string Word) NP(string cat)
    {
        var w = Word(cat);
        return (Proper.Contains(w) ? w : $"{Art(w)} {w}", w);
    }
    // a CLAUSE from a verb case-frame: subject + verb + object, type-consistent. The STATEMENT uses the 3rd-singular form
    // ("sam eats fish"); the returned Verb is the BASE form, for the do-support question ("what does sam eat").
    private (string Text, string Subj, string Verb, string Obj) Clause(bool adjObj = false)
    {
        var f = VerbFrames[_rng.Next(VerbFrames.Length)];
        var nonce = _rng.NextDouble() < NonceRate;
        var baseV = nonce ? Pick(_nonceVerbs) : f.V;          // question form
        var sgV = nonce ? baseV : Sg(f.V);                    // statement form
        var s = NP(f.Subj);
        var o = NP(f.Obj);
        var objText = adjObj ? InsertAdj(o.Surface) : o.Surface;
        return ($"{s.Surface} {sgV} {objText}", s.Surface, baseV, o.Word);
    }
    private string InsertAdj(string np) { var a = Adj(); var sp = np.IndexOf(' '); return sp < 0 ? $"{a} {np}" : np.Insert(sp + 1, a + " "); }
    // 2-3 BARE words from the SAME cluster (conjunctions/lists keep content with its kin, never "cat and screw")
    private string[] SameCluster(int n, bool common = false)
    {
        var pool = common ? CommonClusters : NounClusters;
        var cl = pool[_rng.Next(pool.Length)];
        var pick = new List<string>(n);
        while (pick.Count < n) { var w = cl[_rng.Next(cl.Length)]; if (!pick.Contains(w)) pick.Add(w); }
        return pick.ToArray();
    }

    // ── The schema ladder. Each returns (Input, Output=the bare content token to recover) ─────────────────────
    private (string Input, string Output) Level1()  // fragments — function words bridging DIVERSE content
    {
        // Closed-class glue (articles/possessives/prepositions) must bridge MANY token types to read function-like (its
        // neighbourhood washes out → low coherence). Putting possessives/articles ONLY before a single noun kept them
        // looking content-ish (measured: possessives stuck at coherence ~0.16-0.27). So here they co-occur with
        // ADJECTIVES, OTHER GLUE, CONJUNCTIONS and nouns from ALL clusters — diverse neighbourhoods, not one cluster.
        switch (_rng.Next(13))
        {
            case 0:  { var n = Word("common"); return ($"{Art(n)} {n}", n); }                                  // the book / a cup
            case 1:  { var n = Word("common"); return ($"{Art(n)} {Adj()} {n}", n); }                          // the big cup (article + adjective)
            case 2:  { var n = Word("common"); return ($"{Pick(Poss)} {n}", n); }                              // my hat
            case 3:  { var n = Word("common"); return ($"{Pick(Poss)} {Adj()} {n}", n); }                      // my big hat (possessive + adjective)
            case 4:  { var s = SameCluster(2, common: true); return ($"{Pick(Poss)} {s[0]} {Pick(Conj)} {Pick(Poss)} {s[1]}", s[1]); } // my cat and your dog
            case 5:  { var n = Word("common"); return ($"{Pick(Prep)} {Pick(Poss)} {n}", n); }                 // in my house (preposition + possessive)
            case 6:  { var p = NP("locative"); return ($"{Pick(Prep)} {p.Surface}", p.Word); }                 // on the table / near the tree
            case 7:  { var s = SameCluster(2, common: true); return ($"{Art(s[0])} {s[0]} {Pick(Conj)} {Art(s[1])} {s[1]}", s[1]); } // a cat and the dog (article + conjunction)
            case 8:  { var s = SameCluster(2); return ($"{s[0]} {Pick(Conj)} {s[1]}", s[1]); }                 // cat and dog
            case 9:  { var n = Word("common"); return ($"{Adj()} {n}", n); }                                   // big cat
            case 10: { var s = SameCluster(3); return ($"{s[0]} {s[1]} {Pick(Conj)} {s[2]}", s[2]); }          // tea juice and water
            case 11: { var n = Word("common"); return ($"{Pick(Cop)} {Pick(Poss)} {n}", n); }                 // is my book (copula + possessive)
            default: { var n = Word("thing"); return ($"{Pick(Cop)} {n}", n); }                               // is book
        }
    }
    private (string Input, string Output) Predication()  // L2 — copula + SVO from case-frames
    {
        switch (_rng.Next(6))
        {
            case 0: { var s = NP("agent"); var a = Adj(); return ($"{s.Surface} is {a}", a); }                       // sam is happy / the cat is happy
            case 1: { var s = NP("thing"); var a = Adj(); return ($"{s.Surface} is {a}", a); }                       // the soup is warm
            case 2: { var t = Word("thing"); var col = Word("colour"); return ($"{Pick(Poss)} {t} is {col}", col); } // my hat is blue
            case 3: { var c = Clause();             return (c.Text, c.Obj); }                                        // cat eats fish / sam reads a book
            case 4: { var c = Clause(adjObj: true); return (c.Text, c.Obj); }                                        // sam reads the big book
            default: { var c = Clause();            return (c.Text, c.Obj); }                                        // more SVO weight
        }
    }
    private (string Input, string Output) Questions()  // L3 — statement then question (warms wh + do-support)
    {
        switch (_rng.Next(4))
        {
            case 0: { var c = Clause(); return ($"{c.Text} what does {c.Subj} {c.Verb}", c.Obj); }                   // sam reads a book what does sam read
            case 1: { var s = NP("agent"); var p = NP("place"); return ($"{s.Surface} is {Pick(Prep)} {p.Surface} where is {s.Surface}", p.Word); } // the cat is in the park where is the cat
            case 2: { var s = NP("agent"); var a = Adj(); return ($"{s.Surface} is {a} who is {a}", s.Word); }       // sam is happy who is happy -> sam
            default: { var c = Clause(adjObj: true); return ($"{c.Text} what does {c.Subj} {c.Verb}", c.Obj); }
        }
    }
    private (string Input, string Output) Modification()  // L4 — nested phrases
    {
        switch (_rng.Next(4))
        {
            case 0: { var n = Word("common"); return ($"{Adj()} {Adj()} {n}", n); }                                  // big old cat
            case 1: { var n = Word("common"); return ($"{Pick(Poss)} {Adj()} {n}", n); }                            // my red hat
            case 2: { var s = SameCluster(2, common: true); return ($"the {Adj()} {s[0]} {Pick(Prep)} the {s[1]}", s[1]); } // the big cat near the dog
            default: { var s = SameCluster(2, common: true); return ($"{Adj()} {s[0]} and {Adj()} {s[1]}", s[1]); } // big cat and small dog
        }
    }
    private (string Input, string Output) Discourse()  // L5 — multi-sentence / coreference / paragraphs
    {
        switch (_rng.Next(4))
        {
            case 0: { var c = Clause(); var a = Adj(); return ($"{c.Text} . it is {a}", a); }                        // sam holds a cup . it is red
            case 1: { var c1 = Clause(); var c2 = Clause(); return ($"{c1.Text} . {c2.Text}", c2.Obj); }             // two sentences
            case 2: { var c = Clause(); var a = Adj(); return ($"{c.Subj} is {a} . {c.Text}", c.Obj); }              // sam is happy . sam eats fish
            default: { var c1 = Clause(); var c2 = Clause(); var a = Adj(); return ($"{c1.Text} . {c2.Text} . they are {a}", a); }
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
        _cyclesAtLevel++;
        if (grade.Accuracy >= AdvanceBar) _streak++; else _streak = 0;
        // Ramp the next skill in when the current level is warm enough (held briefly) OR a patience budget elapses, so a
        // slow-warming level can't block the whole ladder (e.g. the L2 verb plateau). Lower levels keep rehearsing via
        // Frame, so they keep sharpening after the focus moves on — overlapping-waves acquisition, not master-then-advance.
        if ((_streak >= 2 || _cyclesAtLevel >= LevelPatience) && _level < MaxLevel) { _level++; _streak = 0; _cyclesAtLevel = 0; }
        _mastered = _level >= MaxLevel && grade.Accuracy >= AdvanceBar;
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

    /// <summary>DIAGNOSTIC: PURE frames for a given level (bypasses the rehearsal mix), so generated English can be
    /// inspected for parsability/plausibility level by level.</summary>
    public IReadOnlyList<(string Input, string Output)> SampleLevel(int lvl, int n)
    {
        var outp = new List<(string, string)>(n);
        for (var i = 0; i < n; i++) outp.Add(ByLevel(Math.Clamp(lvl, 1, MaxLevel)));
        return outp;
    }
}
