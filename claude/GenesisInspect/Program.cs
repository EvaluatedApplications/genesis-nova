using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;

// ════════════════════════════════════════════════════════════════════════════════════════════════════
//  GenesisInspect — a diagnostic CLI for the autonomously-trained Genesis-Nova checkpoint.
//  Loads the latest auto-saved checkpoint (or one you point at) and lets you see WHAT the model is and
//  HOW it answers: architecture + substrate stats, per-query decision paths + platonic activation, and a
//  capability probe. Inspects the SAME runtime inference uses, so nothing here diverges from the real model.
//
//  Usage (from repo root):
//    dotnet run --project tools/GenesisInspect -c Release -- report
//    dotnet run --project tools/GenesisInspect -c Release -- query "what is 12 + 7"
//    dotnet run --project tools/GenesisInspect -c Release -- probe
//    dotnet run --project tools/GenesisInspect -c Release -- space three
//  Flags: --cpu (inspect on CPU), --state-dir <dir> (which model — DEFAULT <repo>\.claude-nova, the
//         ClaudeMemory daemon; pass the AppData models dir for the autonomous trainer), --checkpoint <path>
//         (a specific autosave file; its directory becomes the state dir). The CLI is strictly READ-ONLY.
// ════════════════════════════════════════════════════════════════════════════════════════════════════

var argl = args.ToList();
var useCpu = argl.Remove("--cpu");
string? checkpoint = null;
var ci = argl.IndexOf("--checkpoint");
if (ci >= 0 && ci + 1 < argl.Count) { checkpoint = argl[ci + 1]; argl.RemoveAt(ci + 1); argl.RemoveAt(ci); }
string? stateDirArg = null;
var sdi = argl.IndexOf("--state-dir");
if (sdi >= 0 && sdi + 1 < argl.Count) { stateDirArg = argl[sdi + 1]; argl.RemoveAt(sdi + 1); argl.RemoveAt(sdi); }
var cmd = argl.Count > 0 ? argl[0].ToLowerInvariant() : "report";
var rest = argl.Skip(1).ToList();

void P(string s = "") => Console.WriteLine(s);
void H(string title) { P(); P("══ " + title + " " + new string('═', Math.Max(0, 80 - title.Length - 4))); }

// Walk up from the binary to the repo root (the folder holding GenesisNova.slnx), same as the ClaudeMemory
// daemon does — so the DEFAULT model we inspect is the daemon's checkpoint in <repo>\.claude-nova, NOT the
// autonomous trainer's AppData checkpoint.
static string RepoRoot()
{
    var d = new DirectoryInfo(AppContext.BaseDirectory);
    while (d is not null)
    {
        if (File.Exists(Path.Combine(d.FullName, "GenesisNova.slnx"))) return d.FullName;
        d = d.Parent;
    }
    return Directory.GetCurrentDirectory();
}

// Which model? --state-dir wins; else the directory of an explicit --checkpoint; else the ClaudeMemory
// daemon's .claude-nova. Pointing LocalStateDirectory here makes ResolveCheckpointPath + AutoResume bootstrap
// the SAME checkpoint (NN + .platonic.json companion) the daemon trains — and Diagnose() reports it correctly.
var stateDir = stateDirArg
    ?? (checkpoint is not null ? Path.GetDirectoryName(Path.GetFullPath(checkpoint)) : null)
    ?? Path.Combine(RepoRoot(), ".claude-nova");

P("GenesisInspect — diagnostic CLI for the trained Genesis-Nova model");
var config = new GenesisNovaConfig(
    Backend: useCpu ? ComputeBackend.Cpu : ComputeBackend.Gpu,
    AutoResume: true,       // bootstrap the resolved state dir's checkpoint (+ platonic companion) on construction
    AutoPersist: false,     // STRICTLY READ-ONLY — never write back over a live daemon's checkpoint
    LocalStateDirectory: stateDir);
var runtime = new GenesisEvalAppRuntime(config);
P($"  state dir     : {stateDir}");
P($"  checkpoint    : {runtime.AutoCheckpointPath}");
// A non-standard checkpoint filename (not genesis-nova.autosave.checkpoint.json) won't be picked up by
// AutoResume — load it explicitly. Safe now: AutoPersist=false means Persist-on-load writes nothing.
if (checkpoint is not null &&
    !string.Equals(Path.GetFullPath(checkpoint), runtime.AutoCheckpointPath, StringComparison.OrdinalIgnoreCase))
{
    P($"  explicit load : {checkpoint}");
    await runtime.LoadAsync(checkpoint);
}

void Report()
{
    var d = runtime.Diagnose();
    H("MODEL");
    P("  kind          : hybrid — a small GRU controller operating a structured platonic substrate (not an LLM)");
    P($"  backend       : {d.Backend}");
    P($"  checkpoint    : {d.CheckpointPath}");
    P($"                  exists={d.CheckpointExists}{(d.CheckpointWriteUtc is { } w ? $"   saved {w:u}" : "   (none — showing a FRESH/untrained model)")}");
    P($"  controller    : GRU hidden={d.HiddenSize}   params={d.ParameterCount:N0}   plan-kinds={d.PlanKindCount}");
    P($"  substrate     : face-dim={d.FaceDimension}   vocabulary={d.VocabularySize}");

    H("PLATONIC SPACE");
    P($"  concepts        : {d.NodeCount:N0}");
    P($"  relations       : {d.RelationCount:N0}");
    P($"  function-elts   : {d.FunctionElementCount}  (shapes represented as ElementKind.Function)");
    P($"  learned ops     : {d.LearnedTransformCount} transforms · {d.FoldPathCount} fold-paths · {d.LogLinearFitCount} log-linear fits");
    P($"  chunk store     : {d.ChunkCount} mined chunks across {d.ChunkTagCount} tag(s)");
    P($"  autonomous hist : {d.AutonomousRounds} training round(s)");

    H("CAPACITY & SPACE MANAGER");
    var nodePct = 100.0 * d.NodeCount / Math.Max(1, d.MaxNodes);
    var relPct = 100.0 * d.RelationCount / Math.Max(1, d.MaxRelations);
    var pressure = d.RelationBudget > 0 ? (double)d.RelationCount / d.RelationBudget : 0.0;
    P($"  node fill       : {d.NodeCount:N0} / {d.MaxNodes:N0}   ({nodePct:F2}% of cap)");
    P($"  relation fill   : {d.RelationCount:N0} / {d.MaxRelations:N0}   ({relPct:F3}% of cap)");
    P($"  relation budget : {d.RelationBudget:N0} (soft = nodes×6 + 128)   pressure {pressure:F3}");
    P($"  space manager   : {(d.SpaceManagerEnabled ? "ENABLED" : "OFF")}");
    P($"  hard eviction   : only at 100% of caps — {Math.Max(0, d.MaxNodes - d.NodeCount):N0} nodes below the node cap");
    P($"  prune triggers  : noise-ratio >= ~0.585 OR relation-pressure >= 1.12 (NOT raw count)");
    P($"                    → pressure {pressure:F3} ⇒ {(pressure >= 1.12 ? "would rebalance/prune" : "below the prune threshold (manager would observe/reinforce/expand, not trim)")}");

    if (d.TopRelations.Length > 0)
    {
        H("TOP RELATIONS (by observation count)");
        foreach (var r in d.TopRelations) P($"  {r.Left,-18} <-> {r.Right,-18}  obs {r.ObservationCount}");
    }
    if (d.LearnedTransforms.Length > 0)
    {
        H("LEARNED TRANSFORMS (unary functions T(f))");
        foreach (var t in d.LearnedTransforms) P($"  {t.Name,-18} obs {t.ObservationCount,-4} conf {t.Confidence:F2}  [{t.State}]");
    }
    if (d.FunctionElements.Length > 0)
    {
        H("FUNCTION ELEMENTS (composer shapes)");
        foreach (var f in d.FunctionElements) P($"  {f.Symbol,-18} -> {(f.References.Length > 0 ? string.Join(", ", f.References) : "(leaf)")}");
    }
    if (d.Chunks.Length > 0)
    {
        H("CHUNK STORE (scaffolds mined from correct outputs)");
        foreach (var c in d.Chunks) P($"  [{c.Tag}]  \"{c.Chunk}\"  x{c.Count}");
    }
}

async Task QueryAsync(string input, int maxNodes = 10, int maxEdges = 10)
{
    H($"QUERY  \"{input}\"");
    try
    {
        var pred = await runtime.PredictAsync(input, maxTokens: 16);
        var r = pred.Result!;
        P($"  output        : \"{r.Output}\"");
        P($"  decision path : {r.DecisionPath}");
        P($"  route         : platonic={r.UsedPlatonicQuery}  neural-fallback={r.UsedNeuralFallback}  confidence={r.PlatonicConfidence:F2}  hops={r.PlatonicHopCount}  chunks={r.ChunksGenerated}");
        if (!string.IsNullOrEmpty(r.RoutedTransform)) P($"  routed xform  : {r.RoutedTransform}");

        var act = runtime.AnalyzePlatonicActivation(input, maxNodes: maxNodes, maxEdges: maxEdges);
        P($"  input tokens  : {string.Join(" ", act.InputTokens)}");
        P($"  anchors       : {(act.Anchors.Length > 0 ? string.Join(", ", act.Anchors) : "(none — no input token is a known concept)")}");
        if (act.Nodes.Length > 0)
        {
            P("  activated     :");
            foreach (var n in act.Nodes.Take(maxNodes)) P($"      {n.Name,-18} score {n.Score:F2}  obs {n.ObservationCount}{(n.IsAnchor ? "  [anchor]" : "")}");
        }
        if (act.Edges.Length > 0)
        {
            P("  relations hit :");
            foreach (var e in act.Edges.Take(maxEdges)) P($"      {e.Left,-16} <-> {e.Right,-16} score {e.Score:F2}  obs {e.ObservationCount}");
        }
    }
    catch (Exception ex)
    {
        P($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    }
}

async Task ProbeAsync()
{
    H("CAPABILITY PROBE  (✓ correct · ✗ wrong · ? no ground truth — read the decision path)");
    var probes = new (string Cat, string In, string? Exp)[]
    {
        ("arith:add",          "12 + 7",        "19"),
        ("arith:sub",          "15 - 6",        "9"),
        ("arith:mul",          "6 x 8",         "48"),
        ("arith:div",          "36 / 4",        "9"),
        ("arith:extrapolate",  "84 + 57",       "141"),  // out of typical training range → tests the homomorphism
        ("equivalence",        "three",         null),
        ("equivalence",        "7",             null),
        ("retrieval",          "apple",         null),
        ("predicate",          "compare 7 3",   null),
        ("fold",               "sum 2 3 4",     null),
    };
    var hit = 0; var graded = 0;
    foreach (var (cat, inp, exp) in probes)
    {
        string got, path;
        try
        {
            var pred = await runtime.PredictAsync(inp, maxTokens: 12);
            got = pred.Result!.Output.Trim();
            path = pred.Result!.DecisionPath;
        }
        catch (Exception ex) { got = $"<{ex.GetType().Name}>"; path = "error"; }

        string mark = "  ?";
        if (exp is not null)
        {
            graded++;
            var ok = AnswerEquivalence.Equivalent(got, exp);
            if (ok) hit++;
            mark = ok ? "  ✓" : "  ✗";
        }
        var expNote = exp is null ? "" : $"   expected \"{exp}\"";
        P($"{mark} {cat,-18} \"{inp,-12}\" -> \"{got,-14}\"   [{path}]{expNote}");
    }
    P();
    P($"  graded probes: {hit}/{graded} correct (the ungraded rows show capability + route, not pass/fail).");
}

// GYM-MATCHED PROBE: mirrors GymTrainer's level-L curriculum and grades PER SKILL with the REAL GenesisGrader
// (value-aware, filler-tolerant, competing-answer penalized — the SAME grader the gym trains against), so the
// reading matches what the gym sees. Uses GymTrainer's PUBLIC natural-language frames so framing can't drift.
// A "hit" = grader quality >= 0.5; mean quality shown too (degrees of correctness). Deterministic, read-only.
async Task GymProbeAsync(int level)
{
    H($"GYM CAPABILITY PROBE — level {level} curriculum (per-skill quality + dominant route)");
    var rng = new Random(20260618);
    int maxN = 9 + level * 5;
    int wordCap = Math.Min(20, 9 + level);
    var words = NumberWordVocabulary.Entries.Where(e => e.Value <= wordCap).ToArray();
    string? WordOf(int v) => NumberWordVocabulary.Entries.FirstOrDefault(e => e.Value == v).Word;
    const int G = 8; // fresh generalizing instances per skill
    string Frame(string[] f, params object[] a) => string.Format(f[rng.Next(f.Length)], a);

    // REAL facts (same source + scaling as GymTrainer) so the probe matches what the gym trains.
    var synGroups = GymLanguageFacts.SynonymGroups.Take(Math.Min(GymLanguageFacts.SynonymGroups.Length, 4 + level)).ToArray();
    var synWords = synGroups.SelectMany(g => g).Distinct().ToArray();
    var cats = GymLanguageFacts.Categories.Take(Math.Min(GymLanguageFacts.Categories.Length, 8 + level * 2)).ToArray();
    var catVocab = cats.Select(c => c.Category).Distinct().ToArray();

    // each item: framed query, valid answers, optional competing-answer vocabulary, surface-strict format flag
    var skills = new List<(string Cat, List<(string Q, string[] Allowed, string[]? Vocab, bool Strict)> Items)>();
    void Add(string cat, List<(string, string[], string[]?, bool)> items) => skills.Add((cat, items));
    List<(string, string[], string[]?, bool)> New() => new();

    var syn = New();
    foreach (var g in synGroups)
        foreach (var cue in g)
            syn.Add((Frame(GymTrainer.SynonymFrames, cue), g.Where(w => w != cue).ToArray(), synWords.Where(w => !g.Contains(w)).ToArray(), false)); // any other member correct; other-group = wrong
    Add("synonym (any of group)", syn);
    var cat = New(); foreach (var (item, category) in cats) cat.Add((Frame(GymTrainer.CategoryFrames, item), new[] { category }, catVocab, false));
    Add("category (item->kind)", cat);
    var nw1 = New(); foreach (var (v, w) in words) nw1.Add((Frame(GymTrainer.DigitWordFrames, v), new[] { w }, null, true)); Add("numword digit->word", nw1);
    var nw2 = New(); foreach (var (v, w) in words) nw2.Add((Frame(GymTrainer.WordDigitFrames, w), new[] { v.ToString() }, null, true)); Add("numword word->digit", nw2);

    var add = New(); for (int i = 0; i < G; i++) { int x = rng.Next(0, maxN + 1), y = rng.Next(0, maxN + 1); add.Add(($"{x} + {y}", new[] { (x + y).ToString() }, null, false)); } Add("add", add);
    var sub = New(); for (int i = 0; i < G; i++) { int x = rng.Next(0, maxN + 1), y = rng.Next(0, x + 1); sub.Add(($"{x} - {y}", new[] { (x - y).ToString() }, null, false)); } Add("sub", sub);
    var mul = New(); for (int i = 0; i < G; i++) { int x = rng.Next(1, maxN + 1), y = rng.Next(1, maxN + 1); mul.Add(($"{x} x {y}", new[] { (x * y).ToString() }, null, false)); } Add("mul", mul);
    var a2w = New(); for (int i = 0; i < G; i++) { int x = rng.Next(0, 11), y = rng.Next(0, 20 - x + 1); var s = x + y; a2w.Add((Frame(GymTrainer.ArithWordFrames, x, y), new[] { WordOf(s) ?? s.ToString() }, null, true)); } Add("arith->word", a2w);
    var foldA = New(); for (int i = 0; i < G; i++) { int k = Math.Max(3, 3 + level / 8); var xs = Enumerable.Range(0, k).Select(_ => rng.Next(0, maxN + 1)).ToArray(); foldA.Add((string.Join(" + ", xs), new[] { xs.Sum().ToString() }, null, false)); } Add("fold-add (a+b+c)", foldA);
    var foldM = New(); for (int i = 0; i < G; i++) { int k = Math.Max(3, 3 + level / 8); int cap2 = Math.Max(2, maxN / 3); var xs = Enumerable.Range(0, k).Select(_ => rng.Next(1, cap2 + 1)).ToArray(); foldM.Add((string.Join(" x ", xs), new[] { xs.Aggregate(1L, (s, v) => s * v).ToString() }, null, false)); } Add("fold-mul (a x b x c)", foldM);
    var expr = New(); for (int i = 0; i < G; i++) { int a = rng.Next(1, Math.Max(2, maxN / 2) + 1), b = rng.Next(1, Math.Max(2, maxN / 3) + 1); int prod = a * b; int c = rng.Next(0, maxN + 1); expr.Add(($"{a} x {b} + {c}", new[] { (prod + c).ToString() }, null, false)); } Add("expr (a x b + c)", expr);
    var pred = New(); for (int i = 0; i < G; i++) { int x = rng.Next(0, maxN + 1), y = rng.Next(0, maxN + 1); pred.Add(($"{x} compared to {y}", new[] { x > y ? "greater" : x < y ? "less" : "equal" }, GymTrainer.PredicateVocab, false)); } Add("predicate (compared to)", pred);
    var wadd = New(); for (int i = 0; i < G; i++) { int x = rng.Next(1, maxN + 1), y = rng.Next(1, maxN + 1); wadd.Add(($"what is {x} plus {y}", new[] { (x + y).ToString() }, null, false)); } Add("worded add", wadd);
    var fns = New();
    for (int i = 0; i < G; i++)
    {
        bool m = rng.Next(2) == 0; int kk = m ? rng.Next(2, 5) : rng.Next(2, 7);
        long F(long v) => m ? v * kk : v + kk;
        var ops = new HashSet<int>(); while (ops.Count < 4) ops.Add(rng.Next(1, 13)); var a = ops.ToArray();
        var demos = string.Join(" ", a.Take(3).Select(o => $"fn {o} is {F(o)}"));
        fns.Add(($"{demos} fn {a[3]} is", new[] { F(a[3]).ToString() }, null, false));
    }
    Add("function induction (few-shot)", fns);

    int totHit = 0, totN = 0; double totQ = 0;
    var failing = new List<string>();
    foreach (var (catName, items) in skills)
    {
        int hit = 0; double sumQ = 0; var routes = new Dictionary<string, int>(); var ex = new List<string>();
        foreach (var (q, allowed, vocab, strict) in items)
        {
            string got, path;
            try { var pr = await runtime.PredictAsync(q, maxTokens: 12); got = pr.Result!.Output.Trim(); path = pr.Result!.DecisionPath; }
            catch (Exception e) { got = $"<{e.GetType().Name}>"; path = "error"; }
            // Real grader, capability mode (requirePlatonic:false) — the route column separately shows platonic vs neural.
            double quality = GenesisGrader.Quality(got, allowed, requiredDepth: 1, usedNeuralFallback: false, requirePlatonic: false, answerVocabulary: vocab, surfaceStrict: strict);
            sumQ += quality;
            if (quality >= 0.5) hit++;
            else if (ex.Count < 3) ex.Add($"\"{q}\" -> \"{got}\"  (want \"{string.Join("/", allowed)}\")  q={quality:F2}  [{path}]");
            var rk = path.Split(' ')[0]; routes[rk] = routes.GetValueOrDefault(rk) + 1;
        }
        totHit += hit; totN += items.Count; totQ += sumQ;
        double acc = items.Count > 0 ? (double)hit / items.Count : 0.0;
        double meanQ = items.Count > 0 ? sumQ / items.Count : 0.0;
        var flag = acc >= 0.80 ? "MASTERED " : acc <= 0.20 ? "FAILING  " : "weak     ";
        if (acc <= 0.20) failing.Add(catName);
        var routeStr = string.Join(", ", routes.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}x{kv.Value}"));
        P($"  [{flag}] {catName,-26} {hit,2}/{items.Count,-2} ({acc:P0})  q~{meanQ:F2}   {routeStr}");
        foreach (var e in ex) P($"        - {e}");
    }
    P();
    P($"  OVERALL: {totHit}/{totN} = {(double)totHit / Math.Max(1, totN):P1}  (mean quality {totQ / Math.Max(1, totN):F2}; gym mastery bar 0.80; level advances only when the WHOLE mix holds >=0.80)");
    if (failing.Count > 0) P($"  FAILING COMPLETELY (<=20%): {string.Join(", ", failing)}");
}

switch (cmd)
{
    case "query":
        if (rest.Count == 0) { P("usage: query <text>"); break; }
        await QueryAsync(string.Join(' ', rest));
        break;
    case "probe":
        Report();
        await ProbeAsync();
        break;
    case "gymprobe":
    {
        var lvl = rest.Count > 0 && int.TryParse(rest[0], out var lv) ? lv : 1;
        Report();
        await GymProbeAsync(lvl);
        break;
    }
    case "space":
        if (rest.Count > 0) await QueryAsync(string.Join(' ', rest), maxNodes: 24, maxEdges: 24);
        else Report();
        break;
    case "geometry":
    {
        H("PUSH / PULL GEOMETRY  (semantic-face distance; faces unit-normalised → range [0, 2])");
        var g = runtime.GeometrySummary();
        P($"  concepts        : {g.TotalConcepts}  (mutable, non-frozen: {g.MutableConcepts})");
        P($"  PULL (related)  : mean {g.RelatedMean:F3}   min {g.RelatedMin:F3}   max {g.RelatedMax:F3}   (n={g.RelatedPairs} edged pairs)");
        P($"  PUSH (unrelated): mean {g.UnrelatedMean:F3}   min {g.UnrelatedMin:F3}   max {g.UnrelatedMax:F3}   (n={g.UnrelatedPairs} sampled pairs)");
        P($"  SEPARATION      : {g.Separation:F3}   (unrelated mean − related mean; >0 means push/pull pulled related closer than unrelated)");
        P($"  reference       : 0 = identical, ~1.41 = orthogonal, 2 = antipodal");
        break;
    }
    case "report":
    default:
        Report();
        await ProbeAsync();
        break;
}

P();
P("done.");
