using GenesisNova.Core;
using GenesisNova.Runtime;

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
    case "space":
        if (rest.Count > 0) await QueryAsync(string.Join(' ', rest), maxNodes: 24, maxEdges: 24);
        else Report();
        break;
    case "report":
    default:
        Report();
        await ProbeAsync();
        break;
}

P();
P("done.");
