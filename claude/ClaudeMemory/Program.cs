using System.Text.Json;
using System.Text.RegularExpressions;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Persistence;
using GenesisNova.Runtime;

// ════════════════════════════════════════════════════════════════════════════════════════════════════
//  ClaudeMemory — a project-local, Claude-owned Genesis-Nova ASSOCIATIVE INDEX over the file memory.
//
//  Role (narrow): the flat file memory (MEMORY.md + files) is the SOURCE OF TRUTH — exact, durable, never
//  forgetting. Nova is NOT the store; it is a fuzzy associative INDEX that, for a vague query, points back to
//  the relevant memory KEY(S), which Claude then opens. Content lives in files; Nova only does the reach a
//  keyword grep misses. The index is a PURE FUNCTION of the file-memory index (recomputed, never hand-stored).
//
//  DECOUPLED / ASYNC: run `serve` once (in a window) and it trains in the BACKGROUND — watching MEMORY.md and
//  a fact queue — so Claude never blocks on training. Drop a fact with `enqueue` (instant, no model load) or
//  just edit MEMORY.md; the daemon trains it and keeps the checkpoint current; `recall` reads it any time.
//
//  Usage (from repo root):
//    dotnet run --project claude/ClaudeMemory -c Release -- serve  --index "<MEMORY.md>"   # background trainer
//    dotnet run --project claude/ClaudeMemory -c Release -- enqueue "<cue>" "<response>" [relate]   # instant
//    dotnet run --project claude/ClaudeMemory -c Release -- recall  "<query>"
//    dotnet run --project claude/ClaudeMemory -c Release -- rebuild --index "<MEMORY.md>"  # one-shot fresh
//    dotnet run --project claude/ClaudeMemory -c Release -- stats | log [N] | help
//  Flags: --index <file> (default $CLAUDE_MEMORY_FILE, else claude/truth/memory.truth), --interval N (serve, s),
//         --gpu (default CPU), --reps N, --dir <state-dir>.
// ════════════════════════════════════════════════════════════════════════════════════════════════════

var a = args.ToList();
var useGpu = a.Remove("--gpu");
var repsOverride = (int?)null;
var ri = a.IndexOf("--reps");
if (ri >= 0 && ri + 1 < a.Count && int.TryParse(a[ri + 1], out var rp)) { repsOverride = Math.Max(1, rp); a.RemoveAt(ri + 1); a.RemoveAt(ri); }
var interval = 30;
var ivi = a.IndexOf("--interval");
if (ivi >= 0 && ivi + 1 < a.Count && int.TryParse(a[ivi + 1], out var iv)) { interval = Math.Max(2, iv); a.RemoveAt(ivi + 1); a.RemoveAt(ivi); }
string? dirOverride = null;
var di = a.IndexOf("--dir");
if (di >= 0 && di + 1 < a.Count) { dirOverride = a[di + 1]; a.RemoveAt(di + 1); a.RemoveAt(di); }
string? indexOverride = null;
var ii = a.IndexOf("--index");
if (ii >= 0 && ii + 1 < a.Count) { indexOverride = a[ii + 1]; a.RemoveAt(ii + 1); a.RemoveAt(ii); }
var cmd = a.Count > 0 ? a[0].ToLowerInvariant() : "help";
var rest = a.Skip(1).ToList();

void P(string s = "") => Console.WriteLine(s);

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

var root = RepoRoot();
var stateDir = dirOverride ?? Path.Combine(root, ".claude-nova");
Directory.CreateDirectory(stateDir);
var logPath = Path.Combine(stateDir, "interaction-log.jsonl");
var queuePath = Path.Combine(stateDir, "queue.jsonl");
var metricsPath = Path.Combine(stateDir, "metrics.jsonl"); // per-cycle held-out learning curve (continuous mastery)
var pidPath = Path.Combine(stateDir, "daemon.pid"); // written by serve so `watch` can report daemon liveness
var checkpoint = Path.Combine(stateDir, "genesis-nova.autosave.checkpoint.json"); // == GenesisLocalStateStore.ResolveCheckpointPath
// The index the checkpoint was last built from is remembered in index-path.txt, so a bare `recall`
// (no --index) uses the SAME index the daemon trained on (needed for memory-name filtering in 2-hop recall).
var indexPointer = Path.Combine(stateDir, "index-path.txt");
var pointed = File.Exists(indexPointer) ? File.ReadAllText(indexPointer).Trim() : null;
// Default to the real file memory (MEMORY.md). The self-managing daemon (EnsureDaemonRunning) and bare `recall`
// have no --index, so the DEFAULT must be MEMORY.md — not the tiny repo `memory.truth` fallback, which silently
// trained the wrong index. Order: explicit --index → $CLAUDE_MEMORY_FILE → remembered index-path.txt → MEMORY.md.
var defaultMemoryMd = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
    ".claude", "projects", "C--Users-dongy", "memory", "MEMORY.md");
var indexPath = indexOverride
    ?? Environment.GetEnvironmentVariable("CLAUDE_MEMORY_FILE")
    ?? (!string.IsNullOrWhiteSpace(pointed) && File.Exists(pointed) ? pointed : null)
    ?? (File.Exists(defaultMemoryMd) ? defaultMemoryMd : Path.Combine(root, "claude", "truth", "memory.truth"));
var reps = repsOverride ?? (cmd is "rebuild" or "serve" ? 5 : 8);

void LogInteraction(string command, string? cue, string? response, string? output)
    => File.AppendAllText(logPath, JsonSerializer.Serialize(new InteractionLogEntry(DateTimeOffset.UtcNow, command, cue, response, output)) + Environment.NewLine);

var stop = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
{
    "the","and","via","not","with","into","that","from","for","its","now","was","are","but","this","than",
    "then","per","each","you","your","yourself","when","which","what","how","why","can","could","should",
    "would","must","dont","don","over","under","onto","off","out","all","any","more","most","they","them",
    "their","get","got","use","used","using","run","runs","running","keep","kept","made","make","makes",
    "one","two","new","old","see","read","added","add","like","does","did","has","have",
};
var indexLine = new Regex(@"^\s*-\s*\[(?<name>[^\]]+)\]\([^)]+\)\s*(?:—|--|-)\s*(?<desc>.+)$");

// ── Code indexer (combined): the daemon also indexes the C# codebase so `find <symbol/concept>` resolves to a
// file:line. Fuzzy + "ever-living": each cycle re-includes the CURRENT symbols (cached by file-mtime; stale
// ones lose relevance as training moves on). code-manifest.tsv is the SOURCE OF TRUTH for locations.
var manifestPath = Path.Combine(stateDir, "code-manifest.tsv");
var sigPath = Path.Combine(stateDir, "trainset.sig"); // signature the checkpoint was built from (memory+code)
var trainFilePath = Path.Combine(stateDir, "trainset.txt"); // facts as `cue => response` for batched TrainAsync
var chunkFilePath = Path.Combine(stateDir, "trainchunk.txt"); // one small batch (chunk) trained per cycle
var breakPath = Path.Combine(stateDir, "break-percent.txt"); // live throttle: rest = batch_time × (break%/100)
var reType = new Regex(@"\b(?:class|record|struct|interface|enum)\s+([A-Za-z_]\w*)");
var reMember = new Regex(@"^\s*public\s+(?:static\s+|async\s+|virtual\s+|override\s+|sealed\s+|abstract\s+|readonly\s+|partial\s+|new\s+|unsafe\s+|extern\s+)*[\w<>\[\],\.\?]+\s+([A-Za-z_]\w*)\s*[\(\{]");
var reCamel = new Regex(@"(?<=[a-z0-9])(?=[A-Z])|[_\s]+");
var reCall = new Regex(@"([A-Za-z_]\w*)\s*\(");
var codeFactsCache = new List<Fact>();
var codeFactsStamp = -1L;

// Extract queryable terms from a memory's description. Indexes EVERY meaningful term (not just the first
// few), and splits hyphenated compounds so a query for any component matches: "router-confidence" yields
// "router-confidence" + "router" + "confidence"; "number-word" yields "number" + "word". This is the core
// of recall coverage — a term missed here is a memory you can never reach by that word.
IEnumerable<string> Keywords(string text)
{
    var terms = new List<string>();
    void Consider(string t) { if (t.Length >= 3 && !stop.Contains(t)) terms.Add(t); }
    foreach (Match m in Regex.Matches(text.ToLowerInvariant(), "[a-z0-9][a-z0-9-]*"))
    {
        var tok = m.Value.Trim('-');
        if (tok.Length == 0) continue;
        Consider(tok); // whole term, including compounds
        if (tok.Contains('-'))
            foreach (var part in tok.Split('-', StringSplitOptions.RemoveEmptyEntries))
                Consider(part); // and each hyphen component
    }
    return terms.Distinct(StringComparer.OrdinalIgnoreCase).Take(20);
}

// QUERY SYNTAX (ONE canonical form). Genesis is a programming language, not an LLM — it wants ONE exact query
// syntax and generalises the OPERAND, exactly as arithmetic trains one form `a + b` and generalises to unseen
// operands (you don't teach six ways to phrase "add"). The lead verb `find` is the constant op-token: it recurs
// on every query so it washes to a pure route-trigger, while the topic keyword is the swappable operand that
// carries the signal. So `find <unseen-topic>` routes via the learned syntax — operand generalisation, not
// phrasing memorisation. Recall issues queries in THIS syntax.
string[] queryTemplates = { "find {0}" };

// The op-token of each query syntax (the leading verb) is a ROUTE TRIGGER, not a relation concept. Register
// it on every runtime so the trainer/inference exclude it from relation coupling + retrieval anchoring (it
// would otherwise edge to every key and collapse recall to one answer). Re-registered each startup — it is a
// pure function of the language definition, so it is not persisted. See LANGUAGE_CREATOR.md §2.
void RegisterOps(GenesisEvalAppRuntime rt)
{
    // Op-tokens = route triggers (excluded from relation-edge formation; the GRU still conditions on them).
    // `find` (fuzzy concept→key) + the code-structure verbs `contains`/`calls`. The model learns each verb's
    // behaviour from accurate verb-prefixed training examples — typing is composed, not a core kind field.
    var ops = queryTemplates.Select(t => t.Split(' ', 2)[0])
        .Concat(new[] { "contains", "calls" })
        .Distinct(StringComparer.OrdinalIgnoreCase);
    foreach (var op in ops) rt.RegisterOperationToken(op);
}

// Memory keys are hyphenated filenames (nova-claude-memory) — heavily OVERLAPPING multi-token targets the
// decoder can't construct (it over-emits the shared nova-/-guide tokens and repeats them). Train them as UNIQUE
// SINGLE TOKENS by stripping hyphens (novaclaudememory); recall maps the single token back to the file by
// Canon-match. Distinct single tokens are 100%-learnable (Tests/AssociativeIndexLearningTests); phrases (spaces)
// and paths/symbols (no hyphen) are untouched.
static string Key(string v) => string.IsNullOrEmpty(v) || v.Contains(' ') ? v : v.Replace("-", string.Empty);

List<Fact> GenerateTruth()
{
    var facts = new List<Fact>();
    var memDir = File.Exists(indexPath) ? Path.GetDirectoryName(indexPath) : null;
    if (File.Exists(indexPath))
        foreach (var raw in File.ReadLines(indexPath))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var m = indexLine.Match(line);
            if (m.Success)
            {
                var name = m.Groups["name"].Value.Trim();
                var kws = Keywords(m.Groups["desc"].Value).ToList();
                // EVERY keyword as a `find <keyword>` query — SYNTAX on every example, never a bare cue. A
                // keyword that hits multiple memories is aggregated into a comma-list (fuzzy) by WriteTrainFile.
                foreach (var kw in kws)
                    facts.Add(new Fact($"find {kw}", Key(name), false));
                // Memory<->memory edges from [[links]] in the entry's file: an associative graph (Nova's
                // strength) so recall surfaces RELATED memories, not just keyword hits. Bidirectional relate.
                if (memDir is not null)
                {
                    var file = Path.Combine(memDir, name + ".md");
                    if (File.Exists(file))
                        foreach (Match lm in Regex.Matches(File.ReadAllText(file), @"\[\[([^\]]+)\]\]"))
                        {
                            var target = lm.Groups[1].Value.Trim();
                            if (target.Length > 0 && !target.Equals(name, StringComparison.OrdinalIgnoreCase))
                            {
                                facts.Add(new Fact($"find {Key(name)}", Key(target), false));  // find <memory> -> its linked memories
                                facts.Add(new Fact($"find {Key(target)}", Key(name), false));  // both directions, single-token keys
                            }
                        }
                }
            }
            else if (line.Contains("<=>")) { var p = line.Split("<=>", 2); facts.Add(new Fact(Key(p[0].Trim()), Key(p[1].Trim()), true)); }
            else if (line.Contains("=>")) { var p = line.Split("=>", 2); facts.Add(new Fact(Key(p[0].Trim()), Key(p[1].Trim()), false)); }
        }
    var claudeDir = Path.Combine(root, "claude");
    if (Directory.Exists(claudeDir))
        foreach (var dir in Directory.GetDirectories(claudeDir))
            if (Directory.GetFiles(dir, "*.csproj").Length > 0)
                facts.Add(new Fact($"{Path.GetFileName(dir)} tool", $"claude/{Path.GetFileName(dir)}", false));

    // AUGMENTATION: hand-authored Q->key and memory<->memory cross-links that enrich the graph beyond what
    // the descriptions alone capture (the "training-focused linking memories"). Trained alongside MEMORY.md.
    var bridges = Path.Combine(claudeDir, "truth", "bridges.truth");
    if (File.Exists(bridges))
        foreach (var raw in File.ReadLines(bridges))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            if (line.Contains("<=>")) { var p = line.Split("<=>", 2); facts.Add(new Fact(Key(p[0].Trim()), Key(p[1].Trim()), true)); }
            else if (line.Contains("=>")) { var p = line.Split("=>", 2); facts.Add(new Fact(Key(p[0].Trim()), Key(p[1].Trim()), false)); }
        }
    facts.AddRange(GenerateCodeFacts()); // combined: memory associations + C# code symbols in one model
    return facts;
}

// The C# files we index (src/ + claude/, excluding bin/obj).
IEnumerable<string> CodeFiles()
{
    foreach (var dirName in new[] { "src", "claude" })
    {
        var dir = Path.Combine(root, dirName);
        if (!Directory.Exists(dir)) continue;
        foreach (var f in Directory.EnumerateFiles(dir, "*.cs", SearchOption.AllDirectories))
            if (!f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
                yield return f;
    }
}

// Aggregate mtime signature of the code — changes whenever any indexed .cs changes.
long CodeStamp()
{
    var s = 0L;
    foreach (var f in CodeFiles()) unchecked { s = (s * 1000003) + File.GetLastWriteTimeUtc(f).Ticks; }
    return s;
}

// Code facts: scan src/ + claude/ *.cs and model the codebase RELATIONALLY in the same model:
//   • keyword/name → Symbol          (lookup: `find <concept>` → symbol → file:line via the manifest)
//   • EnclosingType ⟷ Member         (containment: a file/type contains its functions)
//   • Caller ⟷ Callee                (call graph: functions call functions)
// code-manifest.tsv (Symbol → file:line) is the source of truth for locations. Cached by CodeStamp so cycles
// don't rescan; refreshes (ever-living) when any .cs changes.
List<Fact> GenerateCodeFacts()
{
    var stamp = CodeStamp();
    if (stamp == codeFactsStamp) return codeFactsCache;

    var fileLines = new List<(string Rel, string[] Lines)>();
    foreach (var f in CodeFiles())
        try { fileLines.Add((Path.GetRelativePath(root, f).Replace('\\', '/'), File.ReadAllLines(f))); } catch { }

    var facts = new List<Fact>();
    var manifest = new List<string>();
    var symbols = new HashSet<string>(StringComparer.Ordinal); // every symbol name (for call resolution)
    var declared = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    // PASS 1 — declarations, keyword facts, and TYPE⟷member containment relations.
    foreach (var (rel, lines) in fileLines)
    {
        var doc = new List<string>();
        string? currentType = null;
        for (var i = 0; i < lines.Length; i++)
        {
            var t = lines[i].TrimStart();
            if (t.StartsWith("///")) { doc.Add(t.TrimStart('/').Trim()); continue; }
            string? sym = null; var isType = false;
            var tm = reType.Match(lines[i]);
            if (tm.Success) { sym = tm.Groups[1].Value; isType = true; }
            else { var mm = reMember.Match(lines[i]); if (mm.Success) sym = mm.Groups[1].Value; }
            if (sym is not null && sym.Length >= 3)
            {
                symbols.Add(sym);
                if (isType) currentType = sym;
                if (declared.Add(sym))
                {
                    manifest.Add($"{sym}\t{rel}:{i + 1}");
                    facts.Add(new Fact($"find {sym.ToLowerInvariant()}", sym, false)); // find <name> -> the symbol
                    var nameKw = reCamel.Split(sym).Where(s => s.Length >= 3).Select(s => s.ToLowerInvariant());
                    var docKw = doc.SelectMany(d => Regex.Matches(d.ToLowerInvariant(), "[a-z][a-z0-9]{2,}").Select(m => m.Value));
                    foreach (var kw in nameKw.Concat(docKw).Where(k => !stop.Contains(k)).Distinct().Take(4))
                        facts.Add(new Fact($"find {kw}", sym, false)); // find <concept> -> the symbol
                }
                if (!isType && currentType is not null && !currentType.Equals(sym, StringComparison.OrdinalIgnoreCase))
                    facts.Add(new Fact($"contains {currentType}", sym, false)); // `contains <Type>` => member (WriteTrainFile aggregates to a list)
            }
            doc.Clear();
        }
    }

    // PASS 2 — call graph: caller ⟷ callee, both known symbols, deduped + bounded.
    var callPairs = new HashSet<string>();
    const int MaxCalls = 2500;
    foreach (var (_, lines) in fileLines)
    {
        if (callPairs.Count >= MaxCalls) break;
        string? currentMember = null;
        foreach (var raw in lines)
        {
            var t = raw.TrimStart();
            if (t.StartsWith("//")) continue;
            if (reType.IsMatch(raw)) { currentMember = null; continue; }
            var mm = reMember.Match(raw);
            if (mm.Success) currentMember = mm.Groups[1].Value;
            if (currentMember is null) continue;
            foreach (Match cm in reCall.Matches(raw))
            {
                var callee = cm.Groups[1].Value;
                if (callee.Length < 3 || callee.Equals(currentMember, StringComparison.Ordinal) || !symbols.Contains(callee)) continue;
                if (callPairs.Add($"{currentMember}|{callee}"))
                    facts.Add(new Fact($"calls {currentMember}", callee, false)); // `calls <Caller>` => callee (WriteTrainFile aggregates)
                if (callPairs.Count >= MaxCalls) break;
            }
        }
    }

    try { File.WriteAllLines(manifestPath, manifest); } catch { /* best effort */ }
    codeFactsStamp = stamp;
    codeFactsCache = facts;
    return facts;
}

// Signature of the whole training set (MEMORY.md mtime + code mtime). When this changes, the checkpoint is
// stale and must be RESEEDED — this is what makes the daemon pick up a code scan (not just MEMORY.md edits).
long TrainsetSig()
{
    var idx = File.Exists(indexPath) ? File.GetLastWriteTimeUtc(indexPath).Ticks : 0L;
    unchecked { return (idx * 31) + CodeStamp(); }
}

// Evaluation probes — query → expected memory key. The mastery signal. Two kinds:
//   • HELD-OUT operands: `find <kw>` for keywords the find-SYNTAX was never trained on (only the bare
//     keyword→key edge exists), so a correct hit proves the syntax GENERALIZES to an unseen operand — the
//     whole point (operand generalization under one fixed syntax). These are skipped from find-form training.
//   • in-sample sanity: `find <top-kw>` (the operand IS trained in find-form) + bridge questions.
// Probes are NEVER trained directly (that would defeat held-out); they only measure.
List<MemoryProbe> GenerateProbes()
{
    var probes = new List<MemoryProbe>();
    var verb = queryTemplates.Select(t => t.Split(' ', 2)[0]).FirstOrDefault() ?? "find";
    if (File.Exists(indexPath))
        foreach (var raw in File.ReadLines(indexPath))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var m = indexLine.Match(line);
            if (!m.Success) continue;
            var name = m.Groups["name"].Value.Trim();
            var kws = Keywords(m.Groups["desc"].Value).Where(k => !k.Contains('-')).ToList();
            foreach (var ho in kws.Skip(3).Take(2)) probes.Add(new MemoryProbe($"{verb} {ho}", name, true));   // held-out operand
            if (kws.Count > 0) probes.Add(new MemoryProbe($"{verb} {kws[0]}", name, false));                   // in-sample
        }
    var bridges = Path.Combine(root, "claude", "truth", "bridges.truth");
    if (File.Exists(bridges))
        foreach (var raw in File.ReadLines(bridges))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#') || line.Contains("<=>") || !line.Contains("=>")) continue;
            var p = line.Split("=>", 2);
            probes.Add(new MemoryProbe(p[0].Trim(), p[1].Trim(), false)); // realistic natural-language question
        }
    return probes;
}

// Score the probes: overall accuracy, HELD-OUT accuracy (the generalization headline), route purity
// (fraction answered via the platonic route, not neural fallback), and mean platonic confidence.
async Task<(double Acc, double Held, double Purity, double Conf)> EvaluateAsync(GenesisEvalAppRuntime rt, List<MemoryProbe> probes)
{
    if (probes.Count == 0) return (0, 0, 0, 0);
    double correct = 0, heldCorrect = 0; int heldTotal = 0, platonic = 0;
    double confSum = 0;
    foreach (var pr in probes)
    {
        var res = (await rt.PredictAsync(pr.Query, maxTokens: 16)).Result;
        if (res is null) continue;
        // Keep the answer-present check (recall): the expected key appearing ANYWHERE in the output earns the
        // base credit. OVER-GENERATION PENALTY (a SEPARATE term, not folded into the present-reward): tax every
        // token emitted beyond the key itself, so "too much response" scores strictly less than a clean hit. A
        // verbose-but-correct answer still keeps a floor of credit (the answer IS there); a concise hit scores 1.
        var present = Canon(pr.ExpectedKey).Length > 0 && Canon(res.Output).Contains(Canon(pr.ExpectedKey));
        var outToks = (res.Output ?? "").Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries).Length;
        var keyToks = Math.Max(1, pr.ExpectedKey.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries).Length);
        var excess = Math.Max(0, outToks - keyToks);               // tokens past what the answer needed
        var quality = present ? Math.Clamp(1.0 - 0.34 * excess, 0.2, 1.0) : 0.0;
        correct += quality;
        if (!res.UsedNeuralFallback) platonic++;
        confSum += res.PlatonicConfidence;
        if (pr.HeldOut) { heldTotal++; heldCorrect += quality; }
    }
    var n = probes.Count;
    return (correct / n, heldTotal > 0 ? heldCorrect / heldTotal : correct / n, platonic / (double)n, confSum / n);
}

GenesisNovaConfig MakeConfig(bool resume) => new(
    HiddenSize: 1024, // controller width / platonic face dim — bumped from the 512 default for more capacity
    Backend: useGpu ? ComputeBackend.Gpu : ComputeBackend.Cpu,
    AutoResume: resume, AutoPersist: true, LocalStateDirectory: stateDir);

async Task<double> TrainFactAsync(GenesisEvalAppRuntime rt, Fact f, int? repsOverride = null)
{
    var n = Math.Max(1, repsOverride ?? reps);
    var last = 0.0;
    for (var i = 0; i < n; i++)
    {
        last = (await rt.TrainOneAsync(new GenesisExample(f.Cue, f.Response))).TotalLoss;
        if (f.Relate) await rt.TrainOneAsync(new GenesisExample(f.Response, f.Cue));
    }
    return last;
}

// Build the training set as `cue => answer` lines — but FUZZY where the answer isn't deterministic. Every cue
// collects its FULL set of valid responses (relate facts both directions); a cue with multiple answers becomes
// a comma-separated LIST (in a few shuffled orderings so it's unordered), a single-answer cue stays single.
// Applied uniformly to find / contains / calls — none is forced to one deterministic answer when it has many.
const int MaxAnswersPerCue = 3;  // train only a FEW edges per one-to-many cue. Training MANY single-targets for
                                 // the SAME input (calls X => 8 different callees) is contradictory decoder
                                 // supervision — fan-out conflict that DEGRADED retrieval as depth rose (acc went
                                 // 100%->0% from d2->d8). A few clean edges retrieve reliably; the rest emerge as
                                 // valid OVER-GENERATION (the list, learned for free) which grading credits.
// Full per-cue answer sets (uncapped, deduped, both directions) = the curriculum's SOURCE OF TRUTH. Each bucket
// emits + grades at its OWN difficulty by capping each cue's list, so difficulty is tracked per bucket.
Dictionary<string, List<string>> BuildCueAnswers()
{
    var byCue = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
    void Add(string cue, string resp)
    {
        cue = cue.Trim(); resp = resp.Trim();
        if (cue.Length == 0 || resp.Length == 0) return;
        if (!byCue.TryGetValue(cue, out var l)) byCue[cue] = l = new List<string>();
        if (!l.Contains(resp, StringComparer.OrdinalIgnoreCase)) l.Add(resp);
    }
    foreach (var f in GenerateTruth())
    {
        Add(f.Cue, f.Response);
        if (f.Relate) Add(f.Response, f.Cue);
    }
    return byCue;
}
// ONE SINGLE-ANSWER LINE PER MEMBER. A one-to-many relation (a Type's members, a method's callees, a keyword
// hitting several memories) is held as MANY single edges in the platonic space — its native strength — NOT as a
// comma-list target. The list form was unlearnable: long to generate, CONTRADICTORY across shuffled orderings
// (same input → different token sequence), and unroutable as a single platonic retrieval, and it polluted the
// shared space/decoder so even clean single-answer cues collapsed. Single targets retrieve + route cleanly.
List<string> CueLines(string cue, IReadOnlyList<string> all, int maxAnswers)
{
    var lines = new List<string>();
    foreach (var ans in all.Take(Math.Clamp(maxAnswers, 1, MaxAnswersPerCue)))
        lines.Add($"{cue} => {ans}");
    return lines;
}
// trainset.txt artifact (reference / cold tools) at the given cap. The serve loop drives buckets directly.
List<string> WriteTrainFile(int maxAnswers)
{
    var byCue = BuildCueAnswers();
    var lines = new List<string>();
    foreach (var (cue, all) in byCue) lines.AddRange(CueLines(cue, all, maxAnswers));
    try { File.WriteAllLines(trainFilePath, lines); } catch { /* best effort */ }
    return lines;
}

// Self-managing: ensure a serve daemon is alive (pidfile + live process); if not, spawn one DETACHED in a
// visible window with native stderr → log. This is the ONLY place that launches the daemon, called from the
// query verbs — so querying the memory ensures it, and there is no separate hook/shortcut/manual-launch to
// drift. Never called from `serve` itself (no self-spawn).
void EnsureDaemonRunning()
{
    if (File.Exists(pidPath) && int.TryParse(File.ReadAllText(pidPath).Trim(), out var dp))
    {
        try { var pr = System.Diagnostics.Process.GetProcessById(dp); if (!pr.HasExited && pr.ProcessName.StartsWith("ClaudeMemory", StringComparison.OrdinalIgnoreCase)) return; }
        catch { /* stale pid */ }
    }
    var self = Environment.ProcessPath ?? Path.Combine(AppContext.BaseDirectory, "ClaudeMemory.exe");
    var errLog = Path.Combine(stateDir, "daemon.stderr.log");
    var cmdFile = Path.Combine(Path.GetTempPath(), "claude-memory-daemon.cmd");
    var idx = File.Exists(indexPath) ? indexPath : (pointed ?? indexPath);
    File.WriteAllText(cmdFile, $"@echo off\r\ntitle ClaudeMemory daemon\r\n\"{self}\" serve --gpu --index \"{idx}\" 2>\"{errLog}\"\r\n");
    try { System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo { FileName = cmdFile, UseShellExecute = true, WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal }); }
    catch { /* best effort — recall still works cold from the checkpoint */ }
}

// Canonical form for matching a recalled output against an expected key (drop spacing/punctuation the decoder
// inserts: "nova- retention- diagnosis" == "nova-retention-diagnosis").
static string Canon(string s) => new(((s ?? string.Empty).ToLowerInvariant()).Where(char.IsLetterOrDigit).ToArray());

// Live training throttle (read FRESH each cycle so `break <pct>` takes effect with no restart): the rest
// between batches = batch_time × (break%/100). 0 = full throttle (~100% duty), 100 = current ~50% duty,
// 200 = ~1/3 duty. Default 100.
double ReadBreakPercent()
{
    try { if (File.Exists(breakPath) && double.TryParse(File.ReadAllText(breakPath).Trim(), out var p) && p >= 0) return p; }
    catch { /* default below */ }
    return 100.0;
}

// ── Pre-runtime commands (no model load needed) ───────────────────────────────────────────────────────
if (cmd is "help" or "-h" or "--help")
{
    P("ClaudeMemory — Nova associative index over the file memory (file memory stays the source of truth)");
    P("  serve          continuous-mastery daemon: rehearse the current set toward a held-out bar, then idle;");
    P("                 rebuild on MEMORY.md change, drain the queue, log a learning curve (Ctrl+C to stop)");
    P("  metrics [N]    print the last N learning-curve rows (held-out acc / acc / route purity / confidence)");
    P("  watch [ms]     live monitor: daemon status + learning curve + recent activity, refreshing (Ctrl+C)");
    P("  break <pct>    LIVE throttle (no restart): rest = batch×(pct/100); 0=full throttle, 100=~50% duty, 200=~1/3");
    P("  enqueue \"<cue>\" \"<response>\" [relate]   instantly queue a fact for the daemon (no model load)");
    P("  rebuild        one-shot: recompute the index FRESH from MEMORY.md (drops stale associations)");
    P("  recall \"<q>\"   GRU-routed retrieval (Generate) + ENSURES the daemon is running (self-managing)");
    P("  save \"<cue>\" \"<key>\"   queue a fact (alias of enqueue) and ensure the daemon");
    P("  ensure         spawn the continuous-mastery daemon if it isn't already running");
    P("  stats | log [N] | help");
    P($"  index source : {indexPath}  ({(File.Exists(indexPath) ? "present" : "MISSING — pass --index or set $CLAUDE_MEMORY_FILE")})");
    P($"  state dir    : {stateDir}");
    return;
}

if (cmd is "ensure")
{
    EnsureDaemonRunning();
    P("ensured the continuous-mastery daemon is running (spawned if it was down).");
    return;
}

// Live training throttle — adjust the rest between batches WITHOUT restarting the daemon. The running daemon
// re-reads this each cycle. rest = batch_time × (break%/100): 0 = full throttle, 100 = ~50% duty, 200 = ~1/3.
if (cmd is "break" or "throttle")
{
    if (rest.Count < 1 || !double.TryParse(rest[0].TrimEnd('%'), out var bp) || bp < 0)
    { P($"usage: break <percent>   (0 = full throttle, 100 = ~50% duty, 200 = ~1/3 effort)   current: {ReadBreakPercent():F0}%"); return; }
    File.WriteAllText(breakPath, bp.ToString("0.##"));
    P($"break set to {bp:F0}% — the running daemon applies it on its next cycle (no restart).");
    return;
}

// Wipe the platonic space (the LEARNED relations/concepts) but KEEP the long-lived NN/GRU. The core stores the
// space in a companion file next to the NN checkpoint; deleting it resets the substrate, and on the next load
// the GRU is intact and the space relearns from training (catastrophic forgetting is expected + re-adjusts).
if (cmd is "reset-platonic" or "wipe-platonic")
{
    var companion = checkpoint + ".platonic.json";
    var existed = File.Exists(companion);
    try { if (existed) File.Delete(companion); } catch { /* best effort */ }
    try { foreach (var p in System.Diagnostics.Process.GetProcessesByName("ClaudeMemory")) if (p.Id != Environment.ProcessId) p.Kill(); } catch { /* best effort */ }
    P(existed
        ? "platonic space wiped (companion deleted) + daemon stopped. Next recall/ensure respawns it: the NN/GRU is KEPT, the substrate relearns from scratch."
        : "(no platonic companion file found — nothing to wipe; train once so the daemon writes it)");
    return;
}

if (cmd is "enqueue" or "save")
{
    if (rest.Count < 2) { P("usage: save \"<cue>\" \"<memory-key>\" [relate]"); return; }
    EnsureDaemonRunning(); // self-heal so the queued fact actually gets trained
    var relate = rest.Count >= 3 && rest[2].Equals("relate", StringComparison.OrdinalIgnoreCase);
    File.AppendAllText(queuePath, JsonSerializer.Serialize(new Fact(rest[0], rest[1], relate)) + Environment.NewLine);
    LogInteraction("enqueue", rest[0], rest[1], relate ? "relate" : "teach");
    P($"queued {(relate ? "relation" : "fact")}: \"{rest[0]}\" {(relate ? "<->" : "->")} \"{rest[1]}\"  — the daemon will train it (run `serve`).");
    return;
}

if (cmd is "watch")
{
    var refreshMs = rest.Count > 0 && int.TryParse(rest[0], out var rms) ? Math.Max(500, rms) : 2000;
    var stopW = false;
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; stopW = true; };
    while (!stopW)
    {
        try { Console.Clear(); } catch { /* redirected console */ }
        P($"ClaudeMemory — continuous-mastery monitor    {DateTimeOffset.Now:HH:mm:ss}   (Ctrl+C to exit)");
        var alive = false; var pid = "-";
        if (File.Exists(pidPath) && int.TryParse(File.ReadAllText(pidPath).Trim(), out var dp))
        {
            pid = dp.ToString();
            try { var pr = System.Diagnostics.Process.GetProcessById(dp); alive = !pr.HasExited && pr.ProcessName.StartsWith("ClaudeMemory", StringComparison.OrdinalIgnoreCase); }
            catch { alive = false; }
        }
        P($"  daemon : {(alive ? $"RUNNING (pid {pid})" : "not running — start with `serve`")}");
        P($"  index  : {indexPath}");
        P($"  state  : {stateDir}");
        P("");
        if (File.Exists(metricsPath))
        {
            P("  cycle   loss    held-out   acc   route   conf   trained");
            foreach (var line in File.ReadLines(metricsPath).TakeLast(12))
                try { var e = JsonSerializer.Deserialize<MetricsEntry>(line); if (e is not null) P($"  {e.Cycle,5}  {e.TrainLoss,6:F4}  {e.HeldOutAccuracy,6:P0}  {e.Accuracy,5:P0}  {e.RoutePurity,5:P0}  {e.Confidence,5:F2}   {e.Trained}"); }
                catch { /* skip */ }
        }
        else P("  (no cycles logged yet — the daemon writes one row per training cycle)");
        P("");
        if (File.Exists(logPath))
        {
            P("  recent:");
            foreach (var line in File.ReadLines(logPath).Where(l => l.Contains("\"serve\"")).TakeLast(5))
                try { var e = JsonSerializer.Deserialize<InteractionLogEntry>(line); if (e is not null) P($"    {e.Timestamp.ToLocalTime():HH:mm:ss}  {e.Output}"); }
                catch { /* skip */ }
        }
        for (var slept = 0; slept < refreshMs && !stopW; slept += 100) System.Threading.Thread.Sleep(100);
    }
    return;
}

if (cmd is "metrics" or "curve")
{
    if (!File.Exists(metricsPath)) { P("(no metrics yet — the continuous-mastery daemon writes one row per cycle; run `serve`)"); return; }
    var n = rest.Count > 0 && int.TryParse(rest[0], out var mn) ? mn : 24;
    P("  cycle   loss    held-out   acc   route   conf   trained");
    foreach (var line in File.ReadLines(metricsPath).TakeLast(Math.Max(1, n)))
        try { var e = JsonSerializer.Deserialize<MetricsEntry>(line); if (e is not null) P($"  {e.Cycle,5}  {e.TrainLoss,6:F4}  {e.HeldOutAccuracy,6:P0}  {e.Accuracy,5:P0}  {e.RoutePurity,5:P0}  {e.Confidence,5:F2}   {e.Trained}"); }
        catch { /* skip */ }
    return;
}

if (cmd is "log")
{
    var n = rest.Count > 0 && int.TryParse(rest[0], out var parsed) ? parsed : 15;
    if (!File.Exists(logPath)) { P("(no interaction log yet)"); return; }
    foreach (var line in File.ReadLines(logPath).TakeLast(Math.Max(1, n)))
        try { var e = JsonSerializer.Deserialize<InteractionLogEntry>(line); if (e is not null) P($"  {e.Timestamp:u}  {e.Command,-7} {e.Cue ?? ""}{(e.Output is null ? "" : $"  => {e.Output}")}"); }
        catch { /* skip */ }
    return;
}

if (cmd is "serve")
{
    P($"ClaudeMemory daemon — CONTINUOUS MASTERY trainer (background). Ctrl+C to stop & save.");
    P($"  index   : {indexPath}");
    P($"  queue   : {queuePath}");
    P($"  state   : {stateDir}");
    P($"  every   : {interval}s    backend: {(useGpu ? "GPU" : "CPU")}    reps: {reps}");
    if (!File.Exists(indexPath)) P($"  WARNING : index source missing — only the queue + repo-derived facts will train.");
    if (File.Exists(indexPath)) File.WriteAllText(indexPointer, indexPath); // remember the index for bare recall
    File.WriteAllText(pidPath, System.Diagnostics.Process.GetCurrentProcess().Id.ToString()); // liveness for `watch`

    var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
    void Stamp(string s) { P($"  [{DateTimeOffset.Now:HH:mm:ss}] {s}"); LogInteraction("serve", null, null, s); }

    async Task<GenesisEvalAppRuntime> RebuildFresh()
    {
        var rt = new GenesisEvalAppRuntime(MakeConfig(resume: false)); // FRESH so stale associations drop
        RegisterOps(rt);
        var lines = WriteTrainFile(MaxAnswersPerCue);
        await rt.SaveAsync(checkpoint);
        try { File.WriteAllText(sigPath, TrainsetSig().ToString()); } catch { /* best effort */ }
        Stamp($"reseeded fresh: {lines.Count} examples queued — training in batches (hidden {rt.HiddenSize}, {(useGpu ? "GPU" : "CPU")})");
        return rt;
    }

    // Persistent NN: ALWAYS resume the saved model + space when a checkpoint exists — keep the long-lived GRU
    // and whatever the space has learned. A training-set change (memory/code) only REFRESHES the sampled set
    // (below); it never wipes the NN. Only a missing checkpoint triggers a fresh build. (`reset-platonic` is
    // the explicit way to drop the space.)
    GenesisEvalAppRuntime server;
    if (File.Exists(checkpoint))
    {
        server = new GenesisEvalAppRuntime(MakeConfig(resume: true));
        RegisterOps(server);
        Stamp("resumed saved NN + platonic space");
    }
    else
    {
        server = await RebuildFresh();
    }
    try { File.WriteAllText(sigPath, TrainsetSig().ToString()); } catch { /* best effort */ }
    var lastSig = TrainsetSig();
    var probes = GenerateProbes();
    var cycle = 0;
    var stable = 0;
    const int StableCycles = 3;     // legacy idle latch (now set when ALL buckets are mastered)
    const int EvalEvery = 5;         // run the 82-probe held-out curve every N cycles (observability)
    const int RefreshEvery = 100000; // effectively never: the periodic rebuild only existed to re-permute lists
                                     // (now removed), and it was RESETTING the focus mid drive-to-depth every 10
                                     // cycles so nothing could master. Real memory/code changes still refresh via
                                     // the signature check at the top of the loop (which also re-scans code).
    var lastRefreshCycle = 0;
    const double SaveEverySec = 45;  // persist at most this often — model stays RAM-resident between saves

    // BUCKETED MASTERY CURRICULUM, ported from the autonomous orchestrator's regimen: partition the facts into
    // "sets to master"; focus ONE set until it is Done; pick the next focus by a FEEDBACK-DRIVEN priority signal;
    // anneal the LR as the set nears its bar; DRIVE EACH SET TO DEPTH (its own difficulty = answer-list length,
    // grown only after it masters the current depth); rehearse Done sets to resist forgetting. Idle when all Done.
    const double BucketTarget = 0.85; // per-bucket accuracy bar (fuzzy set-membership over its own cues)
    const int BucketStable = 2;       // consecutive grades at/above the bar → mastered at the current difficulty
    const int FocusBoost = 4;         // times the focus set is repeated in an epoch (mastery acceleration)
    const int FocusGiveUp = 15;       // cycles to STAY on one set per difficulty before rotating out (anti-lock)
    const int MaxBucketLines = 5;     // SPLIT bigger groups into sub-buckets this small so each is masterable
    const int BroadRehearse = 56;     // random lines from the WHOLE CORPUS mixed into every epoch. CRITICAL: training
                                      // only the focus + the few Done sets CATASTROPHICALLY FORGOT everything else —
                                      // the decoder collapsed to a single answer for ALL queries. Broad replay keeps
                                      // every association alive (a flat associative INDEX needs all of them at once).
    const int MaxGradeCues = 8;       // grading cap per set (sets are <= MaxBucketLines, so this rarely binds)
    const int StartDifficulty = 1;    // EASIEST FIRST: every cue trains exactly ONE answer, so early broad
                                      // training reinforces "one answer completes the response" (curbs the
                                      // decoder's over-generation at the source). A GLOBAL difficulty then ramps
                                      // list-lengths up (RampBar/RampStable) only once the index is accurate AND
                                      // concise at the current depth. Distinct from the abandoned PER-BUCKET
                                      // drive-to-depth (which starved coverage); the ramp is global so every cue
                                      // is still present at every level — broad coverage is preserved throughout.
    const double RampBar = 0.85;      // held-out QUALITY (answer-present minus over-generation) needed to ramp
    const int RampStable = 3;         // consecutive eval cycles at/above the bar before list lengths grow
    const int DifficultyStep = 1;     // grow the global list-length cap by this each ramp (1→2→…→MaxAnswersPerCue)
    const double ExplorationBase = 1.0; // priority weight for under-trained sets (decays with attempts)
    const bool RequirePlatonic = true;  // CAPABILITY-MASTERY: a correct answer via the NEURAL FALLBACK does NOT
                                        // count — the mission is to RETRIEVE via the platonic substrate, not
                                        // memorize. Punishes fallback so a set only masters once it truly routes.
    var baseLr = server.LearningRate; // the un-annealed step; anneal multiplies this, restored on stop
    List<Bucket>? buckets = null;
    var allCorpusLines = new List<string>(); // every association (cue => member) — the broad-training corpus
    const int ChunkSize = 96;         // associations trained per cycle (a slice of the shuffled corpus sweep)
    var sweep = new List<string>();   // current shuffled pass over the whole corpus
    var sweepPos = 0;                 // position within the current sweep
    var epochNum = 0;                 // completed full passes (epochs) over the corpus
    var globalDifficulty = StartDifficulty; // GLOBAL list-length cap, ramped up as the index masters each depth
    var rampStreak = 0;               // consecutive at-bar eval cycles accumulated toward the next ramp
    Bucket? focus = null;             // (legacy bucket state kept only for building allCorpusLines / observability)
    // Per-set progress carried across re-bucketing when the set's lines are unchanged (keyed by name+hash).
    var bucketState = new Dictionary<string, (int Difficulty, double Acc, int Streak, int Attempts, bool Done, string Hash)>(StringComparer.OrdinalIgnoreCase);

    var saveTimer = System.Diagnostics.Stopwatch.StartNew();
    var lastAcc = 0.0; var lastHeld = 0.0; var lastPurity = 0.0; var lastConf = 0.0; // last held-out eval, shown every line

    // ── Curriculum helpers ───────────────────────────────────────────────────────────────────────────────
    // The "set" a cue belongs to: contains/calls group by their entity (a Type's members, a method's callees);
    // everything else (find, bridges) groups by the target it resolves to — so each bucket is a coherent set.
    static string BucketKeyOfCue(string cue, string firstAnswer)
    {
        var sp = cue.IndexOf(' ');
        var verb = sp > 0 ? cue[..sp] : cue;
        var operand = sp > 0 ? cue[(sp + 1)..].Trim() : "";
        if ((verb is "contains" or "calls") && operand.Length > 0) return verb + ":" + operand;
        return firstAnswer.Length > 0 ? "to:" + firstAnswer : "cue:" + cue;
    }
    static string LinesHash(IEnumerable<string> lines) => string.Join('\n', lines).GetHashCode().ToString();
    static double AnnealFactor(double acc, double target) // orchestrator anti-oscillation curve, bar-relative
        => acc < target - 0.03 ? 1.00 : acc < target + 0.02 ? 0.30 : 0.10;
    static string Trunc(string s, int n) => s.Length <= n ? s : s[..n];
    // Feedback-driven focus priority (port of the planner's signal): weak sets (far below the bar) + under-trained
    // sets (few attempts) + sets with depth still to climb rank highest. Higher = more deserving of focus.
    double Priority(Bucket b) =>
        (BucketTarget - b.Acc)                          // weakness: how far below the bar
        + ExplorationBase / (1.0 + b.Attempts)          // exploration: prefer under-trained sets
        + 0.10 * (b.MaxDifficulty - b.Difficulty);      // depth remaining (small nudge)
    // The focus set's epoch at ITS difficulty: each cue capped to b.Difficulty answers (+ permutations).
    List<string> EpochLinesFor(Bucket b)
    {
        var lines = new List<string>();
        foreach (var (cue, answers) in b.Cues) lines.AddRange(CueLines(cue, answers, b.Difficulty));
        return lines;
    }
    List<Bucket> BuildBuckets(Dictionary<string, List<string>> cueAnswers)
    {
        var groups = new Dictionary<string, List<(string Cue, List<string> Answers)>>(StringComparer.OrdinalIgnoreCase);
        foreach (var (cue, answers) in cueAnswers)
        {
            if (answers.Count == 0) continue;
            var key = BucketKeyOfCue(cue, answers[0]);
            if (!groups.TryGetValue(key, out var g)) groups[key] = g = new List<(string, List<string>)>();
            g.Add((cue, answers));
        }
        // SPLIT each set into masterable sub-buckets of <= MaxBucketLines cues ("key#1", "key#2", ...).
        var list = new List<Bucket>();
        foreach (var (key, g) in groups.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var parts = (g.Count + MaxBucketLines - 1) / MaxBucketLines;
            for (var p = 0; p < parts; p++)
            {
                var cues = g.Skip(p * MaxBucketLines).Take(MaxBucketLines).ToList();
                var name = parts == 1 ? key : $"{key}#{p + 1}";
                var maxDiff = Math.Clamp(cues.Max(c => c.Answers.Count), 1, MaxAnswersPerCue);
                var b = new Bucket
                {
                    Name = name, Cues = cues, MaxDifficulty = maxDiff,
                    Difficulty = Math.Min(globalDifficulty, maxDiff),
                    Hash = LinesHash(cues.Select(c => c.Cue + "=>" + string.Join(",", c.Answers)))
                };
                // Carry this set's observability progress forward when its cues/answers are unchanged — but NOT
                // Difficulty: depth is governed GLOBALLY by the ramp, re-clamped here so a re-bucket can't desync it.
                if (bucketState.TryGetValue(name, out var st) && st.Hash == b.Hash)
                    (b.Acc, b.Streak, b.Attempts, b.Done) = (st.Acc, st.Streak, st.Attempts, st.Done);
                b.Difficulty = Math.Min(globalDifficulty, maxDiff);
                list.Add(b);
            }
        }
        return list;
    }
    // Grade a set at ITS difficulty (fuzzy set-membership: reward right members, penalize wrong). Deterministic
    // (same cues each time = stable signal). Also returns up to 5 sample display lines (no extra predicts).
    async Task<(double Acc, double Route, List<string> Samples)> GradeBucketAsync(Bucket b)
    {
        var cues = b.Cues.Count > MaxGradeCues ? b.Cues.Take(MaxGradeCues).ToList() : b.Cues;
        if (cues.Count == 0) return (1.0, 1.0, new List<string>());
        double total = 0; var n = 0; var routed = 0;
        var samples = new List<string>();
        foreach (var (cue, answers) in cues)
        {
            // Grade against the FULL valid list: a returned member is CORRECT if it's anywhere in the full set, so
            // listing MORE valid members than the depth asked for never counts against the model — only members
            // that are NOT in the full list are wrong. So "gives 4 answers, the right one is among them" passes.
            var full = answers.Select(x => x.Trim()).Where(x => x.Length > 0).ToList();
            var fullSet = full.Select(Canon).Where(x => x.Length > 0).ToHashSet();
            if (fullSet.Count == 0) continue;
            // Single-answer world, RECALL-FOCUSED: a cue passes when its output contains ANY valid member of the
            // full set (the relation is one-to-many; retrieving one is correct). We DON'T zero it for extra tokens
            // — a fresh decoder over-generates, and penalizing that hid genuine retrievals (acc 0 while rt 100).
            var res = (await server.PredictAsync(cue, maxTokens: 8)).Result; // short: we want the retrieved member
            var raw = res?.Output?.Trim() ?? "";
            var viaPlatonic = res is not null && !res.UsedNeuralFallback; // routed through the substrate, not memorized
            if (viaPlatonic) routed++;
            var got = raw.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(Canon).Where(x => x.Length > 0).ToHashSet();
            var valid = got.Count(g => fullSet.Contains(g)); // members of the output that ARE in the full set
            var wrong = got.Count - valid;
            var hit = valid >= 1 ? 1.0 : 0.0;
            // PUNISH NEURAL FALLBACK: a correct retrieval that didn't route through the substrate scores 0, so a
            // set masters only when it genuinely routes (capability-mastery, like the orchestrator).
            total += (RequirePlatonic && !viaPlatonic) ? 0.0 : hit;
            n++;
            if (samples.Count < 5)
            {
                samples.Add($"      [{(hit > 0 ? "hit" : "miss")}, {wrong} extra]{(viaPlatonic ? "" : " [neural-fallback]")}  {cue} -> {raw}");
                samples.Add($"            valid (any of): {string.Join(", ", full.Take(8))}{(full.Count > 8 ? ", ..." : "")}");
            }
        }
        return (n == 0 ? 1.0 : total / n, n == 0 ? 1.0 : routed / (double)n, samples);
    }

    Stamp($"broad-training index: {probes.Count} probes ({probes.Count(p => p.HeldOut)} held-out); sweep the whole corpus in {ChunkSize}-edge chunks, reshuffle each epoch ({(useGpu ? "GPU" : "CPU")})");

    while (!cts.IsCancellationRequested)
    {
        // (1) Training set changed (memory/code) → REFRESH the sampled set + probes; KEEP the NN + space (no
        // wipe — persistent NN). New facts get trained in subsequent random batches; stale ones fade unsampled.
        var sigNow = TrainsetSig();
        if (sigNow != lastSig)
        {
            lastSig = sigNow;
            try { File.WriteAllText(sigPath, sigNow.ToString()); } catch { /* best effort */ }
            probes = GenerateProbes();
            codeFactsStamp = -1; // force a code re-scan (the periodic refresh used to do this)
            buckets = null;
            Stamp("training set changed (memory/code) — refreshed; NN + space kept.");
        }

        // (2) Drain the fact queue (ad-hoc associations) — new material resets mastery so it gets learned.
        if (File.Exists(queuePath))
        {
            var lines = File.ReadAllLines(queuePath).Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
            if (lines.Count > 0)
            {
                var trained = 0;
                foreach (var l in lines)
                    try { var f = JsonSerializer.Deserialize<Fact>(l); if (f is not null) { await TrainFactAsync(server, f, 1); trained++; } }
                    catch { /* skip malformed */ }
                await server.SaveAsync(checkpoint);
                File.WriteAllText(queuePath, string.Empty);
                stable = 0;
                Stamp($"trained {trained} queued fact(s)");
            }
        }

        // (3) Build / refresh the BUCKETED curriculum. Periodically re-scan memory/code + re-permute (resetting
        // codeFactsStamp forces GenerateCodeFacts to re-run). Re-bucketing carries mastery forward for unchanged
        // sets and resets changed ones.
        if (cycle - lastRefreshCycle >= RefreshEvery)
        {
            lastRefreshCycle = cycle;
            codeFactsStamp = -1;
            buckets = null;
            Stamp("refreshed training set (re-scan + new permutations)");
        }
        if (buckets is null)
        {
            var cueAnswers = BuildCueAnswers();
            WriteTrainFile(MaxAnswersPerCue); // refresh the reference artifact (trainset.txt) at full length
            buckets = BuildBuckets(cueAnswers);
            allCorpusLines = buckets.SelectMany(EpochLinesFor).ToList(); // the whole corpus, broad-trained in sweeps
            sweep = new List<string>(); sweepPos = 0; // force a fresh shuffled sweep over the refreshed corpus
            focus = null;
            var cueCount = buckets.Sum(b => b.Cues.Count);
            Stamp($"corpus: {cueCount} cues, {allCorpusLines.Count} edges; BROAD training, {ChunkSize}/cycle (full epoch every ~{Math.Max(1, allCorpusLines.Count / ChunkSize)} cycles)");
        }
        if (buckets.Count == 0)
        {
            try { await Task.Delay(TimeSpan.FromSeconds(interval), cts.Token); } catch (TaskCanceledException) { break; }
            continue;
        }

        // (4) BROAD SWEEP — the EMPIRICALLY-PROVEN regime (Tests/AssociativeIndexLearningTests: clean broad
        // training learns the index to 100% retrieval + 100% platonic routing, multi-token keys, scale 400, no
        // forgetting). Train EVERY association: a ChunkSize slice of the SHUFFLED corpus per cycle, completing a
        // full epoch each pass, reshuffling between passes. NO per-bucket focus / mastery-idle — that curriculum
        // starved coverage and collapsed the decoder to one answer for every query. The ONLY curriculum knob is
        // the GLOBAL difficulty (list-length per cue), which ramps 1→MaxAnswersPerCue as held-out quality holds —
        // every cue is still swept at every depth, so coverage is never starved. The break throttle paces it.
        if (sweepPos >= sweep.Count)
        {
            sweep = allCorpusLines.OrderBy(_ => Random.Shared.Next()).ToList(); // fresh shuffled epoch
            sweepPos = 0;
            epochNum++;
        }
        var chunk = sweep.Skip(sweepPos).Take(ChunkSize).ToList();
        sweepPos += chunk.Count;
        if (chunk.Count == 0)
        {
            try { await Task.Delay(TimeSpan.FromSeconds(interval), cts.Token); } catch (TaskCanceledException) { break; }
            continue;
        }
        File.WriteAllLines(chunkFilePath, chunk);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        double trainLoss;
        try
        {
            var report = await server.TrainAsync(chunkFilePath, epochs: 1);
            trainLoss = report.AverageLoss.TotalLoss;
        }
        catch (Exception ex)
        {
            Stamp($"train error: {ex.Message}");
            try { await Task.Delay(TimeSpan.FromSeconds(interval), cts.Token); } catch (TaskCanceledException) { break; }
            continue;
        }
        sw.Stop();
        cycle++;

        // Throttled save: keep the model RAM-resident, persist at most every SaveEverySec. Clean-stop always saves.
        if (saveTimer.Elapsed.TotalSeconds >= SaveEverySec) { await server.SaveAsync(checkpoint); saveTimer.Restart(); }

        // Held-out curve (observability) every EvalEvery cycles; carried forward on each report line.
        if (cycle % EvalEvery == 0)
        {
            (lastAcc, lastHeld, lastPurity, lastConf) = await EvaluateAsync(server, probes);
            File.AppendAllText(metricsPath, JsonSerializer.Serialize(
                new MetricsEntry(DateTimeOffset.UtcNow, cycle, chunk.Count, trainLoss, lastAcc, lastHeld, lastPurity, lastConf)) + Environment.NewLine);

            // GLOBAL DIFFICULTY RAMP: grow list-lengths only once held-out QUALITY (accurate AND concise — the
            // over-generation-penalized metric) holds above the bar for RampStable cycles. Broad coverage is
            // preserved (every cue still trains at the new depth); rebuild the corpus + restart the sweep.
            if (globalDifficulty < MaxAnswersPerCue)
            {
                if (lastHeld >= RampBar && ++rampStreak >= RampStable)
                {
                    globalDifficulty = Math.Min(MaxAnswersPerCue, globalDifficulty + DifficultyStep);
                    rampStreak = 0; buckets = null; // force rebuild of allCorpusLines at the new depth + fresh sweep
                    Stamp($"RAMP → difficulty {globalDifficulty}/{MaxAnswersPerCue} answers/cue (held-out quality {lastHeld:P0} ≥ {RampBar:P0} stable)");
                }
                else if (lastHeld < RampBar) rampStreak = 0;
            }
        }

        // Full report on EVERY line. '*' = fresh held-out eval. held = held-out retrieval, route = route purity.
        var freshEval = cycle % EvalEvery == 0 ? "*" : " ";
        Stamp($"cycle {cycle,4} | sweep {epochNum,3} {sweepPos,5}/{sweep.Count,-5} | loss {trainLoss,6:F3} | held {lastHeld,4:P0} acc {lastAcc,4:P0} route {lastPurity,4:P0}{freshEval}| {sw.Elapsed.TotalSeconds,4:F1}s");

        // On eval cycles, show 5 random associations + the model's guess (generous token budget for multi-token keys).
        if (cycle % EvalEvery == 0 && allCorpusLines.Count > 0)
        {
            P("      -- sample: cue -> guess   (want ~ key) --");
            for (var s = 0; s < 5; s++)
            {
                var line = allCorpusLines[Random.Shared.Next(allCorpusLines.Count)];
                var arrow = line.IndexOf(" => ", StringComparison.Ordinal);
                if (arrow <= 0) continue;
                var cue = line[..arrow].Trim();
                var key = line[(arrow + 4)..].Trim();
                var raw = (await server.PredictAsync(cue, maxTokens: 16)).Result?.Output?.Trim() ?? "";
                var hit = Canon(key).Length > 0 && Canon(raw).Contains(Canon(key));
                P($"      [{(hit ? "hit " : "miss")}]  {cue} -> {raw}   (want ~ {key})");
            }
        }

        // Rest = batch_time × (break%/100), read live so `break <pct>` retunes the daemon with no restart.
        var breakPct = ReadBreakPercent();
        var coolDown = TimeSpan.FromMilliseconds(Math.Min(sw.Elapsed.TotalMilliseconds * (breakPct / 100.0), 600000));
        if (coolDown > TimeSpan.Zero)
        {
            try { await Task.Delay(coolDown, cts.Token); }
            catch (TaskCanceledException) { break; }
        }
    }

    await server.SaveAsync(checkpoint);
    try { File.Delete(pidPath); } catch { /* best effort */ }
    Stamp("stopped & saved.");
    return;
}

// ── Commands that need one warm runtime ───────────────────────────────────────────────────────────────
var runtime = new GenesisEvalAppRuntime(MakeConfig(resume: cmd is not "rebuild"));
RegisterOps(runtime);

// recall = the SAME GRU-routed inference the autonomous trainer + tests use: Generate(query) → the GRU
// PredictRoutes the decision, the trained platonic routes (relation-first / concept-chain) retrieve the
// memory key. No hand-coded traversal or query DSL — the GRU figures it out. If it misses, that's a TRAINING
// gap (train the GRU to route memory queries), not a query-syntax gap.
async Task RecallAsync(string query)
{
    // Pass the query through VERBATIM — no hard-coded syntax. The caller writes the query in whatever form they
    // want (e.g. `find license setup` to use the trained find syntax, or a bare/natural query); the GRU routes
    // exactly what it's given. Generate does all the routing/retrieval.
    var pred = await runtime.PredictAsync(query.Trim(), maxTokens: 16);
    var r = pred.Result!;
    P($"recall  \"{query}\"");
    P($"  -> {r.Output}");
    P($"  [route {r.DecisionPath}   confidence {r.PlatonicConfidence:F2}   neural-fallback={r.UsedNeuralFallback}]");
    var oc = Canon(r.Output);
    // Resolve a MEMORY key: training strips hyphens (nova-claude-memory -> novaclaudememory), so Canon-match the
    // single-token output back to its file (Canon also strips hyphens, so the original filename matches).
    var memDir = File.Exists(indexPath) ? Path.GetDirectoryName(indexPath) : null;
    if (oc.Length > 0 && memDir is not null && Directory.Exists(memDir))
        foreach (var f in Directory.EnumerateFiles(memDir, "*.md"))
        {
            var nm = Path.GetFileNameWithoutExtension(f);
            if (Canon(nm) == oc) { P($"  memory: {nm}.md"); break; }
        }
    // If the retrieved key is a code symbol, resolve it to file:line from the manifest (the source of truth).
    if (File.Exists(manifestPath) && oc.Length > 0)
        foreach (var line in File.ReadLines(manifestPath))
        {
            var tab = line.IndexOf('\t');
            if (tab > 0 && Canon(line[..tab]) == oc) { P($"  code: {line[(tab + 1)..]}"); break; }
        }
    LogInteraction("recall", query, null, r.Output);
}

switch (cmd)
{
    case "rebuild":
    case "sync":
        if (!File.Exists(indexPath)) { P($"index source not found: {indexPath} (pass --index or set $CLAUDE_MEMORY_FILE)."); break; }
        File.WriteAllText(indexPointer, indexPath); // remember the index for bare recall
        var facts = GenerateTruth();
        if (facts.Count == 0) { P("no associations derived from the index."); break; }
        foreach (var f in facts) await TrainFactAsync(runtime, f);
        await runtime.SaveAsync(checkpoint);
        LogInteraction(cmd, indexPath, null, $"{facts.Count} associations");
        P($"{(cmd == "rebuild" ? "rebuilt (fresh)" : "synced")} index from {Path.GetFileName(indexPath)}: {facts.Count} associations ({reps} reps each) — saved.");
        break;

    case "teach":
        if (rest.Count < 2) { P("usage: teach \"<cue>\" \"<response>\""); break; }
        await TrainFactAsync(runtime, new Fact(rest[0], rest[1], false));
        await runtime.SaveAsync(checkpoint);
        LogInteraction("teach", rest[0], rest[1], null);
        P($"taught  \"{rest[0]}\" -> \"{rest[1]}\" (saved) [prefer editing MEMORY.md + rebuild, or enqueue]");
        break;

    case "relate":
        if (rest.Count < 2) { P("usage: relate \"<a>\" \"<b>\""); break; }
        await TrainFactAsync(runtime, new Fact(rest[0], rest[1], true));
        await runtime.SaveAsync(checkpoint);
        LogInteraction("relate", rest[0], rest[1], null);
        P($"related \"{rest[0]}\" <-> \"{rest[1]}\" (saved)");
        break;

    case "recall":
        if (rest.Count < 1) { P("usage: recall \"<query>\""); break; }
        EnsureDaemonRunning(); // querying the memory ensures the trainer is up — no separate launch step
        await RecallAsync(string.Join(' ', rest));
        break;

    case "stats":
        var d = runtime.Diagnose();
        P($"ClaudeMemory @ {stateDir}");
        P($"  index      : {indexPath}  ({(File.Exists(indexPath) ? "present" : "missing")})");
        P($"  checkpoint : {(d.CheckpointExists ? $"saved {d.CheckpointWriteUtc:u}" : "(none yet)")}");
        P($"  concepts   : {d.NodeCount:N0}    relations: {d.RelationCount:N0}    vocab: {d.VocabularySize}");
        if (d.TopRelations.Length > 0) { P("  strongest associations (keyword <-> memory key):"); foreach (var rel in d.TopRelations.Take(14)) P($"     {rel.Left,-22} <-> {rel.Right,-30} obs {rel.ObservationCount}"); }
        break;

    default:
        P($"unknown command '{cmd}'. Try: serve | enqueue | rebuild | recall | teach | relate | stats | log | help");
        break;
}

internal sealed record Fact(string Cue, string Response, bool Relate);

// A curriculum "set to master": a coherent group of cues (a Type's members, a method's callees, every keyword
// pointing at one memory) trained to mastery with LR annealing + rehearsal, like an orchestrator lesson. Each
// bucket tracks its OWN difficulty (answer-list length) and DRIVES IT TO DEPTH before it is Done.
internal sealed class Bucket
{
    public string Name = "";
    public List<(string Cue, List<string> Answers)> Cues = new(); // full answer sets = source of truth
    public int Difficulty;       // current answer-list cap for this set (its private drive-to-depth level)
    public int MaxDifficulty;    // deepest meaningful cap (= longest answer list, clamped); 1 = single-answer set
    public double Acc;           // last graded accuracy at the CURRENT difficulty (platonic-routed only)
    public double Route;         // last fraction of this set's cues answered via the platonic path (not neural)
    public int Streak;           // consecutive grades at/above the bar at the current difficulty
    public int Attempts;         // focus cycles spent at the current difficulty (give-up + exploration signal)
    public double LastLoss = double.NaN; // last focus-train token loss (feedback signal)
    public bool Done;            // mastered at MaxDifficulty → drops out of focus, stays in rehearsal
    public string Hash = "";     // cue/answer fingerprint; a change resets this set's progress
}

// A measurement probe: a query and the memory key it SHOULD recall. HeldOut probes use the find-syntax on an
// operand never trained in that syntax — a correct hit proves operand generalization, the mastery target.
internal sealed record MemoryProbe(string Query, string ExpectedKey, bool HeldOut);

// One row of the continuous-mastery learning curve, appended per training cycle.
internal sealed record MetricsEntry(
    DateTimeOffset Timestamp,
    int Cycle,
    int Trained,
    double TrainLoss,
    double Accuracy,
    double HeldOutAccuracy,
    double RoutePurity,
    double Confidence);

internal sealed record InteractionLogEntry(
    DateTimeOffset Timestamp,
    string Command,
    string? Cue,
    string? Response,
    string? Output);
