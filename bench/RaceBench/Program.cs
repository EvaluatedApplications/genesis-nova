using System.Globalization;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using RaceBench;
using TorchSharp;
using static TorchSharp.torch;

// ════════════════════════════════════════════════════════════════════════════════════════════════════
//  GENESIS-NOVA  vs  TRANSFORMER  — full curriculum (every creator), equal budget.
//  Same tokenizer, same pooled examples, same epochs, matched parameter count. Both trained FLAT (nova's
//  mastery-gated regime is NOT used here, to keep the training procedure identical — conservative for nova).
// ════════════════════════════════════════════════════════════════════════════════════════════════════

var logPath = Path.Combine(AppContext.BaseDirectory, "race-log.txt");
using var log = new StreamWriter(logPath) { AutoFlush = true };
void P(string s) { Console.WriteLine(s); log.WriteLine(s); }
void Rule() => P(new string('═', 86));

var dev = cuda.is_available() ? CUDA : CPU;
const int HIDDEN = 256, SEED = 7;  // SMALL: nova keeps all faces (poly/log/char/word); the
                                                // question is whether a transformer this size has the capacity
var rng = new Random(SEED);

// ── Full curriculum: pool every creator, split disjoint train / held-out per creator ─────────────────
var creators = ExampleCreatorRegistry.All;
var train = new List<(string Input, string Output)>();
var heldByCreator = new Dictionary<string, List<(string Input, string Output)>>();
foreach (var c in creators)
{
    var ex = new List<(string, string)>();
    foreach (var diff in new[] { 0, 1 }) ex.AddRange(c.Generate(400, diff, true));
    var uniq = ex.GroupBy(e => e.Item1).Select(g => g.First()).OrderBy(_ => rng.Next()).ToList();
    var nTrain = Math.Min((int)(uniq.Count * 0.65), 160);
    train.AddRange(uniq.Take(nTrain));
    heldByCreator[c.Name] = uniq.Skip(nTrain).Take(140).ToList();
}
var heldAll = heldByCreator.Values.SelectMany(x => x).ToList();
var evalHeld = heldAll.OrderBy(_ => rng.Next()).Take(160).ToList();
var evalTrain = train.OrderBy(_ => rng.Next()).Take(120).ToList();

// ── Shared tokenizer, warmed so vocab is final before either model is built ───────────────────────────
var tok = new WhitespaceGenesisTokenizer();
foreach (var (i, o) in train.Concat(heldAll)) { tok.Encode(i); tok.Encode(o); }
var vocab = tok.VocabularySize;

// ── Build nova ────────────────────────────────────────────────────────────────────────────────────────
var cfg = new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05);
var model = new GenesisNeuralModel(cfg);
var memory = new PlatonicSpaceMemory(faceDimension: cfg.FaceDimension, seed: SEED);
var novaTrainer = new GenesisTrainer(tok, model, memory, cfg);
var inference = new GenesisInferenceEngine(tok, model, memory, null,
    transformAccumulator: novaTrainer.TransformAccumulator, foldPathDiscovery: novaTrainer.FoldPathDiscovery);
novaTrainer.SetInferencePolicy(inference);
novaTrainer.TrainStep(new GenesisExample("0 + 0", "0"));
var novaParams = model.ParameterCount();

// ── Transformer sized to ≈ nova's (small) parameter count — EQUAL PARAMS, both small. The bet: nova's
//    structural priors (homomorphism, relations) need little capacity, so a transformer this size struggles. ──
long XfParams(int d, int L, int ff, int maxLen)
{
    long V = vocab + 1;
    long perBlock = (long)d * 3 * d + 3 * d + (long)d * d + d + (long)d * ff * d + ff * d + (long)ff * d * d + d + 4L * d;
    return V * d + (long)maxLen * d + L * perBlock + 2L * d + (V * d + V);
}
const int FF = 4, MAXLEN = 32;
var configs = new (int d, int L, int h)[] { (96, 2, 4), (112, 2, 4), (128, 2, 4), (96, 3, 4), (128, 3, 4), (160, 2, 8) };
var best = configs.OrderBy(c => Math.Abs(XfParams(c.d, c.L, FF, MAXLEN) - novaParams)).First();
var xf = new TransformerTrainer(tok, vocab, dModel: best.d, heads: best.h, layers: best.L,
    ffMult: FF, maxLen: MAXLEN, maxAnswer: 8, dev, lr: 3e-4);

double NovaMB = novaParams * 8.0 / (1024 * 1024);
double XfMB = xf.ParameterCount * 16.0 / (1024 * 1024);

Rule();
P("  GENESIS-NOVA  vs  TRANSFORMER    —    FULL CURRICULUM (every creator), equal budget");
Rule();
P($"  device      : {dev.type}");
P($"  curriculum  : {string.Join(", ", creators.Select(c => c.Name))}");
P($"  data        : train {train.Count}   held-out {heldAll.Count}   tokenizer shared   (runs until a key is pressed)");
P($"  nova        : {novaParams,10:N0} params   ~{NovaMB,5:F1} MB   (GRU controller + platonic substrate, SGD)");
P($"  transformer : {xf.ParameterCount,10:N0} params   ~{XfMB,5:F1} MB   (d={best.d} L={best.L} h={best.h}, Adam)");
P($"  budget      : EQUAL PARAMETERS, both SMALL — nova's structure needs little capacity; can a transformer this size find it?");
Rule();
P("  ASYNC RACE — each model trains in its own task and posts a line the moment it finishes an epoch.");
P("  (GPU kernels are serialized on the single CUDA stream; the race is per-epoch throughput — faster pulls ahead.)");
P("  >>> PRESS ANY KEY to stop the race and print the final per-creator breakdown. <<<");
P("  ──────────────────────────────────────────────────────────────────────────────────────────────────────────");

double Acc(Func<string, string> gen, List<(string Input, string Output)> data)
{
    var c = 0;
    foreach (var (i, o) in data) if (AnswerEquivalence.Equivalent(gen(i), o)) c++;
    return c / (double)Math.Max(1, data.Count);
}
string NovaGen(string i) => inference.Generate(new GenerationRequest(i, 8)).Output.Trim();

// Two locks: `gpu` serializes ALL GPU/model work (one CUDA stream — concurrent kernel launches are unsafe),
// `outLock` serializes console+log output. Each model runs its OWN epoch loop in its OWN task and reports the
// instant it finishes an epoch, so the faster model gets through the GPU lock more often and pulls ahead.
var gpu = new object();
var outLock = new object();
var swRace = System.Diagnostics.Stopwatch.StartNew();
double lastNovaHeld = 0, lastXfHeld = 0;
const int CHUNK = 32; // GPU-lock granularity: hand the GPU back every CHUNK examples so the two tasks interleave

void Post(string who, ConsoleColor color, int ep, double tr, double held, double otherHeld)
{
    var lead = held > otherHeld + 1e-9 ? "▲ ahead" : held < otherHeld - 1e-9 ? "▼ behind" : "= even";
    var line = $"  {who,-11} ep {ep,3}   train {tr,5:P0}   held-out {held,5:P0}   {lead,-8} (t {swRace.Elapsed.TotalSeconds,6:F1}s)";
    lock (outLock)
    {
        log.WriteLine(line);
        Console.ForegroundColor = color; Console.WriteLine(line); Console.ResetColor();
    }
}

// Run until the user presses a key (interactive console). A safety cap bounds non-interactive/piped runs.
const int SAFETYCAP = 100_000;
var cts = new System.Threading.CancellationTokenSource();
var keyWatcher = Task.Run(() =>
{
    if (Console.IsInputRedirected) return; // no interactive key (piped/redirected) — run to the safety cap
    Console.ReadKey(intercept: true);
    cts.Cancel();
    lock (outLock) { Console.WriteLine(); Console.WriteLine("  ⏹  stop requested — each model finishes its current epoch, then the final breakdown…"); }
});

var novaTask = Task.Run(() =>
{
    var rngN = new Random(SEED + 11);
    for (var ep = 1; ep <= SAFETYCAP && !cts.IsCancellationRequested; ep++)
    {
        var order = train.OrderBy(_ => rngN.Next()).ToList();
        for (var b = 0; b < order.Count; b += CHUNK)
            lock (gpu) { foreach (var (i, o) in order.Skip(b).Take(CHUNK)) novaTrainer.TrainStep(new GenesisExample(i, o)); }
        double nt, nh;
        lock (gpu) { nt = Acc(NovaGen, evalTrain); nh = Acc(NovaGen, evalHeld); }
        lastNovaHeld = nh;
        Post("NOVA", ConsoleColor.Cyan, ep, nt, nh, lastXfHeld);
    }
});

var xfTask = Task.Run(() =>
{
    var rngX = new Random(SEED + 22);
    for (var ep = 1; ep <= SAFETYCAP && !cts.IsCancellationRequested; ep++)
    {
        var order = train.OrderBy(_ => rngX.Next()).ToList();
        for (var b = 0; b < order.Count; b += CHUNK)
            lock (gpu) { xf.TrainBatch(order.Skip(b).Take(CHUNK).ToList()); }
        double tt, th;
        lock (gpu) { tt = Acc(xf.Generate, evalTrain); th = Acc(xf.Generate, evalHeld); }
        lastXfHeld = th;
        Post("TRANSFORMER", ConsoleColor.Yellow, ep, tt, th, lastNovaHeld);
    }
});

await Task.WhenAll(novaTask, xfTask);

Rule();
P("  FINAL held-out accuracy per creator (full held-out sets):");
P("  ───────────────────────────────────────────────────────────");
foreach (var c in creators)
{
    var h = heldByCreator[c.Name];
    if (h.Count == 0) continue;
    var n = Acc(NovaGen, h); var t = Acc(xf.Generate, h);
    P($"    {c.Name,-28}  nova {n,6:P0}   transformer {t,6:P0}   (n={h.Count})");
}
var novaAll = Acc(NovaGen, heldAll); var xfAll = Acc(xf.Generate, heldAll);
Rule();
P($"  OVERALL held-out : nova {novaAll:P1}  vs  transformer {xfAll:P1}");
P($"  footprint        : nova ~{NovaMB:F1} MB   transformer ~{XfMB:F1} MB   (equal params; nova half the VRAM)");
P($"  note             : both trained FLAT until stopped (nova's mastery-gated regime not used here);");
P($"                     a transformer needs far more epochs to fit train — this is the equal-budget result.");
Rule();
Console.WriteLine($"\n(log written to {logPath})");
