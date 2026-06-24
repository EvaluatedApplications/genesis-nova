using System.Globalization;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Runtime;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using RaceBench;
using TorchSharp;
using static TorchSharp.torch;

// ════════════════════════════════════════════════════════════════════════════════════════════════════
//  GENESIS-NOVA  vs  TRANSFORMER  — full curriculum (every creator), equal budget.
//  Same tokenizer, same pooled examples, same epochs, matched parameter count. Each model is trained in ITS OWN
//  real regime: nova correctness-gated (train only what's wrong — the production gym's TrainOnFailureOnly, which
//  stops flat re-training from eroding its relations), the transformer flat SGD (it needs the repetition).
// ════════════════════════════════════════════════════════════════════════════════════════════════════

// Redirect libtorch's native stderr to a log BEFORE any torch call loads the native library. libtorch emits
// cosmetic notices on stderr (e.g. the "non-leaf Tensor .grad" warning from deep inside TorchSharp's SGD on
// the nova side) that would otherwise spam the race window. Same approach the ClaudeMemory daemon uses to keep
// its window to just the learning curve. The warnings still land in race-stderr.log for debugging.
NativeStderr.RedirectToFile(Path.Combine(AppContext.BaseDirectory, "race-stderr.log"));

var logPath = Path.Combine(AppContext.BaseDirectory, "race-log.txt");
using var log = new StreamWriter(logPath) { AutoFlush = true };
void P(string s) { Console.WriteLine(s); log.WriteLine(s); }
void Rule() => P(new string('═', 86));

var dev = cuda.is_available() ? CUDA : CPU;
const int HIDDEN = 256, SEED = 7;  // SMALL: nova keeps all faces (poly/log/char/word); the
                                                // question is whether a transformer this size has the capacity
var rng = new Random(SEED);

// ── Full curriculum: pool every creator, split disjoint train / held-out per creator ─────────────────
// Plus a LEARNABLE association-recall task (race-local): arithmetic/number-word are computed/codec (nothing to
// learn), so they can't show a learning curve — this one CAN ONLY be learned, and its held-out is inferrable
// (seen entities, new phrasings), so we measure LEARNING, not just built-in compute.
var creators = ExampleCreatorRegistry.All.Append(new AssociationRecallCreator()).ToList();
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

// ── Build nova — the SAME brain the desktop app ships ───────────────────────────────────────────────────
// GenesisRuntimeState is the canonical wiring (substrate selection + NovaConfig.ApplyTo + SetInferencePolicy);
// .WithProductionMechanisms() turns on exactly what the app turns on (dialectical core + conscious field +
// keep-core + the meaning-space self + function gradient). So the race tracks the app automatically — change a
// mechanism in GenesisNovaConfig.WithProductionMechanisms() and the race follows. We choose only the small race
// dims / seed / LR here (the equal-small-params contest).
var cfg = new GenesisNovaConfig(HiddenSize: HIDDEN, LearningRate: 0.05, Seed: SEED).WithProductionMechanisms();
var nova = new GenesisRuntimeState(cfg);
var model = nova.Model;
var novaTrainer = nova.Trainer;
var inference = nova.Inference;
novaTrainer.TrainStep(new GenesisExample("0 + 0", "0")); // force lazy params to initialise before counting
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
P($"  nova        : {novaParams,10:N0} params   ~{NovaMB,5:F1} MB   (conscious-field cognition over the dialectical substrate, SGD)");
P($"  transformer : {xf.ParameterCount,10:N0} params   ~{XfMB,5:F1} MB   (d={best.d} L={best.L} h={best.h}, Adam)");
P($"  budget      : EQUAL PARAMETERS, both SMALL — nova's structure needs little capacity; can a transformer this size find it?");
Rule();
P("  GATED RACE — both models train in LOCKSTEP: neither starts epoch N+1 until BOTH have finished epoch N.");
P("  (per-epoch barrier ⇒ EQUAL training work; GPU kernels serialized on one CUDA stream. Compare held-out at matched epochs.)");
P("  >>> PRESS ANY KEY to stop the race and print the final per-creator breakdown. <<<");
P("  ──────────────────────────────────────────────────────────────────────────────────────────────────────────");

double Acc(Func<string, string> gen, List<(string Input, string Output)> data)
{
    var c = 0;
    foreach (var (i, o) in data) if (AnswerEquivalence.Equivalent(gen(i), o)) c++;
    return c / (double)Math.Max(1, data.Count);
}
string NovaGen(string i) => inference.Generate(new GenerationRequest(i, 8)).Output.Trim();

// EPOCH 0 — the UNTRAINED baseline, printed so LEARNING is visible at a glance: skills that are COMPUTED
// (arithmetic) or CODEC already score here with ZERO training, while skills that must be LEARNED (association-
// recall) start near zero. A line that's already high at epoch 0 was never learned; a line that climbs from
// epoch 0 is the model actually learning.
P($"  {"NOVA",-11} ep   0   train {Acc(NovaGen, evalTrain),5:P0}   held-out {Acc(NovaGen, evalHeld),5:P0}   (untrained baseline)");
P($"  {"TRANSFORMER",-11} ep   0   train {Acc(xf.Generate, evalTrain),5:P0}   held-out {Acc(xf.Generate, evalHeld),5:P0}   (untrained baseline)");
P("  ──────────────────────────────────────────────────────────────────────────────────────────────────────────");

// Two locks: `gpu` serializes ALL GPU/model work (one CUDA stream — concurrent kernel launches are unsafe),
// `outLock` serializes console+log output. Each model runs its OWN epoch loop in its OWN task and reports the
// instant it finishes an epoch, so the faster model gets through the GPU lock more often and pulls ahead.
var gpu = new object();
var outLock = new object();
double lastNovaHeld = 0, lastXfHeld = 0;
const int CHUNK = 32; // GPU-lock granularity: hand the GPU back every CHUNK examples so the two tasks interleave

// Per-epoch BARRIER (2 participants): neither model begins epoch N+1 until BOTH have finished epoch N. This
// makes the training budgets identical (equal epochs) — the comparison is held-out quality at matched work,
// not a throughput race. Both tasks always reach the barrier once per epoch, so on cancel both release and
// then exit at the top of the next iteration (no deadlock).
var epochBarrier = new System.Threading.Barrier(2);

void Post(string who, ConsoleColor color, int ep, double tr, double held, double otherHeld, double epochSecs, string extra = "")
{
    var lead = held > otherHeld + 1e-9 ? "▲ ahead" : held < otherHeld - 1e-9 ? "▼ behind" : "= even";
    var tail = extra.Length > 0 ? "  " + extra : "";
    var line = $"  {who,-11} ep {ep,3}   train {tr,5:P0}   held-out {held,5:P0}   {lead,-8} (epoch {epochSecs,5:F1}s){tail}";
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
        var epSw = System.Diagnostics.Stopwatch.StartNew(); // time THIS epoch's work (train + eval), not wall clock
        var order = train.OrderBy(_ => rngN.Next()).ToList();
        // CORRECTNESS-GATED, exactly like the production gym (TrainOnFailureOnly): predict each example and train ONLY
        // the currently-WRONG ones. This is nova's REAL training regime — flat re-training of already-correct examples
        // ERODES the platonic relations (framing-word/entity hubs accumulate and drown the signal); the gate stops
        // touching a skill once mastered, so it holds instead of rotting. The transformer keeps its flat SGD regime
        // (it NEEDS repetition) — each model is trained the way it actually is.
        var trained = 0;
        for (var b = 0; b < order.Count; b += CHUNK)
            lock (gpu)
            {
                foreach (var (i, o) in order.Skip(b).Take(CHUNK))
                    if (!AnswerEquivalence.Equivalent(NovaGen(i), o)) { novaTrainer.TrainStep(new GenesisExample(i, o)); trained++; }
            }
        double nt, nh;
        lock (gpu) { nt = Acc(NovaGen, evalTrain); nh = Acc(NovaGen, evalHeld); }
        lastNovaHeld = nh;
        Post("NOVA", ConsoleColor.Cyan, ep, nt, nh, lastXfHeld, epSw.Elapsed.TotalSeconds, $"trained {trained,3}/{order.Count}");
        try { epochBarrier.SignalAndWait(cts.Token); } catch (OperationCanceledException) { break; }
    }
});

var xfTask = Task.Run(() =>
{
    var rngX = new Random(SEED + 22);
    for (var ep = 1; ep <= SAFETYCAP && !cts.IsCancellationRequested; ep++)
    {
        var epSw = System.Diagnostics.Stopwatch.StartNew(); // time THIS epoch's work (train + eval), not wall clock
        var order = train.OrderBy(_ => rngX.Next()).ToList();
        for (var b = 0; b < order.Count; b += CHUNK)
            lock (gpu) { xf.TrainBatch(order.Skip(b).Take(CHUNK).ToList()); }
        double tt, th;
        lock (gpu) { tt = Acc(xf.Generate, evalTrain); th = Acc(xf.Generate, evalHeld); }
        lastXfHeld = th;
        Post("TRANSFORMER", ConsoleColor.Yellow, ep, tt, th, lastNovaHeld, epSw.Elapsed.TotalSeconds);
        try { epochBarrier.SignalAndWait(cts.Token); } catch (OperationCanceledException) { break; }
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
P($"  note             : each model trained in its OWN regime — nova correctness-gated (the production gym's");
P($"                     TrainOnFailureOnly), the transformer flat SGD; equal epochs, equal params.");
Rule();
Console.WriteLine($"\n(log written to {logPath})");

/// <summary>
/// Redirects the process's CRT stderr (file descriptor 2) to a file. libtorch's native warnings are written
/// via the C runtime's stderr (fileno == 2), which bypasses .NET's Console.SetError AND the Win32 SetStdHandle.
/// Because the universal CRT (ucrtbase.dll) is a single shared module, dup2-ing fd 2 here redirects libtorch's
/// stderr too, so its cosmetic notices land in a log instead of spamming the race window.
/// </summary>
internal static class NativeStderr
{
    // _open flags: _O_WRONLY (1) | _O_CREAT (0x100) | _O_TRUNC (0x200); pmode _S_IWRITE (0x80).
    private const int O_WRONLY = 0x1, O_CREAT = 0x100, O_TRUNC = 0x200, S_IWRITE = 0x80, STDERR_FD = 2;

    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, CharSet = System.Runtime.InteropServices.CharSet.Ansi, EntryPoint = "_open")]
    private static extern int CrtOpen(string filename, int oflag, int pmode);

    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, EntryPoint = "_dup2")]
    private static extern int CrtDup2(int fd1, int fd2);

    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, EntryPoint = "_close")]
    private static extern int CrtClose(int fd);

    public static void RedirectToFile(string path)
    {
        try
        {
            var fd = CrtOpen(path, O_WRONLY | O_CREAT | O_TRUNC, S_IWRITE);
            if (fd >= 0)
            {
                CrtDup2(fd, STDERR_FD); // make fd 2 point at the file → libtorch's stderr follows
                CrtClose(fd);
            }
        }
        catch { /* best-effort: if redirect fails, the warnings simply remain on-screen */ }
    }
}
