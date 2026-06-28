using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using RetentionBench;
using TorchSharp;

// ════════════════════════════════════════════════════════════════════════════════════════════════════
//  RETENTION-AT-SCALE BENCH — does the substrate LEARN and RETAIN a growing stream of facts as it scales?
//  A large procedural fact universe (nonce entity -> category) is trained WAVE by WAVE; we continuously re-probe
//  the OLDEST facts (wave 0) to plot the forgetting curve, and watch the space fill toward its node cap and start
//  evicting. The question: when capacity is exceeded, does relevance-decay eviction RETAIN the useful bindings, or
//  does it collapse like the toy datasets did? Anti-forgetting stack ON (correctness-gated train + rehearsal +
//  relevance-decay eviction + belief-revision); batched-GPU clouds ON.
// ════════════════════════════════════════════════════════════════════════════════════════════════════

NativeStderr.RedirectToFile(Path.Combine(AppContext.BaseDirectory, "retention-stderr.log"));
var logPath = Path.Combine(AppContext.BaseDirectory, "retention-log.txt");
using var log = new StreamWriter(logPath) { AutoFlush = true };
void P(string s) { Console.WriteLine(s); log.WriteLine(s); }
void Rule() => P(new string('═', 100));

double Arg(int i, double d) => args.Length > i && double.TryParse(args[i], out var v) ? v : d;
var minutes      = Arg(0, 20);
var numCategories = (int)Arg(1, 400);
var members      = (int)Arg(2, 50);     // facts = numCategories * members  (400 * 50 = 20,000 by default)
var hidden       = (int)Arg(3, 512);
var waveSize     = (int)Arg(4, 400);
const int SEED = 7, PROBE_EVERY = 3, ANCHORS = 48;

var dev = torch.cuda.is_available() ? "CUDA" : "CPU";
var work = Path.Combine(Path.GetTempPath(), "retention-" + Guid.NewGuid().ToString("N")[..8]);
Directory.CreateDirectory(work);

// Build the production brain + turn batched-GPU clouds on (the scaling enabler).
var cfg = new GenesisNovaConfig(HiddenSize: hidden, LearningRate: 0.05, Seed: SEED,
    AutoResume: false, AutoPersist: true, LocalStateDirectory: work).WithProductionMechanisms() with { BatchedCloudGpu = true };
var runtime = new GenesisEvalAppRuntime(cfg);

var knowledge = new ProceduralKnowledge(numCategories, members, waveSize, SEED);
var curriculum = new RetentionCurriculum(knowledge, trainPerCycle: 96, probeCount: 48, rehearsalFraction: 0.30, masteryBar: 0.80, seed: SEED);

// RETENTION ANCHORS = a fixed sample of WAVE 0 (the oldest facts), probed identically over the whole run.
var rng = new Random(SEED + 1);
var anchors = knowledge.Wave(0).OrderBy(_ => rng.Next()).Take(ANCHORS)
    .Select(f => (Query: RetentionCurriculum.ProbeFrame(f.Entity), f.Category)).ToList();

Rule();
P("  RETENTION-AT-SCALE — procedural fact stream (entity → category), trained wave by wave");
Rule();
P($"  device     : {dev}    hidden {hidden}    batched-GPU clouds ON");
P($"  universe   : {knowledge.Facts.Length:N0} facts   {numCategories} categories × {members} members   waveSize {waveSize}  ({knowledge.WaveCount} waves)");
P($"  anchors    : {anchors.Count} wave-0 facts re-probed every {PROBE_EVERY} cycles (the forgetting curve)");
P($"  budget     : {minutes} min (press any key to stop early)    workdir {work}");
P($"  space caps : eviction by relevance-decay keeps the active set bounded; we run PAST the node cap on purpose");
Rule();
P($"  {"cyc",4}  {"wave",4}  {"facts",7}  {"nodes",7}  {"rels",8}  {"curAcc",6}  {"W0-ret",6}  {"ex/s",7}  note");
P("  " + new string('─', 96));

double ProbeAnchors()
{
    int hit = 0, tot = 0;
    foreach (var (q, cat) in anchors)
    {
        var pd = runtime.TryPredictAsync(q, 8, 4000).GetAwaiter().GetResult();
        var r = pd?.Result;
        if (r is null) continue;
        tot++;
        if (GenesisGrader.Quality(r.Output, new[] { cat }, 1, r.UsedNeuralFallback, requirePlatonic: false,
                answerVocabulary: knowledge.Categories, surfaceStrict: false) >= 0.5) hit++;
    }
    return tot == 0 ? double.NaN : hit / (double)tot;
}

var sw = System.Diagnostics.Stopwatch.StartNew();
long totalTrained = 0;
double bestW0 = 0, lastW0 = double.NaN;

void OnCycle(CycleMetrics m)
{
    totalTrained += m.TrainedCount;
    if (m.Cycle % PROBE_EVERY != 0) return;
    lastW0 = ProbeAnchors();
    if (!double.IsNaN(lastW0)) bestW0 = Math.Max(bestW0, lastW0);
    var diag = runtime.Diagnose();
    var exPerSec = totalTrained / Math.Max(0.001, sw.Elapsed.TotalSeconds);
    var note = curriculum.Difficulty >= knowledge.WaveCount - 1 ? "ALL WAVES IN" : $"learning w{curriculum.Difficulty}";
    P($"  {m.Cycle,4}  {curriculum.Difficulty,4}  {curriculum.FactsIntroduced,7:N0}  {diag.NodeCount,7:N0}  {diag.RelationCount,8:N0}  {m.Accuracy,6:P0}  {lastW0,6:P0}  {exPerSec,7:F0}  {note}");
}

var cts = new CancellationTokenSource(TimeSpan.FromMinutes(minutes));
_ = Task.Run(() => { if (!Console.IsInputRedirected) { Console.ReadKey(intercept: true); cts.Cancel(); } });

var opt = new GenesisModularTrainingOrchestrator.Options
{
    MasteryBar = 0.80,
    RequirePlatonic = false,           // grade facts on correctness, not which route fired
    WorkDir = work,
    AutosaveSeconds = 120,
    TrainOnFailureOnly = true,          // anti-erosion: train only currently-wrong examples
    ThrottlePercent = () => 0,
};

try { await new GenesisModularTrainingOrchestrator().RunAsync(runtime, curriculum, opt, OnCycle, cts.Token); }
catch (OperationCanceledException) { /* time/keypress budget reached */ }

await runtime.SaveAsync(runtime.AutoCheckpointPath);
var finalDiag = runtime.Diagnose();
var finalW0 = ProbeAnchors();
Rule();
P($"  DONE in {sw.Elapsed.TotalMinutes:F1} min   trained {totalTrained:N0} examples");
P($"  waves introduced : {curriculum.WavesIntroduced}/{knowledge.WaveCount}   facts seen ~{curriculum.FactsIntroduced:N0}");
P($"  space final      : nodes {finalDiag.NodeCount:N0}   relations {finalDiag.RelationCount:N0}   (cap reached ⇒ eviction active)");
P($"  WAVE-0 RETENTION : final {finalW0:P0}   best {bestW0:P0}   ← did the oldest facts survive the whole stream?");
Rule();
Console.WriteLine($"\n(log → {logPath})");

/// <summary>Redirects CRT stderr (fd 2) to a file so libtorch's native notices don't spam the live window (copied
/// from RaceBench — ucrtbase _dup2 on fd 2, since libtorch bypasses Console.SetError).</summary>
internal static class NativeStderr
{
    private const int O_WRONLY = 0x1, O_CREAT = 0x100, O_TRUNC = 0x200, S_IWRITE = 0x80, STDERR_FD = 2;
    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, CharSet = System.Runtime.InteropServices.CharSet.Ansi, EntryPoint = "_open")]
    private static extern int CrtOpen(string filename, int oflag, int pmode);
    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, EntryPoint = "_dup2")]
    private static extern int CrtDup2(int fd1, int fd2);
    [System.Runtime.InteropServices.DllImport("ucrtbase.dll", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl, EntryPoint = "_close")]
    private static extern int CrtClose(int fd);
    public static void RedirectToFile(string path)
    {
        try { var fd = CrtOpen(path, O_WRONLY | O_CREAT | O_TRUNC, S_IWRITE); if (fd >= 0) { CrtDup2(fd, STDERR_FD); CrtClose(fd); } }
        catch { }
    }
}
