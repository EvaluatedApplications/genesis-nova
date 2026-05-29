using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Repl;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public static class GenesisCli
{
    public static async Task<bool> TryHandleAsync(string[] args)
    {
        if (args.Contains("--genesis-repl", StringComparer.OrdinalIgnoreCase))
        {
            await new GenesisRepl(new GenesisEvalAppRuntime()).RunAsync();
            return true;
        }

        if (args.Contains("--genesis-gen-examples", StringComparer.OrdinalIgnoreCase))
        {
            var count = ParseInt(args, "--count", fallback: 100);
            var difficulty = ParseInt(args, "--difficulty", fallback: 0);
            var output = ResolvePath(ReadArg(args, "--output"));

            if (string.IsNullOrWhiteSpace(output))
                throw new ArgumentException("Missing --output for --genesis-gen-examples.");

            var examples = GenesisTrainingOrchestrator.GenerateExamplesFromCreators(count, difficulty);
            GenesisTrainingDataLoader.SaveToFile(output, examples);
            Console.WriteLine($"generated count={count} difficulty={difficulty} saved={output}");
            return true;
        }

        if (!args.Contains("--genesis-train", StringComparer.OrdinalIgnoreCase))
            return false;

        var file = ResolvePath(ReadArg(args, "--file"));
        if (string.IsNullOrWhiteSpace(file))
            throw new ArgumentException("Missing --file for --genesis-train.");

        var epochs = ParseInt(args, "--epochs", fallback: 3);
        var evalSamples = ParseNullableInt(args, "--eval-samples");
        var savePath = ResolvePath(ReadArg(args, "--save"));
        var logPath = ResolvePath(ReadArg(args, "--log"));
        var threads = ParseNullableInt(args, "--threads");
        var hiddenSize = ParseNullableInt(args, "--hidden-size");
        var backend = ParseBackend(ReadArg(args, "--backend"));
        var autoScaleVram = backend == ComputeBackend.Gpu
            && !args.Contains("--no-auto-scale-vram", StringComparer.OrdinalIgnoreCase);
        var targetVramUtil = ParseDouble(args, "--target-vram-util", fallback: 0.9);
        var reserveVramMb = ParseInt(args, "--reserve-vram-mb", fallback: 512);
        var deterministic = args.Contains("--deterministic", StringComparer.OrdinalIgnoreCase);
        var parallelMath = !args.Contains("--no-parallel-math", StringComparer.OrdinalIgnoreCase) && !deterministic;
        var baselineCheckpoint = ResolvePath(ReadArg(args, "--baseline-checkpoint"));
        var maxExactDrop = ParseDouble(args, "--max-exact-drop", fallback: 0.01);
        var effectiveThreads = deterministic ? 1 : (threads ?? 0);

        var resolvedHidden = hiddenSize ?? 8192;
        if (autoScaleVram && backend == ComputeBackend.Gpu)
        {
            var examples = GenesisTrainingDataLoader.LoadFromFile(file);
            if (GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalMb, out var freeMb))
            {
                var usableVramMb = freeMb > 0 ? freeMb : totalMb;
                resolvedHidden = Math.Max(resolvedHidden, GpuCapacityPlanner.EstimateHiddenSizeFromDataset(
                    examples,
                    routeCount: 8,
                    vramMb: usableVramMb,
                    targetUtilization: targetVramUtil,
                    reserveVramMb: reserveVramMb));
                Console.WriteLine(
                    $"autoscale vram totalMb={totalMb} freeMb={freeMb} targetUtil={targetVramUtil:F2} reserveMb={reserveVramMb} hidden={resolvedHidden}");
            }
            else
            {
                Console.WriteLine("autoscale skipped: could not query NVIDIA VRAM.");
            }
        }

        var runtime = new GenesisEvalAppRuntime(new GenesisNovaConfig(
            HiddenSize: resolvedHidden,
            EnableParallelMath: parallelMath,
            MaxDegreeOfParallelism: effectiveThreads,
            Deterministic: deterministic,
            Backend: backend,
            AutoPersist: true,
            AutoResume: false,
            AutoScaleVram: autoScaleVram,
            TargetVramUtilization: targetVramUtil,
            ReserveVramMb: reserveVramMb));

        Console.WriteLine(
            $"runtime backend={backend} hidden={resolvedHidden} deterministic={deterministic} " +
            $"parallelMath={parallelMath} threads={(deterministic ? 1 : (threads ?? Environment.ProcessorCount))}");

        var report = await runtime.TrainAsync(
            filePath: file,
            epochs: epochs,
            savePath: savePath,
            logPath: logPath);

        Console.WriteLine(
            $"done examples={report.ExampleCount} epochs={report.Epochs} " +
            $"loss={report.AverageLoss.TotalLoss:F4} contradiction={report.ContradictionRate:F4}");

        var eval = await runtime.EvaluateFileAsync(file, evalSamples);
        Console.WriteLine(
            $"accuracy exact={eval.ExactMatchAccuracy:P2} ({eval.ExactMatchCount}/{eval.SampleCount}) " +
            $"route={eval.RouteAccuracy:P2} ({eval.RouteCorrectCount}/{eval.RouteLabeledCount})");

        if (!string.IsNullOrWhiteSpace(baselineCheckpoint))
        {
            var baselineRuntime = new GenesisEvalAppRuntime(new GenesisNovaConfig(
                EnableParallelMath: false,
                MaxDegreeOfParallelism: 1,
                Deterministic: true,
                Backend: ComputeBackend.Cpu,
                AutoPersist: false,
                AutoResume: false,
                AutoScaleVram: false));
            await baselineRuntime.LoadAsync(baselineCheckpoint);
            var baselineEval = await baselineRuntime.EvaluateFileAsync(file, evalSamples);
            var exactDelta = eval.ExactMatchAccuracy - baselineEval.ExactMatchAccuracy;
            Console.WriteLine(
                $"parity baselineExact={baselineEval.ExactMatchAccuracy:P2} " +
                $"newExact={eval.ExactMatchAccuracy:P2} delta={exactDelta:P2}");

            if (exactDelta < -Math.Abs(maxExactDrop))
                throw new InvalidOperationException(
                    $"Parity gate failed: exact accuracy dropped by {Math.Abs(exactDelta):P2} " +
                    $"which exceeds allowed {Math.Abs(maxExactDrop):P2}.");
        }

        if (!string.IsNullOrWhiteSpace(savePath))
            Console.WriteLine($"saved={savePath}");

        var preview = await runtime.PredictAsync("hello", 8);
        Console.WriteLine(
            $"preview output={preview.Result?.Output ?? string.Empty} " +
            $"route={preview.Result?.DecisionPath ?? "unknown"} " +
            $"confidence={(preview.Result?.PlatonicConfidence ?? 0.0):F2} " +
            $"hops={(preview.Result?.PlatonicHopCount ?? 0)} " +
            $"fallback={(preview.Result?.UsedNeuralFallback ?? false)} " +
            $"chunks={(preview.Result?.ChunksGenerated ?? 0)}");

        return true;
    }

    private static string? ReadArg(string[] args, string name)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }

    private static int ParseInt(string[] args, string name, int fallback)
    {
        var raw = ReadArg(args, name);
        if (int.TryParse(raw, out var parsed))
            return Math.Max(1, parsed);
        return fallback;
    }

    private static int? ParseNullableInt(string[] args, string name)
    {
        var raw = ReadArg(args, name);
        if (int.TryParse(raw, out var parsed))
            return Math.Max(1, parsed);
        return null;
    }

    private static double ParseDouble(string[] args, string name, double fallback)
    {
        var raw = ReadArg(args, name);
        if (double.TryParse(raw, out var parsed))
            return parsed;
        return fallback;
    }

    private static string? ResolvePath(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw) || Path.IsPathRooted(raw))
            return raw;

        var cwd = Directory.GetCurrentDirectory();
        var candidate = Path.GetFullPath(Path.Combine(cwd, raw));
        if (File.Exists(candidate) || Directory.Exists(candidate))
            return candidate;

        var parent = Directory.GetParent(cwd)?.FullName;
        if (!string.IsNullOrWhiteSpace(parent))
        {
            candidate = Path.GetFullPath(Path.Combine(parent, raw));
            if (File.Exists(candidate) || Directory.Exists(candidate))
                return candidate;
        }

        return Path.GetFullPath(Path.Combine(cwd, raw));
    }

    private static ComputeBackend ParseBackend(string? raw)
        => raw?.Trim().ToLowerInvariant() switch
        {
            "gpu" => ComputeBackend.Gpu,
            "cpu" => ComputeBackend.Cpu,
            _ => ComputeBackend.Gpu  // Default to GPU
        };
}
