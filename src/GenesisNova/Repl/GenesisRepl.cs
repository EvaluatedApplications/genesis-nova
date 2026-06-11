using GenesisNova.Data;
using GenesisNova.Core;
using GenesisNova.Runtime;
using System.Diagnostics;

namespace GenesisNova.Repl;

public sealed class GenesisRepl
{
    private readonly GenesisEvalAppRuntime _runtime;

    public GenesisRepl(GenesisEvalAppRuntime? runtime = null)
    {
        _runtime = runtime ?? new GenesisEvalAppRuntime(new GenesisNovaConfig
        {
            Backend = ComputeBackend.Gpu,
            HiddenSize = 512,
            AutoScaleVram = true
        });
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        Console.WriteLine("Genesis Nova REPL (model + platonic space)");
        Console.WriteLine("Training stays on CPU; inference uses VRAM-backed weights when available.");
        Console.WriteLine("Type 'help' for commands.");

        // Detect if stdin is available (not redirected in Debug console)
        bool isInteractive = !Console.IsInputRedirected;
        
        if (!isInteractive)
        {
            // In Visual Studio debug console or when stdin is not available
            Console.WriteLine();
            Console.WriteLine("════════════════════════════════════════════════════════════════");
            Console.WriteLine("⚠ Running in non-interactive mode (VS Debug console detected)");
            Console.WriteLine("════════════════════════════════════════════════════════════════");
            Console.WriteLine();
            Console.WriteLine("✓ To use REPL interactively, run from PowerShell/CMD:");
            Console.WriteLine();
            Console.WriteLine("  cd src\\bin\\Release\\net8.0");
            Console.WriteLine("  .\\GenesisNova.exe --genesis-repl");
            Console.WriteLine();
            PrintHelp();
            Console.WriteLine();
            Console.WriteLine("Press any key to close...");
            Console.ReadKey(intercept: true);
            return;
        }

        while (!ct.IsCancellationRequested)
        {
            Console.Write("genesis> ");
            var line = Console.ReadLine();
            if (line is null)
                break;

            var trimmed = line.Trim();
            if (trimmed.Length == 0)
                continue;
            if (trimmed.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;
            if (trimmed.Equals("help", StringComparison.OrdinalIgnoreCase))
            {
                PrintHelp();
                continue;
            }

            try
            {
                var output = await HandleAsync(trimmed, ct);
                if (!string.IsNullOrWhiteSpace(output))
                    Console.WriteLine(output);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"error: {ex.Message}");
            }
        }

    }

    private async Task<string?> HandleAsync(string line, CancellationToken ct)
    {
        if (line.StartsWith("config ", StringComparison.OrdinalIgnoreCase))
        {
            var subcommand = line["config ".Length..].Trim();
            return HandleConfig(subcommand);
        }

        if (line.StartsWith("trainfile ", StringComparison.OrdinalIgnoreCase))
        {
            var payload = line["trainfile ".Length..].Trim();
            var (path, epochs) = ParsePathAndEpochs(payload);
            var sw = Stopwatch.StartNew();
            var report = await _runtime.TrainAsync(path, epochs);
            sw.Stop();
            return
                $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4} " +
                $"success={report.ExampleSuccessRate:P1} time={sw.ElapsedMilliseconds / 1000.0:F2}s";
        }

        if (line.StartsWith("train ", StringComparison.OrdinalIgnoreCase))
        {
            var ex = ParseExample(line["train ".Length..].Trim());
            var loss = await _runtime.TrainOneAsync(ex);
            return $"loss={loss.TotalLoss:F4} token={loss.TokenLoss:F4} route={loss.RouteLoss:F4}";
        }

        if (line.StartsWith("predict ", StringComparison.OrdinalIgnoreCase))
        {
            var input = line["predict ".Length..].Trim();
            var prediction = await _runtime.PredictAsync(input);
            var predictionResult = prediction.Result!;
            return $"output={predictionResult.Output} route={predictionResult.DecisionPath} confidence={predictionResult.PlatonicConfidence:F2} hops={predictionResult.PlatonicHopCount} fallback={predictionResult.UsedNeuralFallback} biasCount={predictionResult.AppliedBiasCount} chunks={predictionResult.ChunksGenerated}";
        }

        if (line.StartsWith("save ", StringComparison.OrdinalIgnoreCase))
        {
            var path = line["save ".Length..].Trim();
            await _runtime.SaveAsync(path);
            return $"saved {path}";
        }

        if (line.StartsWith("load ", StringComparison.OrdinalIgnoreCase))
        {
            var path = line["load ".Length..].Trim();
            await _runtime.LoadAsync(path);
            return $"loaded {path}";
        }

        if (line.Equals("stats", StringComparison.OrdinalIgnoreCase))
        {
            return $"vocab={_runtime.VocabularySize} hidden={_runtime.HiddenSize}";
        }

        var fallbackPrediction = await _runtime.PredictAsync(line);
        var fallbackResult = fallbackPrediction.Result!;
        return $"output={fallbackResult.Output} route={fallbackResult.DecisionPath} confidence={fallbackResult.PlatonicConfidence:F2} hops={fallbackResult.PlatonicHopCount} fallback={fallbackResult.UsedNeuralFallback} biasCount={fallbackResult.AppliedBiasCount} chunks={fallbackResult.ChunksGenerated}";
    }

    private string HandleConfig(string subcommand)
    {
        var parts = subcommand.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
        {
            return PrintConfigMenu();
        }

        var cmd = parts[0].ToLowerInvariant();
        if (cmd == "l2" || cmd == "regularization")
        {
            return HandleL2Config(parts);
        }

        if (cmd == "help")
        {
            return PrintConfigMenu();
        }

        return "Unknown config command. Type 'config help' for options.";
    }

    private string HandleL2Config(string[] parts)
    {
        if (parts.Length == 1)
        {
            // Show current L2 config
            var current = _runtime.Config.L2RegularizationCoefficient;
            return PrintL2Menu(current);
        }

        if (parts.Length >= 2)
        {
            if (parts[1] == "off")
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = 0.0 });
                return "✓ L2 regularization: OFF (0.0)\n  Effect: No compression penalty; maximum neural learning capacity\n  Use when: Learning quality is priority, compression can be enabled later";
            }

            if (parts[1] == "mild")
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = 1e-5 });
                return "✓ L2 regularization: MILD (1e-5)\n  Effect: Weights can grow; neural layer stores most knowledge\n  Use when: Model needs capacity, early training phases";
            }

            if (parts[1] == "balanced")
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = 1e-4 });
                return "✓ L2 regularization: BALANCED (1e-4)\n  Effect: Moderate weight penalty; good balance\n  Use when: Want some platonic learning without starving neural network";
            }

            if (parts[1] == "aggressive")
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = 1e-3 });
                return "✓ L2 regularization: AGGRESSIVE (1e-3)\n  Effect: Heavy weight penalty; forces symbolic learning\n  Use when: Want to maximize platonic space usage, compress knowledge";
            }

            if (parts[1] == "extreme")
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = 1e-2 });
                return "✓ L2 regularization: EXTREME (1e-2)\n  Effect: Neural layer heavily constrained; nearly all learning symbolic\n  Use when: Debugging, understanding platonic discovery; may hurt performance";
            }

            if (double.TryParse(parts[1], out var value) && value >= 0)
            {
                _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = value });
                return $"✓ L2 regularization set to: {value:E2}";
            }
        }

        return PrintL2Menu(_runtime.Config.L2RegularizationCoefficient);
    }

    private string PrintL2Menu(double current)
    {
        return $"""
        ╔════════════════════════════════════════════════════════════════════════╗
        ║            L2 REGULARIZATION (Weight Decay) Configuration              ║
        ║                         Current: {current:E2}                           ║
        ╚════════════════════════════════════════════════════════════════════════╝
        
        Weight regularization penalizes large neural weights, forcing the model to 
        compress knowledge into the platonic symbolic space instead of bloating the 
        neural layer.
        
        Preset Options:
        ─────────────────────────────────────────────────────────────────────────
        config l2 off            (0.0)   → No compression
                                           - No weight penalty
                                           - Maximum neural learning capacity
                                           - Use: When learning quality is the immediate goal
        
        config l2 mild           (1e-5)  → Minimal pressure; neural layer is comfortable
                                           - Allows weight growth if needed
                                           - Neural network stores most information
                                           - Use: Early training, testing model capacity
        
        config l2 balanced        (1e-4)  → Medium pressure; good default
                                           - Weights stay moderate; gradual symbolic learning
                                           - Balanced neural + platonic knowledge
                                           - Use: Standard training, most scenarios
        
        config l2 aggressive      (1e-3)  → High pressure; forces symbolic learning ⚡
                                           - Heavy penalty on weight magnitude
                                           - Model learns to compress into platonic space
                                           - Use: Maximize symbolic reasoning, lean models
        
        config l2 extreme         (1e-2)  → Severe pressure; nearly all symbolic
                                           - Neural layer heavily constrained
                                           - Almost all information in platonic space
                                           - Use: Debugging, understanding discovery
                                           - Warning: May hurt performance/convergence
        
        Custom:
        ─────────────────────────────────────────────────────────────────────────
        config l2 <value>                 → Set custom coefficient (e.g., 5e-4)
        
        How It Works:
        ─────────────────────────────────────────────────────────────────────────
        loss = token_loss + route_loss + λ × 0.5 × Σ(weight²)
        
        Higher λ → weights must stay small → neural storage capacity shrinks → 
        model compresses knowledge into platonic space (symbolic, interpretable)
        
        Like natural selection: "You can't store much in neural weights? Then learn
        to encode it symbolically where it persists and is debuggable."
        """;
    }

    private string PrintConfigMenu()
    {
        return $"""
        ╔════════════════════════════════════════════════════════════════════════╗
        ║                    Genesis Nova Configuration Menu                    ║
        ╚════════════════════════════════════════════════════════════════════════╝
        
        config l2                       → Show L2 regularization menu
        config l2 <preset>              → Set L2 (off|mild|balanced|aggressive|extreme)
        config l2 <value>               → Set custom L2 coefficient
        config help                     → Show this menu
        """;
    }

    private static (string Path, int Epochs) ParsePathAndEpochs(string payload)
    {
        var parts = payload.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            throw new FormatException("trainfile requires a path.");
        var epochs = 1;
        if (parts.Length > 1 && int.TryParse(parts[1], out var parsed))
            epochs = Math.Max(1, parsed);
        return (parts[0], epochs);
    }

    private static GenesisExample ParseExample(string payload)
    {
        var arrow = payload.IndexOf("=>", StringComparison.Ordinal);
        if (arrow < 1 || arrow >= payload.Length - 2)
            throw new FormatException("Expected: train <input> => <output>");

        var input = payload[..arrow].Trim();
        var right = payload[(arrow + 2)..].Trim();
        var output = right;
        int? routeLabel = null;

        var pipe = right.IndexOf('|');
        if (pipe >= 0)
        {
            output = right[..pipe].Trim();
            routeLabel = ParseRouteLabel(right[(pipe + 1)..]);
        }

        return new GenesisExample(input, output, RouteLabel: routeLabel);
    }

    private static int? ParseRouteLabel(string metadata)
    {
        foreach (var part in metadata.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var keyValue = part.Split(new[] { '=', ':' }, 2, StringSplitOptions.TrimEntries);
            if (keyValue.Length == 2 &&
                keyValue[0].Equals("route", StringComparison.OrdinalIgnoreCase) &&
                int.TryParse(keyValue[1], out var parsed) &&
                parsed is >= 0 and <= 2)
                return parsed;
        }

        return null;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("\n=== Training ===");
        Console.WriteLine("train <input> => <output>                 - Train single example");
        Console.WriteLine("trainfile <path> [epochs]                 - Train from file");
        Console.WriteLine("\n=== Configuration ===");
        Console.WriteLine("config                                    - Show configuration menu");
        Console.WriteLine("config l2                                 - Show L2 regularization settings");
        Console.WriteLine("config l2 <preset>                        - Set L2 (off|mild|balanced|aggressive|extreme)");
        Console.WriteLine("\n=== Queries ===");
        Console.WriteLine("predict <input>                           - Generate output via the model + platonic space");
        Console.WriteLine("<any text>                                - Same as predict <any text>");
        Console.WriteLine("stats                                     - Model statistics");
        Console.WriteLine("\n=== State ===");
        Console.WriteLine("save <path>                               - Save checkpoint");
        Console.WriteLine("load <path>                               - Load checkpoint");
        Console.WriteLine("help                                      - This help");
        Console.WriteLine("exit                                      - Exit REPL");
        Console.WriteLine();
    }
}
