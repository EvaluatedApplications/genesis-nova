using GenesisNova.Data;
using GenesisNova.Runtime;
using System.Diagnostics;

namespace GenesisNova.Repl;

public sealed class GenesisRepl
{
    private readonly GenesisEvalAppRuntime _runtime;
    private CancellationTokenSource? _idleIntrospectionCts;
    private Task? _idleIntrospectionTask;
    private bool _verbose = false;

    public GenesisRepl(GenesisEvalAppRuntime? runtime = null)
    {
        _runtime = runtime ?? new GenesisEvalAppRuntime();
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        Console.WriteLine("Genesis Nova REPL (differentiable core)");
        Console.WriteLine("Type 'help' for commands. Type 'introspect-idle' to enable background introspection.");

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
                await _runtime.ObserveConversationAsync(trimmed, output ?? string.Empty, resetSignal: trimmed.Equals("reset", StringComparison.OrdinalIgnoreCase));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"error: {ex.Message}");
            }
        }

        // Stop idle introspection on exit
        await StopIdleIntrospectionAsync();
    }

    private async Task<string?> HandleAsync(string line, CancellationToken ct)
    {
        if (line.Equals("introspect-idle", StringComparison.OrdinalIgnoreCase))
        {
            if (_idleIntrospectionTask != null)
                return "idle introspection already running";
            StartIdleIntrospection();
            return "idle introspection started (runs continuously in background)";
        }

        if (line.Equals("introspect-stop", StringComparison.OrdinalIgnoreCase))
        {
            await StopIdleIntrospectionAsync();
            return "idle introspection stopped";
        }

        if (line.Equals("verbose", StringComparison.OrdinalIgnoreCase))
        {
            _verbose = !_verbose;
            return $"verbose mode: {(_verbose ? "ON" : "OFF")}";
        }

        if (line.StartsWith("trainfile ", StringComparison.OrdinalIgnoreCase))
        {
            var payload = line["trainfile ".Length..].Trim();
            var (path, epochs) = ParsePathAndEpochs(payload);
            var sw = Stopwatch.StartNew();
            var report = await _runtime.TrainAsync(path, epochs);
            sw.Stop();
            return $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4} time={sw.ElapsedMilliseconds / 1000.0:F2}s";
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
            var predict = await _runtime.PredictAsync(input);
            var result = predict.Result!;
            return $"output={result.Output}";
        }

        if (line.StartsWith("introspect", StringComparison.OrdinalIgnoreCase))
        {
            var payload = line["introspect".Length..].Trim();
            var cycles = 1;
            if (payload.Length > 0 && int.TryParse(payload, out var parsed))
                cycles = Math.Max(1, parsed);
            var state = await _runtime.IntrospectAsync(cycles);
            return $"introspected={state.Processed} queue={state.QueueDepth}";
        }

        if (line.StartsWith("concept ", StringComparison.OrdinalIgnoreCase))
        {
            var concept = line["concept ".Length..].Trim();
            return await _runtime.DescribeConceptAsync(concept);
        }

        if (line.StartsWith("relate ", StringComparison.OrdinalIgnoreCase))
        {
            var parts = line["relate ".Length..]
                .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length < 3 || !double.TryParse(parts[2], out var contradiction))
                throw new FormatException("Expected: relate <left> <right> <contradiction0to1>");
            await _runtime.RelateAsync(parts[0], parts[1], contradiction);
            return "relation updated";
        }

        if (line.Equals("queue", StringComparison.OrdinalIgnoreCase))
        {
            return $"queue={_runtime.QueueSize}";
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
            return $"vocab={_runtime.VocabularySize} hidden={_runtime.HiddenSize} queue={_runtime.QueueSize}";
        }

        if (line.Equals("context", StringComparison.OrdinalIgnoreCase))
        {
            return _runtime.ConversationBrief;
        }

        if (line.Equals("compact", StringComparison.OrdinalIgnoreCase))
        {
            var result = await _runtime.CompactConversationAsync("manual compact");
            return $"compacted={result.Compacted} turns={result.RecentTurnCount}";
        }

        if (line.Equals("reset", StringComparison.OrdinalIgnoreCase))
        {
            return "reset signal recorded; memory retained and compacted";
        }

        return "unknown command";
    }

    private void StartIdleIntrospection()
    {
        _idleIntrospectionCts = new CancellationTokenSource();
        _idleIntrospectionTask = Task.Run(async () =>
        {
            var cycleCount = 0;
            while (!_idleIntrospectionCts.Token.IsCancellationRequested)
            {
                try
                {
                    cycleCount++;
                    var state = await _runtime.IntrospectAsync(1);
                    if (_verbose && cycleCount % 10 == 0)
                        Console.WriteLine($"[idle] introspected={cycleCount} queue={state.QueueDepth}");
                    await Task.Delay(100, _idleIntrospectionCts.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    if (_verbose)
                        Console.WriteLine($"[idle] error: {ex.Message}");
                }
            }
        });
    }

    private async Task StopIdleIntrospectionAsync()
    {
        if (_idleIntrospectionCts != null)
        {
            _idleIntrospectionCts.Cancel();
            if (_idleIntrospectionTask != null)
            {
                try
                {
                    await _idleIntrospectionTask;
                }
                catch (OperationCanceledException)
                {
                    // Expected
                }
            }
            _idleIntrospectionCts.Dispose();
            _idleIntrospectionCts = null;
            _idleIntrospectionTask = null;
        }
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
            throw new FormatException("Expected: train <input> => <output> [| route=N]");

        var input = payload[..arrow].Trim();
        var right = payload[(arrow + 2)..].Trim();
        var output = right;
        int? route = null;

        var pipe = right.IndexOf('|');
        if (pipe >= 0)
        {
            output = right[..pipe].Trim();
            var meta = right[(pipe + 1)..].Trim();
            foreach (var field in meta.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (!field.StartsWith("route=", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (int.TryParse(field[6..].Trim(), out var parsed))
                    route = parsed;
            }
        }

        return new GenesisExample(input, output, route);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("\n=== Training ===");
        Console.WriteLine("train <input> => <output> [| route=N]     - Train single example");
        Console.WriteLine("trainfile <path> [epochs]                 - Train from file");
        Console.WriteLine("\n=== Introspection ===");
        Console.WriteLine("introspect [cycles]                       - Manual introspection");
        Console.WriteLine("introspect-idle                           - Start background introspection");
        Console.WriteLine("introspect-stop                           - Stop background introspection");
        Console.WriteLine("\n=== Queries ===");
        Console.WriteLine("predict <input>                           - Generate output");
        Console.WriteLine("concept <name>                            - Describe concept");
        Console.WriteLine("relate <left> <right> <contradiction>    - Add relation");
        Console.WriteLine("queue                                     - Queue depth");
        Console.WriteLine("stats                                     - Model statistics");
        Console.WriteLine("context                                   - Conversation brief");
        Console.WriteLine("\n=== State ===");
        Console.WriteLine("save <path>                               - Save checkpoint");
        Console.WriteLine("load <path>                               - Load checkpoint");
        Console.WriteLine("compact                                   - Compact memory");
        Console.WriteLine("reset                                     - Reset signal");
        Console.WriteLine("verbose                                   - Toggle verbose mode");
        Console.WriteLine("help                                      - This help");
        Console.WriteLine("exit                                      - Exit REPL");
        Console.WriteLine();
    }
}
