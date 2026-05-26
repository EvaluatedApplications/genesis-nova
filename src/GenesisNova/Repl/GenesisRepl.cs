using GenesisNova.Data;
using GenesisNova.Runtime;

namespace GenesisNova.Repl;

public sealed class GenesisRepl
{
    private readonly GenesisEvalAppRuntime _runtime;

    public GenesisRepl(GenesisEvalAppRuntime? runtime = null)
    {
        _runtime = runtime ?? new GenesisEvalAppRuntime();
    }

    public async Task RunAsync(CancellationToken ct = default)
    {
        Console.WriteLine("Genesis Nova REPL (differentiable core)");
        Console.WriteLine("Commands: train, trainfile, predict, introspect, concept, relate, queue, save, load, stats, context, compact, reset, help, exit");

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
                var output = await HandleAsync(trimmed);
                if (!string.IsNullOrWhiteSpace(output))
                    Console.WriteLine(output);
                await _runtime.ObserveConversationAsync(trimmed, output ?? string.Empty, resetSignal: trimmed.Equals("reset", StringComparison.OrdinalIgnoreCase));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"error: {ex.Message}");
            }
        }

    }

    private async Task<string?> HandleAsync(string line)
    {
        if (line.StartsWith("trainfile ", StringComparison.OrdinalIgnoreCase))
        {
            var payload = line["trainfile ".Length..].Trim();
            var (path, epochs) = ParsePathAndEpochs(payload);
            var report = await _runtime.TrainAsync(path, epochs);
            return $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4}";
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
        Console.WriteLine("train <input> => <output> [| route=N]");
        Console.WriteLine("trainfile <path> [epochs]");
        Console.WriteLine("predict <input>");
        Console.WriteLine("introspect [cycles]");
        Console.WriteLine("concept <name>");
        Console.WriteLine("relate <left> <right> <contradiction0to1>");
        Console.WriteLine("queue");
        Console.WriteLine("save <path>");
        Console.WriteLine("load <path>");
        Console.WriteLine("stats");
        Console.WriteLine("context");
        Console.WriteLine("compact");
        Console.WriteLine("reset");
        Console.WriteLine("exit");
    }
}
