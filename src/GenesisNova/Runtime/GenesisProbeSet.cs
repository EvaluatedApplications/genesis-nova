using GenesisNova.Infer;

namespace GenesisNova.Runtime;

public sealed class GenesisProbeSet
{
    // Deterministic, single-answer probes that the system is ACTUALLY trained on — a representative slice of the gym
    // skills, so the journal headline reflects real capability (the old set was half chat/world-facts the gym never
    // trains, capping the score near 0.5 regardless of progress). Retrieval (multi-answer) is excluded — it belongs
    // in the value-aware per-cycle accuracy, not a surface-strict smoke gate.
    private static readonly GenesisProbe[] Probes =
    [
        new("1 + 1", "2"),
        new("9 - 2", "7"),
        new("3 x 4", "12"),
        new("2 + 5 + 3", "10"),
        new("3 x 4 + 2", "14"),
        new("7 compared to 4", "greater"),
        new("what is 3 plus 4", "7"),
        new("5 in words", "five"),
        new("3 + 4 in words", "seven"),
        new("fn 2 is 4 fn 3 is 6 fn 5 is", "10")
    ];

    public GenesisProbeReport Evaluate(GenesisRuntimeState state)
    {
        var results = new List<GenesisProbeResult>(Probes.Length);
        foreach (var probe in Probes)
        {
            var generation = state.Inference.Generate(new GenerationRequest(
                Input: probe.Prompt,
                MaxNewTokens: Math.Max(4, probe.Expected.Length + 4),
                ChunkTokenBudget: 16));
            var output = generation.Output ?? string.Empty;
            var passed = IsMatch(output, probe.Expected);
            results.Add(new GenesisProbeResult(probe.Prompt, probe.Expected, output, passed));
        }

        var score = results.Count == 0 ? 0.0 : results.Count(r => r.Passed) / (double)results.Count;
        return new GenesisProbeReport(DateTimeOffset.UtcNow, score, results.ToArray());
    }

    private static bool IsMatch(string output, string expected)
    {
        var normalizedOutput = Normalize(output);
        var normalizedExpected = Normalize(expected);
        return normalizedOutput.Equals(normalizedExpected, StringComparison.OrdinalIgnoreCase)
            || normalizedOutput.StartsWith(normalizedExpected + " ", StringComparison.OrdinalIgnoreCase);
    }

    private static string Normalize(string text)
        => string.Join(' ', (text ?? string.Empty).Trim().Split(
            [' ', '\t', '\r', '\n'],
            StringSplitOptions.RemoveEmptyEntries));
}

public sealed record GenesisProbe(string Prompt, string Expected);

public sealed record GenesisProbeResult(
    string Prompt,
    string Expected,
    string Output,
    bool Passed);

public sealed record GenesisProbeReport(
    DateTimeOffset TimestampUtc,
    double Score,
    GenesisProbeResult[] Results);
