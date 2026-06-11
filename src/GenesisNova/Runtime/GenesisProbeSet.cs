using GenesisNova.Infer;

namespace GenesisNova.Runtime;

public sealed class GenesisProbeSet
{
    private static readonly GenesisProbe[] Probes =
    [
        new("1+1", "2"),
        new("2+3", "5"),
        new("4-1", "3"),
        new("2*3", "6"),
        new("8/2", "4"),
        new("hello", "hello!"),
        new("thanks", "you're welcome!"),
        new("ok", "ok!"),
        new("what is the capital of france", "Paris."),
        new("how many days in a week", "Seven.")
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
