using System.Diagnostics;
using GenesisNova.Data;
using GenesisNova.Tokenization;

namespace GenesisNova.Runtime;

public static class GpuCapacityPlanner
{
    public static bool TryGetNvidiaVramMb(out int totalMb, out int freeMb)
    {
        totalMb = 0;
        freeMb = 0;
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=memory.total,memory.free --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var proc = Process.Start(psi);
            if (proc is null)
                return false;
            var output = proc.StandardOutput.ReadToEnd().Trim();
            proc.WaitForExit(2000);
            if (string.IsNullOrWhiteSpace(output))
                return false;

            var firstLine = output
                .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault();
            if (string.IsNullOrWhiteSpace(firstLine))
                return false;

            var parts = firstLine.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
                return false;
            if (!int.TryParse(parts[0], out totalMb))
                return false;
            if (!int.TryParse(parts[1], out freeMb))
                return false;

            return totalMb > 0;
        }
        catch
        {
            return false;
        }
    }

    public static int EstimateHiddenSizeFromDataset(
        IReadOnlyList<GenesisExample> examples,
        int routeCount,
        int vramMb,
        double targetUtilization,
        int reserveVramMb,
        int maxHiddenSize = 6144)
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        foreach (var ex in examples)
        {
            _ = tokenizer.Encode(ex.Input);
            _ = tokenizer.Encode(ex.Output, addEos: true);
        }

        var vocab = Math.Max(8, tokenizer.VocabularySize);
        var usableMb = Math.Max(256, (int)(vramMb * Math.Clamp(targetUtilization, 0.5, 0.98)) - Math.Max(0, reserveVramMb));
        var bytesBudget = usableMb * 1024.0 * 1024.0;

        // Model params ~= emb(v*h) + out(h*v) + route(h*r) + biases.
        var denom = 4.0 * ((2.0 * vocab) + routeCount);
        var numerator = bytesBudget - (4.0 * (vocab + routeCount));
        var hidden = (int)Math.Floor(Math.Max(48.0, numerator / Math.Max(1.0, denom)));
        hidden = (int)Math.Round(hidden * 1.35);
        return Math.Clamp(hidden, 48, Math.Max(48, maxHiddenSize));
    }

    public static int EstimateHiddenSizeForInferenceOnly(
        IReadOnlyList<GenesisExample> examples,
        int routeCount,
        int vramMb,
        double targetUtilization,
        int reserveVramMb,
        Action<string>? debugOutput = null,
        int maxHiddenSize = 6144)
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        foreach (var ex in examples)
        {
            _ = tokenizer.Encode(ex.Input);
            _ = tokenizer.Encode(ex.Output, addEos: true);
        }

        var vocab = Math.Max(8, tokenizer.VocabularySize);
        var usableMb = Math.Max(256, (int)(vramMb * Math.Clamp(targetUtilization, 0.5, 0.98)) - Math.Max(0, reserveVramMb));
        var bytesBudget = usableMb * 1024.0 * 1024.0;

        debugOutput?.Invoke($"[GPU calc] vocab={vocab} usableMb={usableMb} bytesBudget={bytesBudget:F0}");

        // For inference-only: weights + minimal forward activations
        // Params: emb(v*h) + out(h*v) + route(h*r) + biases ~= 4 * (2*vocab*h + h*r + vocab + r)
        // Simplifies to: h ≈ bytesBudget / (4 * (2*vocab + route))
        // (vocab and route constants are negligible for large models)
        var denom = 4.0 * ((2.0 * vocab) + routeCount);
        debugOutput?.Invoke($"[GPU calc] denom={denom:F0}");
        
        var hidden = (int)Math.Floor(Math.Max(48.0, bytesBudget / Math.Max(1.0, denom)));
        hidden = (int)Math.Round(hidden * 1.35);
        debugOutput?.Invoke($"[GPU calc] hidden before clamp={hidden}");
        
        var clamped = Math.Clamp(hidden, 48, Math.Max(48, maxHiddenSize));
        debugOutput?.Invoke($"[GPU calc] hidden after clamp={clamped}");
        
        return clamped;
    }

    public static int ResolveTrainingHiddenCap(int vramMb)
    {
        if (vramMb <= 4096)
            return 512;
        if (vramMb <= 6144)
            return 512;
        if (vramMb <= 8192)
            return 768;

        return 1024;
    }

    public static int EstimateTrainingBatchSize(
        int exampleCount,
        int hiddenSize,
        int averageTargetTokens,
        bool gpuAvailable,
        int vramMb,
        double targetUtilization,
        int reserveVramMb,
        int cpuThreads,
        Action<string>? debugOutput = null)
    {
        if (exampleCount <= 0)
            return 1;

        var sequenceTokens = Math.Max(1, averageTargetTokens);

        if (!gpuAvailable || vramMb <= 0)
        {
            var cpuBatch = Math.Clamp(Math.Max(1, cpuThreads / 2), 1, 8);
            cpuBatch = Math.Min(cpuBatch, exampleCount);
            debugOutput?.Invoke($"[GPU calc] cpu batch={cpuBatch} examples={exampleCount} hidden={hiddenSize} avgTokens={sequenceTokens}");
            return cpuBatch;
        }

        var usableMb = Math.Max(256, (int)(vramMb * Math.Clamp(targetUtilization, 0.5, 0.98)) - Math.Max(0, reserveVramMb));
        var baseBatch = Math.Clamp(usableMb / 256, 2, 64);
        var sequenceFactor = Math.Clamp(16.0 / Math.Max(4.0, sequenceTokens), 0.5, 2.0);
        var hiddenFactor = hiddenSize >= 8192 ? 1.15 : hiddenSize >= 4096 ? 1.05 : 0.95;
        var estimated = (int)Math.Round(baseBatch * sequenceFactor * hiddenFactor);
        var batch = Math.Clamp(Math.Min(exampleCount, estimated), 1, 64);
        debugOutput?.Invoke(
            $"[GPU calc] batch={batch} usableMb={usableMb} baseBatch={baseBatch} hidden={hiddenSize} avgTokens={sequenceTokens}");
        return batch;
    }
}
