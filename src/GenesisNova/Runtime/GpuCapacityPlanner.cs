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
        int reserveVramMb)
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
        return Math.Clamp(hidden, 48, 8192);
    }
}
