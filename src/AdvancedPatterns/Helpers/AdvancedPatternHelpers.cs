using System.Security.Cryptography;
using System.Text;

namespace EvalApp.Solid.Starter.Features.AdvancedPatterns;

public static class AdvancedPatternHelpers
{
    public static async IAsyncEnumerable<int> StreamItemsAsync(IEnumerable<int> input)
    {
        foreach (var item in input)
        {
            await Task.Yield();
            yield return item;
        }
    }

    public static async ValueTask<AdvancedDemoData> ComputeDigestAsync(AdvancedDemoData data, CancellationToken ct)
    {
        var values = data.MaterializedItems ?? [];
        var payloadBuilder = new StringBuilder();

        foreach (var value in values)
        {
            ct.ThrowIfCancellationRequested();
            payloadBuilder.Append(value.ToString("X"));
            await Task.Yield();
        }

        using var sha256 = SHA256.Create();
        var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(payloadBuilder.ToString()));
        var digest = Convert.ToHexString(hash);

        return data.AppendTrace("Cpu:DigestComputed") with { CpuDigest = digest };
    }

    public static async ValueTask<AdvancedDemoData> PersistSnapshotAsync(AdvancedDemoData data, CancellationToken ct)
    {
        var path = Path.Combine(
            Path.GetTempPath(),
            $"evalapp-solid-advanced-{Guid.NewGuid():N}.txt");

        var payload = new StringBuilder()
            .Append("Quote=").Append(data.Quote)
            .Append(";Source=").Append(data.QuoteSource)
            .Append(";Success=").Append(data.SuccessCount)
            .Append(";Error=").Append(data.ErrorCount)
            .Append(";Items=").Append(string.Join(",", data.MaterializedItems ?? []))
            .ToString();

        await File.WriteAllTextAsync(path, payload, ct);
        return data.AppendTrace("Disk:SnapshotWritten") with { SnapshotPath = path };
    }
}
