using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using GenesisNova.Cognition;

namespace GenesisNova.Persistence;

/// <summary>
/// Binary, content-addressed, SHARDED checkpoint storage (see MODEL_STORAGE.md). A model is a directory of small
/// binary shards + a text manifest, so it fits git (every shard &lt; the 50&#160;MB GitHub warning), dedups
/// identical shards across forks/saves, and makes a "better model" PR a reviewable manifest diff. Weights are raw
/// little-endian <c>double</c> (exact resume for forkers); small structured state is UTF-8 JSON. Both are chunked
/// the same way. The platonic substrate is a SEPARATE sharded directory so it stays independently resettable.
/// </summary>
public static class GenesisShardedCheckpointStore
{
    /// <summary>Target shard size. 32&#160;MiB: under GitHub's 50&#160;MB warning, ~4 shards per ~100&#160;MB GRU
    /// matrix, ~15 for a full 2048 model — a flat, simple shards/ dir. See MODEL_STORAGE.md §3.</summary>
    public const long TargetShardBytes = 32L * 1024 * 1024;
    public const int FormatVersion = 1;

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

    // ── Path helpers ──────────────────────────────────────────────────────────────────────────────────────
    /// <summary>Sharded model directory for a (legacy-style) checkpoint path: strips the ".json".</summary>
    public static string ModelDir(string checkpointPath) => Path.ChangeExtension(checkpointPath, null)!;
    /// <summary>Companion substrate directory — separate so it can be wiped without touching the NN.</summary>
    public static string SubstrateDir(string checkpointPath) => ModelDir(checkpointPath) + ".platonic";
    public static bool ModelExists(string modelDir) => File.Exists(Path.Combine(modelDir, "manifest.json"));

    // ── Write ─────────────────────────────────────────────────────────────────────────────────────────────
    public static void WriteModel(string modelDir, GenesisCheckpoint nn)
    {
        var sections = new Dictionary<string, Section>(StringComparer.Ordinal);
        var shardsDir = Path.Combine(modelDir, "shards");
        Directory.CreateDirectory(shardsDir);

        // META = the whole checkpoint with the big matrices EMPTIED (shapes kept) and the substrate dropped
        // (stored separately). Everything small (config, vocab, biases, spacing/casing, conversation, curriculum)
        // rides along as JSON.
        static MatrixSnapshot? Strip(MatrixSnapshot? m) => m is null ? null : m with { Values = Array.Empty<double>() };
        var meta = nn with
        {
            Embeddings = nn.Embeddings with { Values = Array.Empty<double>() },
            OutputWeights = nn.OutputWeights with { Values = Array.Empty<double>() },
            RouteWeights = Strip(nn.RouteWeights),
            EditWeights = Strip(nn.EditWeights),
            GruWih = Strip(nn.GruWih),
            GruWhh = Strip(nn.GruWhh),
            QueryOpWeights = Strip(nn.QueryOpWeights),
            QueryOperandWeights = Strip(nn.QueryOperandWeights),
            PlanWeights = Strip(nn.PlanWeights),
            TrunkWeights = Strip(nn.TrunkWeights),
            PlatonicSpace = null,
        };
        AddJson(sections, shardsDir, "meta", meta);

        // Big tensors as raw f64 shards (the only sections that get large). Small biases ride in meta JSON.
        AddF64(sections, shardsDir, "embeddings", nn.Embeddings);
        AddF64(sections, shardsDir, "outputWeights", nn.OutputWeights);
        AddF64(sections, shardsDir, "routeWeights", nn.RouteWeights);
        AddF64(sections, shardsDir, "editWeights", nn.EditWeights);
        AddF64(sections, shardsDir, "gruWih", nn.GruWih);
        AddF64(sections, shardsDir, "gruWhh", nn.GruWhh);
        AddF64(sections, shardsDir, "queryOpWeights", nn.QueryOpWeights);
        AddF64(sections, shardsDir, "queryOperandWeights", nn.QueryOperandWeights);
        AddF64(sections, shardsDir, "planWeights", nn.PlanWeights);
        AddF64(sections, shardsDir, "trunkWeights", nn.TrunkWeights);

        WriteManifest(modelDir, shardsDir, nn.Version, sections);
    }

    public static void WriteSubstrate(string substrateDir, PlatonicMemorySnapshot substrate)
    {
        var sections = new Dictionary<string, Section>(StringComparer.Ordinal);
        var shardsDir = Path.Combine(substrateDir, "shards");
        Directory.CreateDirectory(shardsDir);
        AddJson(sections, shardsDir, "platonic", substrate);
        WriteManifest(substrateDir, shardsDir, 0, sections);
    }

    // ── Read ──────────────────────────────────────────────────────────────────────────────────────────────
    public static GenesisCheckpoint ReadModel(string modelDir)
    {
        var (manifest, shardsDir) = ReadManifest(modelDir);
        var cp = ReadJson<GenesisCheckpoint>(manifest, shardsDir, "meta")
                 ?? throw new InvalidOperationException("checkpoint manifest is missing the 'meta' section.");

        cp = cp with { Embeddings = cp.Embeddings with { Values = ReadF64(manifest, shardsDir, "embeddings") } };
        cp = cp with { OutputWeights = cp.OutputWeights with { Values = ReadF64(manifest, shardsDir, "outputWeights") } };
        if (manifest.Sections.ContainsKey("routeWeights") && cp.RouteWeights is not null)
            cp = cp with { RouteWeights = cp.RouteWeights with { Values = ReadF64(manifest, shardsDir, "routeWeights") } };
        if (manifest.Sections.ContainsKey("editWeights") && cp.EditWeights is not null)
            cp = cp with { EditWeights = cp.EditWeights with { Values = ReadF64(manifest, shardsDir, "editWeights") } };
        if (manifest.Sections.ContainsKey("gruWih") && cp.GruWih is not null)
            cp = cp with { GruWih = cp.GruWih with { Values = ReadF64(manifest, shardsDir, "gruWih") } };
        if (manifest.Sections.ContainsKey("gruWhh") && cp.GruWhh is not null)
            cp = cp with { GruWhh = cp.GruWhh with { Values = ReadF64(manifest, shardsDir, "gruWhh") } };
        if (manifest.Sections.ContainsKey("queryOpWeights") && cp.QueryOpWeights is not null)
            cp = cp with { QueryOpWeights = cp.QueryOpWeights with { Values = ReadF64(manifest, shardsDir, "queryOpWeights") } };
        if (manifest.Sections.ContainsKey("queryOperandWeights") && cp.QueryOperandWeights is not null)
            cp = cp with { QueryOperandWeights = cp.QueryOperandWeights with { Values = ReadF64(manifest, shardsDir, "queryOperandWeights") } };
        if (manifest.Sections.ContainsKey("planWeights") && cp.PlanWeights is not null)
            cp = cp with { PlanWeights = cp.PlanWeights with { Values = ReadF64(manifest, shardsDir, "planWeights") } };
        if (manifest.Sections.ContainsKey("trunkWeights") && cp.TrunkWeights is not null)
            cp = cp with { TrunkWeights = cp.TrunkWeights with { Values = ReadF64(manifest, shardsDir, "trunkWeights") } };
        return cp;
    }

    public static PlatonicMemorySnapshot? ReadSubstrate(string substrateDir)
    {
        if (!ModelExists(substrateDir)) return null;
        var (manifest, shardsDir) = ReadManifest(substrateDir);
        return ReadJson<PlatonicMemorySnapshot>(manifest, shardsDir, "platonic");
    }

    /// <summary>Copy a sharded model dir (and its substrate dir, if any) to a destination — for last-good promotion.</summary>
    public static void CopyModel(string srcModelDir, string dstModelDir)
    {
        CopyDir(srcModelDir, dstModelDir);
        var srcSub = srcModelDir + ".platonic";
        if (Directory.Exists(srcSub)) CopyDir(srcSub, dstModelDir + ".platonic");
    }

    // ── Section helpers ───────────────────────────────────────────────────────────────────────────────────
    private sealed record Section(string Kind, long Length, int Rows, int Cols, List<string> Shards);
    private sealed record ManifestDoc(int FormatVersion, int ModelVersion, long ShardBytes, string CreatedUtc,
        Dictionary<string, Section> Sections);

    private static void AddF64(IDictionary<string, Section> sections, string shardsDir, string name, MatrixSnapshot? m)
    {
        if (m is null) return;
        var bytes = new byte[(long)m.Values.Length * sizeof(double)];
        Buffer.BlockCopy(m.Values, 0, bytes, 0, bytes.Length);
        sections[name] = new Section("f64", bytes.LongLength, m.Rows, m.Cols, ChunkAndWrite(shardsDir, bytes));
    }

    private static void AddJson<T>(IDictionary<string, Section> sections, string shardsDir, string name, T value)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(value, Json);
        sections[name] = new Section("json", bytes.LongLength, 0, 0, ChunkAndWrite(shardsDir, bytes));
    }

    private static double[] ReadF64(ManifestDoc manifest, string shardsDir, string name)
    {
        if (!manifest.Sections.TryGetValue(name, out var s)) return Array.Empty<double>();
        var bytes = ReadAndConcat(shardsDir, s.Shards, s.Length);
        var values = new double[bytes.Length / sizeof(double)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static T? ReadJson<T>(ManifestDoc manifest, string shardsDir, string name)
    {
        if (!manifest.Sections.TryGetValue(name, out var s)) return default;
        var bytes = ReadAndConcat(shardsDir, s.Shards, s.Length);
        return JsonSerializer.Deserialize<T>(bytes, Json);
    }

    // ── Shard I/O ─────────────────────────────────────────────────────────────────────────────────────────
    private static List<string> ChunkAndWrite(string shardsDir, byte[] data)
    {
        var hashes = new List<string>();
        for (long off = 0; off < data.LongLength || (data.LongLength == 0 && off == 0); off += TargetShardBytes)
        {
            var len = (int)Math.Min(TargetShardBytes, data.LongLength - off);
            if (len < 0) break;
            var chunk = new byte[len];
            Array.Copy(data, off, chunk, 0, len);
            var hash = Convert.ToHexString(SHA256.HashData(chunk)).ToLowerInvariant();
            hashes.Add(hash);
            var path = Path.Combine(shardsDir, hash + ".gnv");
            if (!File.Exists(path)) // content-addressed → identical chunks are written once
            {
                var tmp = $"{path}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";
                File.WriteAllBytes(tmp, chunk);
                File.Move(tmp, path, overwrite: true);
            }
            if (data.LongLength == 0) break;
        }
        return hashes;
    }

    private static byte[] ReadAndConcat(string shardsDir, IReadOnlyList<string> hashes, long totalLen)
    {
        var result = new byte[totalLen];
        long pos = 0;
        foreach (var h in hashes)
        {
            var chunk = File.ReadAllBytes(Path.Combine(shardsDir, h + ".gnv"));
            Array.Copy(chunk, 0, result, pos, chunk.Length);
            pos += chunk.Length;
        }
        return result;
    }

    // ── Manifest I/O + prune ──────────────────────────────────────────────────────────────────────────────
    private static void WriteManifest(string dir, string shardsDir, int modelVersion, Dictionary<string, Section> sections)
    {
        var manifest = new ManifestDoc(FormatVersion, modelVersion, TargetShardBytes,
            DateTimeOffset.UtcNow.ToString("O"), sections);
        var tmp = Path.Combine(dir, $"manifest.json.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp");
        File.WriteAllText(tmp, JsonSerializer.Serialize(manifest, Json));
        File.Move(tmp, Path.Combine(dir, "manifest.json"), overwrite: true);

        // Prune shards no longer referenced (stale from a previous save) so the dir only holds what's relevant.
        var referenced = new HashSet<string>(sections.Values.SelectMany(s => s.Shards), StringComparer.Ordinal);
        foreach (var f in Directory.EnumerateFiles(shardsDir, "*.gnv"))
        {
            var name = Path.GetFileNameWithoutExtension(f);
            if (!referenced.Contains(name))
                try { File.Delete(f); } catch { /* best-effort */ }
        }
    }

    private static (ManifestDoc Manifest, string ShardsDir) ReadManifest(string dir)
    {
        var manifest = JsonSerializer.Deserialize<ManifestDoc>(File.ReadAllText(Path.Combine(dir, "manifest.json")), Json)
            ?? throw new InvalidOperationException($"Failed to read manifest in {dir}");
        return (manifest, Path.Combine(dir, "shards"));
    }

    private static void CopyDir(string src, string dst)
    {
        Directory.CreateDirectory(dst);
        Directory.CreateDirectory(Path.Combine(dst, "shards"));
        File.Copy(Path.Combine(src, "manifest.json"), Path.Combine(dst, "manifest.json"), overwrite: true);
        var srcShards = Path.Combine(src, "shards");
        if (Directory.Exists(srcShards))
            foreach (var f in Directory.EnumerateFiles(srcShards, "*.gnv"))
                File.Copy(f, Path.Combine(dst, "shards", Path.GetFileName(f)), overwrite: true);
    }
}
