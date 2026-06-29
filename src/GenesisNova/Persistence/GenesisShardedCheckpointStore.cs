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
    public static void WriteModel(string modelDir, GenesisCheckpoint nn, string? generation = null)
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
            // Navigator: keep the param NAMES + SHAPES (and arch dims) in meta, but EMPTY the values — the bytes go to
            // the concatenated "navigator" f32 shard below (the big tensors, like the model's, don't bloat the meta JSON).
            Navigator = nn.Navigator is null
                ? null
                : nn.Navigator with { Parameters = nn.Navigator.Parameters.Select(p => p with { Values = Array.Empty<float>() }).ToArray() },
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

        // NAVIGATOR policy-net weights: concatenate every param tensor (row-major, in the meta's param order) into ONE
        // native-f32 shard (half the bytes of the old f64 encoding, lossless for f32 params — halves autosave I/O).
        // Optional — pre-navigator checkpoints (Navigator null) write nothing here, so they stay loadable.
        if (nn.Navigator is { Parameters.Length: > 0 } nav)
        {
            var total = nav.Parameters.Sum(p => (long)p.Values.Length);
            var blob = new float[total];
            long off = 0;
            foreach (var p in nav.Parameters) { Array.Copy(p.Values, 0L, blob, off, p.Values.Length); off += p.Values.Length; }
            AddF32Raw(sections, shardsDir, "navigator", blob);
        }

        WriteManifest(modelDir, shardsDir, nn.Version, sections, generation);
    }

    public static void WriteSubstrate(string substrateDir, PlatonicMemorySnapshot substrate, string? generation = null)
    {
        var sections = new Dictionary<string, Section>(StringComparer.Ordinal);
        var shardsDir = Path.Combine(substrateDir, "shards");
        Directory.CreateDirectory(shardsDir);
        AddJson(sections, shardsDir, "platonic", substrate);
        WriteManifest(substrateDir, shardsDir, 0, sections, generation);
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

        // NAVIGATOR weights: split the one concatenated native-f32 shard back into the per-param tensors by their (meta)
        // shapes. Absent for pre-navigator checkpoints (cp.Navigator null / no "navigator" section) → null (fresh net).
        if (cp.Navigator is { Parameters.Length: > 0 } nav && manifest.Sections.ContainsKey("navigator"))
        {
            var blob = ReadF32(manifest, shardsDir, "navigator");
            var restored = new NavParameterSnapshot[nav.Parameters.Length];
            long off = 0;
            for (var i = 0; i < nav.Parameters.Length; i++)
            {
                var p = nav.Parameters[i];
                var len = p.Shape.Aggregate(1L, (a, b) => a * b);
                var vals = new float[len];
                if (off + len <= blob.LongLength) Array.Copy(blob, off, vals, 0L, len);
                off += len;
                restored[i] = p with { Values = vals };
            }
            cp = cp with { Navigator = nav with { Parameters = restored } };
        }
        return cp;
    }

    public static PlatonicMemorySnapshot? ReadSubstrate(string substrateDir)
    {
        if (!ModelExists(substrateDir)) return null;
        var (manifest, shardsDir) = ReadManifest(substrateDir);
        return ReadJson<PlatonicMemorySnapshot>(manifest, shardsDir, "platonic");
    }

    // ── Crash-atomicity: per-save generation stamp ───────────────────────────────────────────────────────────
    /// <summary>The pointer-marker JSON, carrying the save <paramref name="generation"/> (the commit token).</summary>
    public static string PointerJson(string? generation)
        => JsonSerializer.Serialize(new ShardedPointer(true, FormatVersion, generation), Json);

    /// <summary>The generation stamped in a sharded dir's manifest (null if absent / unreadable / legacy).</summary>
    public static string? ReadGeneration(string dir)
    {
        try
        {
            var file = Path.Combine(dir, "manifest.json");
            if (!File.Exists(file)) return null;
            return JsonSerializer.Deserialize<ManifestDoc>(File.ReadAllText(file), Json)?.Generation;
        }
        catch { return null; }
    }

    /// <summary>The generation recorded in the pointer marker at <paramref name="pointerPath"/> (null if absent /
    /// unreadable / legacy pointer without a generation).</summary>
    public static string? ReadPointerGeneration(string pointerPath)
    {
        try
        {
            if (!File.Exists(pointerPath)) return null;
            return JsonSerializer.Deserialize<ShardedPointer>(File.ReadAllText(pointerPath), Json)?.Generation;
        }
        catch { return null; }
    }

    /// <summary>True if the checkpoint at <paramref name="pointerPath"/> is a CONSISTENT save: the pointer's
    /// generation matches the model manifest's AND (if a substrate dir exists) the substrate manifest's. A torn save
    /// (crash between writing the two dirs) leaves them at different generations → false. A LEGACY checkpoint (no
    /// generation anywhere) returns true — we can't verify it, so we don't block resuming it.</summary>
    public static bool IsConsistent(string pointerPath)
    {
        var pointerGen = ReadPointerGeneration(pointerPath);
        if (pointerGen is null) return true; // legacy / pre-generation checkpoint — cannot verify, don't block
        if (!string.Equals(ReadGeneration(ModelDir(pointerPath)), pointerGen, StringComparison.Ordinal)) return false;
        var substrateDir = SubstrateDir(pointerPath);
        if (ModelExists(substrateDir) && !string.Equals(ReadGeneration(substrateDir), pointerGen, StringComparison.Ordinal)) return false;
        return true;
    }

    /// <summary>Copy a sharded model dir (and its substrate dir, if any) to a destination — for last-good promotion.</summary>
    public static void CopyModel(string srcModelDir, string dstModelDir)
    {
        CopyDir(srcModelDir, dstModelDir);
        var srcSub = srcModelDir + ".platonic";
        if (Directory.Exists(srcSub)) CopyDir(srcSub, dstModelDir + ".platonic");
    }

    /// <summary>PROMOTE a local checkpoint to a STARTER location (e.g. the repo's <c>models/genesis-nova</c>) so it can
    /// be committed and SEED fresh starts. Copies the local pointer + model + substrate, mirroring the destination.
    /// Both args are POINTER paths (the <c>.json</c> marker); the model/substrate dirs derive from them.</summary>
    public static void PromoteToStarter(string localPointerPath, string starterPointerPath)
    {
        if (!File.Exists(localPointerPath)) throw new FileNotFoundException("no local checkpoint to promote", localPointerPath);
        Directory.CreateDirectory(Path.GetDirectoryName(starterPointerPath)!);
        CopyModel(ModelDir(localPointerPath), ModelDir(starterPointerPath)); // model + .platonic, mirror dst
        File.Copy(localPointerPath, starterPointerPath, overwrite: true);    // pointer LAST (the commit marker)
    }

    /// <summary>SEED a fresh local checkpoint from a committed STARTER (the reverse of <see cref="PromoteToStarter"/>):
    /// if NO local checkpoint exists yet and the starter is present + CONSISTENT, copy the starter's model + substrate +
    /// pointer into the local location so a clone / fresh machine begins from the shared warmed model instead of an
    /// empty brain. NO-OP once a local checkpoint exists (the local fork then evolves on its own) or if the starter is
    /// absent / torn. Returns true iff it seeded. Both args are POINTER paths.</summary>
    public static bool SeedFromStarter(string starterPointerPath, string localPointerPath)
    {
        if (File.Exists(localPointerPath)) return false;                   // local fork already exists — never overwrite
        if (!File.Exists(starterPointerPath)) return false;               // no starter committed
        if (!ModelExists(ModelDir(starterPointerPath))) return false;     // pointer without a model dir
        if (!IsConsistent(starterPointerPath)) return false;              // don't seed a torn starter
        Directory.CreateDirectory(Path.GetDirectoryName(localPointerPath)!);
        CopyModel(ModelDir(starterPointerPath), ModelDir(localPointerPath)); // model + .platonic
        File.Copy(starterPointerPath, localPointerPath, overwrite: true);    // pointer LAST (the commit marker)
        return true;
    }

    // ── Section helpers ───────────────────────────────────────────────────────────────────────────────────
    private sealed record Section(string Kind, long Length, int Rows, int Cols, List<string> Shards);
    // Generation = a per-SAVE id stamped on BOTH the model and substrate manifests AND the pointer marker, so a load
    // can verify they all came from the SAME save. A crash between writing the two dirs leaves them at DIFFERENT
    // generations → detectable → fall back to last-good instead of resuming an inconsistent (torn) brain.
    private sealed record ManifestDoc(int FormatVersion, int ModelVersion, long ShardBytes, string CreatedUtc,
        Dictionary<string, Section> Sections, string? Generation = null);

    // The pointer marker (the legacy ".json" path) — the COMMIT of a save, written LAST and atomically. It carries
    // the save Generation so a torn save (dirs written, commit's generation not matching) is detectable.
    private sealed record ShardedPointer(bool GnvSharded, int FormatVersion, string? Generation);

    private static void AddF64(IDictionary<string, Section> sections, string shardsDir, string name, MatrixSnapshot? m)
    {
        if (m is null) return;
        var bytes = new byte[(long)m.Values.Length * sizeof(double)];
        Buffer.BlockCopy(m.Values, 0, bytes, 0, bytes.Length);
        sections[name] = new Section("f64", bytes.LongLength, m.Rows, m.Cols, ChunkAndWrite(shardsDir, bytes));
    }

    // A raw f64 vector section (no Rows/Cols semantics — used for the concatenated navigator weight blob).
    private static void AddF64Raw(IDictionary<string, Section> sections, string shardsDir, string name, double[] values)
    {
        var bytes = new byte[(long)values.Length * sizeof(double)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        sections[name] = new Section("f64", bytes.LongLength, values.Length, 1, ChunkAndWrite(shardsDir, bytes));
    }

    // A raw NATIVE-f32 vector section — used for the concatenated navigator weight blob (half the bytes of f64, lossless
    // for the f32 policy-net params, so autosave I/O is halved).
    private static void AddF32Raw(IDictionary<string, Section> sections, string shardsDir, string name, float[] values)
    {
        var bytes = new byte[(long)values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        sections[name] = new Section("f32", bytes.LongLength, values.Length, 1, ChunkAndWrite(shardsDir, bytes));
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

    // Read a native-f32 raw vector section (the navigator weight blob). A pre-f32 checkpoint that stored this section as
    // f64 is handled transparently: if the byte length is a multiple of 8 (and the f32 view would be even-padded) we fall
    // back to decoding f64 then narrowing, so an OLD navigator shard still loads. New saves are always f32.
    private static float[] ReadF32(ManifestDoc manifest, string shardsDir, string name)
    {
        if (!manifest.Sections.TryGetValue(name, out var s)) return Array.Empty<float>();
        var bytes = ReadAndConcat(shardsDir, s.Shards, s.Length);
        if (string.Equals(s.Kind, "f64", StringComparison.Ordinal))
        {
            // Legacy navigator shard written as f64 — decode wide then narrow to f32 (the params are f32 anyway).
            var wide = new double[bytes.Length / sizeof(double)];
            Buffer.BlockCopy(bytes, 0, wide, 0, bytes.Length);
            var narrowed = new float[wide.Length];
            for (var i = 0; i < wide.Length; i++) narrowed[i] = (float)wide[i];
            return narrowed;
        }
        var values = new float[bytes.Length / sizeof(float)];
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
    private static void WriteManifest(string dir, string shardsDir, int modelVersion, Dictionary<string, Section> sections, string? generation = null)
    {
        var manifest = new ManifestDoc(FormatVersion, modelVersion, TargetShardBytes,
            DateTimeOffset.UtcNow.ToString("O"), sections, generation);
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
        var dstShards = Path.Combine(dst, "shards");
        Directory.CreateDirectory(dstShards);
        File.Copy(Path.Combine(src, "manifest.json"), Path.Combine(dst, "manifest.json"), overwrite: true);
        var srcShards = Path.Combine(src, "shards");
        var copied = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (Directory.Exists(srcShards))
            foreach (var f in Directory.EnumerateFiles(srcShards, "*.gnv"))
            {
                var name = Path.GetFileName(f);
                File.Copy(f, Path.Combine(dstShards, name), overwrite: true);
                copied.Add(name);
            }

        // Prune stale shards left in the DESTINATION from a previous copy — the destination must MIRROR the
        // source (the copied shards are exactly the referenced set). Without this, lastgood accumulated every
        // shard ever promoted and grew without bound (the Save path already prunes in WriteManifest; this copy
        // path did not — the lastgood dir reached 100s of GB / ~13k files).
        foreach (var f in Directory.EnumerateFiles(dstShards, "*.gnv"))
            if (!copied.Contains(Path.GetFileName(f)))
                try { File.Delete(f); } catch { /* best-effort */ }
    }
}
