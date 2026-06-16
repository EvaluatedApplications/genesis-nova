using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// A keyed frequency store of TEXT CHUNKS mined from graded-correct outputs, grouped by a tag (the shape they
/// scaffold). The Seq composer binds the most-reinforced chunk to a substrate-computed value — the cache/
/// binding idea done element-natively (the scaffold is LEARNED from positive results, not a baked-in template).
/// Extracted from <see cref="PlatonicSpaceMemory"/> (single-responsibility); kept out of the concept store
/// (frequency, not geometry) so it never perturbs concept retrieval.
/// </summary>
internal sealed class ChunkElementStore
{
    private readonly Dictionary<string, Dictionary<string, int>> _store =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Record one observation of <paramref name="chunk"/> as a scaffold for <paramref name="tag"/>.</summary>
    public void Mine(string tag, string chunk)
    {
        if (string.IsNullOrWhiteSpace(tag) || string.IsNullOrWhiteSpace(chunk))
            return;
        var t = tag.Trim().ToLowerInvariant();
        var c = chunk.Trim();
        if (!_store.TryGetValue(t, out var counts))
        {
            counts = new Dictionary<string, int>(StringComparer.Ordinal);
            _store[t] = counts;
        }
        counts[c] = counts.TryGetValue(c, out var n) ? n + 1 : 1;
    }

    /// <summary>The most-reinforced chunk mined for <paramref name="tag"/> (false if none yet).</summary>
    public bool TryGetTop(string tag, out string chunk)
    {
        chunk = string.Empty;
        if (!_store.TryGetValue(tag.Trim().ToLowerInvariant(), out var counts) || counts.Count == 0)
            return false;
        chunk = counts
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .First().Key;
        return true;
    }

    /// <summary>Project to snapshot rows for persistence.</summary>
    public PlatonicChunkSnapshot[] Export()
        => _store
            .SelectMany(tag => tag.Value.Select(c => new PlatonicChunkSnapshot(tag.Key, c.Key, c.Value)))
            .ToArray();

    /// <summary>
    /// Restore ADDITIVELY (counts merged, not replaced): ImportSnapshot is also the rebuild step for
    /// ApplyMaintenance, whose pruned snapshot carries no chunks — merging means a maintenance pass never
    /// wipes the mined scaffolds, while a checkpoint load (fresh store) restores them exactly.
    /// </summary>
    public void ImportMerge(IReadOnlyList<PlatonicChunkSnapshot>? chunks)
    {
        if (chunks is null)
            return;
        foreach (var chunk in chunks)
        {
            if (string.IsNullOrWhiteSpace(chunk.Tag) || string.IsNullOrWhiteSpace(chunk.Chunk))
                continue;
            var t = chunk.Tag.Trim().ToLowerInvariant();
            if (!_store.TryGetValue(t, out var counts))
            {
                counts = new Dictionary<string, int>(StringComparer.Ordinal);
                _store[t] = counts;
            }
            counts[chunk.Chunk] = (counts.TryGetValue(chunk.Chunk, out var n) ? n : 0) + Math.Max(1, chunk.Count);
        }
    }
}
