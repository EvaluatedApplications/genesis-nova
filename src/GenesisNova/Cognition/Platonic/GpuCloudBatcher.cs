using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// BATCHED GPU recompute of the distributional cloud (SYSTEM_OVERVIEW.md — "statistics on the GPU"). The per-observation
/// scalar <c>RecomputeCloud</c> is the substrate's hot loop and its bottleneck (measured 638 obs/s at dim=512): it sums an
/// endpoint's whole neighbourhood (O(degree)·semLen scalar mul-adds) on the CPU, on EVERY observation that touches it. This
/// replaces it with DEFER + BATCH:
/// <list type="bullet">
/// <item>DEDUP — a hub touched N times between flushes is recomputed ONCE, not N times (an algorithmic win, not a constant
/// factor: scalar does N·degree work, batched does degree work).</item>
/// <item>GPU — the 310-dim multiply-accumulate moves from scalar C# onto CUDA as one batched op.</item>
/// </list>
///
/// FORMULATION (sparse-free, robust in TorchSharp): the token matrix <c>T = [V, semLen]</c> lives on the device, APPEND-ONLY
/// (tokens are deterministic &amp; immutable — rows never change, only grow with vocab). A flush over D dirty rows, each with
/// entries (column c, weight w) including its self-token (c = col(self), w = 1), builds flat COO arrays and computes
/// <code>
///   contrib = T.index_select(0, colIdx) * weight[:, None]     // [E, semLen]
///   result  = zeros(D, semLen).index_add_(0, rowIdx, contrib) // scatter-add the contributions per dirty row
/// </code>
/// then copies <c>result</c> back so each dirty element's SemanticFace gets its (raw, unnormalized) cloud — identical to the
/// scalar definition up to float32 rounding (validated by cosine in CloudRecomputeBenchmark). Falls back to CPU tensors when
/// CUDA is absent (still gets the dedup + vectorized win). Not thread-safe; driven from the single substrate loop.
/// </summary>
internal sealed class GpuCloudBatcher : IDisposable
{
    private readonly int _semLen;
    private readonly Device _device;
    private readonly Dictionary<string, int> _col = new(StringComparer.Ordinal); // symbol → stable column id in T
    private readonly List<float[]> _pending = new();                              // token rows awaiting append to T
    private Tensor? _t;                                                           // [V, semLen] token matrix on device

    public GpuCloudBatcher(int semLen, bool preferCuda)
    {
        _semLen = semLen;
        _device = preferCuda && cuda_is_available() ? new Device(DeviceType.CUDA, 0) : new Device(DeviceType.CPU);
    }

    public bool OnCuda => _device.type == DeviceType.CUDA;

    private int Column(string symbol, Func<string, double[]> tokenOf)
    {
        if (_col.TryGetValue(symbol, out var c)) return c;
        c = _col.Count;
        _col[symbol] = c;
        var src = tokenOf(symbol);
        var row = new float[_semLen];
        for (var i = 0; i < _semLen && i < src.Length; i++) row[i] = (float)src[i];
        _pending.Add(row);
        return c;
    }

    // Append any newly-assigned token rows to the device matrix T (tokens are immutable, so this only ever grows T).
    private void EnsureTokenMatrix()
    {
        if (_pending.Count == 0) return;
        var p = _pending.Count;
        var flat = new float[p * _semLen];
        for (var i = 0; i < p; i++) Array.Copy(_pending[i], 0, flat, i * _semLen, _semLen);
        _pending.Clear();
        using var add = tensor(flat, dtype: ScalarType.Float32).reshape(p, _semLen).to(_device);
        if (_t is null) { _t = add.clone(); }
        else { var merged = cat(new[] { _t, add }, 0); _t.Dispose(); _t = merged; }
    }

    /// <summary>Recompute the cloud of every <paramref name="selves"/> entry in one batched op. For each self, the caller
    /// supplies its neighbour entries (symbol + affinity weight); the self-token (weight 1) is added here. <paramref
    /// name="writeRow"/> receives (rowIndex, rawCloud) — the buffer is REUSED, so copy out synchronously.</summary>
    public void Flush(
        IReadOnlyList<string> selves,
        Func<string, double[]> tokenOf,
        Func<string, IReadOnlyList<(string nbr, double weight)>> entriesOf,
        Action<int, float[]> writeRow)
    {
        if (selves.Count == 0) return;

        var rows = new List<long>();
        var cols = new List<long>();
        var wts = new List<float>();
        for (var r = 0; r < selves.Count; r++)
        {
            var self = selves[r];
            rows.Add(r); cols.Add(Column(self, tokenOf)); wts.Add(1f);          // self-token, weight 1
            foreach (var (nbr, w) in entriesOf(self))
            {
                if (Math.Abs(w) < 1e-9) continue;                                // skip ~zero affinity (bit-identical: 0·t = 0)
                rows.Add(r); cols.Add(Column(nbr, tokenOf)); wts.Add((float)w);
            }
        }
        EnsureTokenMatrix();

        using var rowIdx = tensor(rows.ToArray(), dtype: ScalarType.Int64, device: _device);
        using var colIdx = tensor(cols.ToArray(), dtype: ScalarType.Int64, device: _device);
        using var weight = tensor(wts.ToArray(), dtype: ScalarType.Float32, device: _device).unsqueeze(1);
        using var gathered = _t!.index_select(0, colIdx);                        // [E, semLen]
        using var contrib = gathered.mul(weight);                                // [E, semLen]
        using var result = zeros(new long[] { selves.Count, _semLen }, dtype: ScalarType.Float32, device: _device);
        result.index_add_(0, rowIdx, contrib, 1);                                // scatter-add per dirty row (alpha = 1)

        using var cpu = result.cpu();
        var data = cpu.data<float>();
        var buf = new float[_semLen];
        for (var r = 0; r < selves.Count; r++)
        {
            var off = (long)r * _semLen;
            for (var i = 0; i < _semLen; i++) buf[i] = data[off + i];
            writeRow(r, buf);
        }
    }

    public void Dispose() => _t?.Dispose();
}
