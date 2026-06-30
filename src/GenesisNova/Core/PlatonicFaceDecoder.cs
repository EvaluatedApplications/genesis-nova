using System.Globalization;
using System.Text;

namespace GenesisNova.Core;

/// <summary>
/// Pure inverse of <see cref="PlatonicFaceComposer"/>'s face encoders — ported faithfully from the
/// genesis-engine source of truth (Genesis.Shared.KnnDecoder + Genesis.Shared.NumericCodec) and
/// adapted to nova's <see cref="FaceLayout"/>.
/// <para>
/// Turns a predicted embedding back into a number or string by reading the SAME faces the composer
/// wrote. Every constant here is derived identically to the matching encode method so decode is the
/// exact algebraic inverse:
/// <list type="bullet">
///   <item><see cref="DecodeNumericFromPrediction"/> ↔ <see cref="PlatonicFaceComposer.GetFreshNumericEmbedding"/></item>
/// </list>
/// </para>
/// <para>
/// No torch, no state, no allocation beyond per-call scratch — every method is a pure function of its
/// arguments (and the deterministic atom generators in <see cref="PlatonicFaceComposer"/>).
/// </para>
/// </summary>
internal static class PlatonicFaceDecoder
{
    /// <summary>
    /// Character vocabulary for slot decoding — must match the source-of-truth
    /// (Genesis.Shared.EmbeddingGenerator.BuildCharVocab) exactly: lower-case a-z, upper-case A-Z,
    /// digits 0-9, then the punctuation block "!@#$%^&amp;*()_+-=[]{}|;':\",./&lt;&gt;? " (trailing space).
    /// </summary>
    public static readonly char[] CharVocab = BuildCharVocab();

    private static char[] BuildCharVocab()
    {
        var chars = new List<char>();
        for (var c = 'a'; c <= 'z'; c++) chars.Add(c);
        for (var c = 'A'; c <= 'Z'; c++) chars.Add(c);
        for (var c = '0'; c <= '9'; c++) chars.Add(c);
        chars.AddRange("!@#$%^&*()_+-=[]{}|;':\",./<>? ");
        return chars.ToArray();
    }

    /// <summary>
    /// Which arithmetic face an inline numeric decode is evaluating. Mirrors the source's
    /// Genesis.Shared.DecodeFace enum.
    /// </summary>
    private enum DecodeFace
    {
        Polynomial,
        Logarithmic
    }

    /// <summary>
    /// Decode a numeric value from a predicted embedding by inspecting BOTH the polynomial and
    /// logarithmic faces and selecting the one with better self-consistency — the inverse of
    /// <see cref="PlatonicFaceComposer.GetFreshNumericEmbedding"/>.
    /// <para>
    /// Polynomial decode: value = predicted[0] × 10.
    /// Logarithmic decode: value = exp(predicted[logStart] × 10), where logStart = numericDims.
    /// </para>
    /// <para>
    /// <paramref name="preferFace"/> hint convention (int):
    /// <list type="bullet">
    ///   <item>0 = auto — pick the higher-quality face (default)</item>
    ///   <item>1 = poly — prefer the polynomial face when its quality &gt; 0.3</item>
    ///   <item>2 = log — prefer the logarithmic face when its quality &gt; 0.3</item>
    /// </list>
    /// Any other value is treated as 0 (auto).
    /// </para>
    /// </summary>
    /// <returns>
    /// Decoded value, quality score [0..1], and the selected face as "poly", "log", or "none"
    /// (none = no numeric signal — both faces decode to ~zero with no quality).
    /// </returns>
    public static (double Value, double Quality, string Face) DecodeNumericFromPrediction(
        double[] predicted, int dim, int preferFace = 0)
    {
        if (predicted is null || dim <= 0 || predicted.Length == 0)
            return (0.0, 0.0, "none");

        var numericDims = FaceLayout.NumericDims(dim);
        var logStart = numericDims;

        // Polynomial decode: value = predicted[0] * 10
        var polyValue = predicted[0] * 10.0;
        var polyQuality = ComputeNumericDecodeQualityInline(
            predicted, polyValue, dim, DecodeFace.Polynomial, numericDims, logStart,
            isPreferred: preferFace == 1);

        // Logarithmic decode: value = exp(predicted[logStart] * 10).
        // Log face always decodes positive (exp > 0); sign lives in the poly face.
        // For mul/div (preferFace == 2) the log face is the homomorphic one and MUST be decoded even
        // when its leading slot is ~0 — that means the result is ~1 (exp(0)=1), e.g. 5/5 or 1*1. The
        // >1e-12 guard only applies to the auto/poly path, where an all-zero log face means "no log
        // signal" rather than "the result is 1".
        var attemptLog = logStart < dim && (preferFace == 2 || Math.Abs(predicted[logStart]) > 1e-12);
        var logValue = attemptLog
            ? Math.Exp(predicted[logStart] * 10.0)
            : double.NaN;
        var logQuality = !double.IsNaN(logValue) && !double.IsInfinity(logValue)
            ? ComputeNumericDecodeQualityInline(
                predicted, logValue, dim, DecodeFace.Logarithmic, numericDims, logStart,
                isPreferred: preferFace == 2)
            : 0.0;

        var quality = Math.Max(polyQuality, logQuality);

        // Face selection: when a preferred face is hinted, use it if it has reasonable quality
        // (> 0.3). This prevents the non-homomorphic face from winning — e.g. for mul(3,2):
        // poly decodes to 5 (sum), log decodes to 6 (product); both self-consistent, only log correct.
        double bestValue;
        string bestFace;
        if (preferFace == 2 && logQuality > 0.3)
        {
            bestValue = logValue;
            bestFace = "log";
        }
        else if (preferFace == 1 && polyQuality > 0.3)
        {
            bestValue = polyValue;
            bestFace = "poly";
        }
        else if (polyQuality >= logQuality)
        {
            bestValue = polyValue;
            bestFace = "poly";
        }
        else
        {
            bestValue = logValue;
            bestFace = "log";
        }

        // No numeric signal at all → report "none" so callers can decline to emit a number.
        if (quality <= 0.0 && Math.Abs(polyValue) < 1e-12 && (double.IsNaN(logValue) || double.IsInfinity(logValue)))
            return (0.0, 0.0, "none");

        return (bestValue, quality, bestFace);
    }

    /// <summary>
    /// Computes numeric decode quality inline without allocating a re-embed array — ported from the
    /// source's NumericCodec.ComputeNumericDecodeQualityInline.
    /// <para>
    /// Quality = clamp(1 − √residual, 0, 1), residual = Σ (predicted[i] − expected[i])² over the
    /// first min(numericDims, 5) face dims. For poly expected = value × 10^-(i+1); for log
    /// expected = ln|value| × 10^-(i+1).
    /// </para>
    /// </summary>
    private static double ComputeNumericDecodeQualityInline(
        double[] predicted, double value, int dim, DecodeFace face, int numericDims, int logStart,
        bool isPreferred = false)
    {
        var faceStart = face == DecodeFace.Logarithmic ? logStart : 0;
        var faceDims = Math.Min(numericDims, 5);

        // Require minimum signal to distinguish real numeric values from text noise. Skip the guard
        // when the caller has confirmed this IS an arithmetic result (isPreferred) — zero signal then
        // means zero value, not noise (e.g. add(3,-3)=0 has exactly zero poly signal).
        var signalNorm = 0.0;
        for (var i = 0; i < faceDims && faceStart + i < dim; i++)
            signalNorm += predicted[faceStart + i] * predicted[faceStart + i];
        if (!isPreferred && signalNorm < 4e-4)
            return 0.0;

        var residual = 0.0;
        for (var i = 0; i < faceDims && faceStart + i < dim; i++)
        {
            var expected = face == DecodeFace.Logarithmic && Math.Abs(value) > 1e-12
                ? Math.Log(Math.Abs(value)) * Math.Pow(10.0, -(i + 1))
                : value * Math.Pow(10.0, -(i + 1));
            var diff = predicted[faceStart + i] - expected;
            residual += diff * diff;
        }
        return Math.Clamp(1.0 - Math.Sqrt(residual), 0.0, 1.0);
    }

    // ============================================================================================
    // ADDRESS-SPACE DECODERS (active when dim ≥ 512) — exact inverses of the encoders in
    // PlatonicFaceComposer. Each regenerates candidate codes from the same deterministic generators.
    // ============================================================================================

    /// <summary>
    /// The missing spelling inverse: decode the token from the spelling band [48,208) by reading each
    /// char-slot back to its nearest char in <see cref="CharVocab"/>, stopping at the first empty slot.
    /// Exact inverse of <see cref="PlatonicFaceComposer.GetCharComposedEmbedding"/> at address-space dims
    /// (e.g. encode("cat",512) then CharSlotDecode → "cat"). Empty string below address-space dims.
    /// </summary>
    public static string CharSlotDecode(double[] face, int dim)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim))
            return string.Empty;
        var sb = new StringBuilder();
        for (var i = 0; i < FaceLayout.SpellingSlots; i++)
        {
            var slotStart = FaceLayout.SpellingStart + (i * FaceLayout.SpellingSlotDims);
            if (RangeNorm(face, slotStart, slotStart + FaceLayout.SpellingSlotDims) < 1e-6)
                break; // first empty slot ends the token
            sb.Append(NearestChar(face, slotStart));
        }
        return sb.ToString();
    }

    /// <summary>Nearest char in <see cref="CharVocab"/> for the 10-dim slot at <paramref name="slotStart"/>.</summary>
    private static char NearestChar(double[] face, int slotStart)
    {
        var best = '\0';
        var bestDist = double.MaxValue;
        foreach (var c in CharVocab)
        {
            var atom = PlatonicFaceComposer.SpellingCharAtom(c);
            var dist = 0.0;
            for (var k = 0; k < FaceLayout.SpellingSlotDims && slotStart + k < face.Length; k++)
            {
                var diff = face[slotStart + k] - atom[k];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = c; }
        }
        return best;
    }

    /// <summary>
    /// Decode the kind band [42,48): the one-hot dim with the largest magnitude, or
    /// <see cref="PlatonicKind.None"/> if nothing rises above 0.25 (the all-zero "number" code).
    /// Inverse of <see cref="PlatonicFaceComposer.EncodeKind"/>.
    /// </summary>
    public static PlatonicKind DecodeKind(double[] face, int dim)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim))
            return PlatonicKind.None;
        var bestIdx = -1;
        var bestVal = 0.0;
        for (var k = 0; k < FaceLayout.KindDims && FaceLayout.KindStart + k < face.Length; k++)
        {
            var v = Math.Abs(face[FaceLayout.KindStart + k]);
            if (v > bestVal) { bestVal = v; bestIdx = k; }
        }
        if (bestIdx < 0 || bestVal < 0.25)
            return PlatonicKind.None;
        return (PlatonicKind)(bestIdx + 1);
    }

    /// <summary>
    /// Decode the op band [400,416) to the nearest operation token in <paramref name="opVocab"/>.
    /// Empty string when the band is ~zero (no op) or the vocab is empty. Inverse of
    /// <see cref="PlatonicFaceComposer.EncodeOp"/>.
    /// </summary>
    public static string DecodeOp(double[] face, int dim, IReadOnlyCollection<string> opVocab)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim) || opVocab is null || opVocab.Count == 0)
            return string.Empty;
        if (RangeNorm(face, FaceLayout.OpStart, FaceLayout.OpStart + FaceLayout.OpDims) < 1e-6)
            return string.Empty;
        var best = string.Empty;
        var bestDist = double.MaxValue;
        foreach (var op in opVocab)
        {
            if (string.IsNullOrEmpty(op)) continue;
            var code = PlatonicFaceComposer.OpCode(op);
            var dist = 0.0;
            for (var k = 0; k < FaceLayout.OpDims && FaceLayout.OpStart + k < face.Length; k++)
            {
                var diff = face[FaceLayout.OpStart + k] - code[k];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = op; }
        }
        return best;
    }

    /// <summary>Decoded structure: a shared role/label (when a vocab is supplied) and the ordered children.</summary>
    public readonly record struct StructureDecode(string Label, IReadOnlyList<string> Children);

    /// <summary>
    /// Decode the structure band [208,400): ordered child digests + a role/label. Per slot, a non-zero
    /// numeric head decodes a child VALUE exactly (v = digest[0]·10); otherwise the first 2 spelling
    /// digest-slots decode the child's leading chars. Decoding stops at the first empty slot. The label
    /// is resolved (nearest in <paramref name="labelVocab"/>) only when a vocab is supplied.
    /// <b>Boundary:</b> children longer than 2 chars resolve only to this digest here — the substrate
    /// layer disambiguates them against realised coordinates. Inverse of
    /// <see cref="PlatonicFaceComposer.EncodeStructure"/>.
    /// </summary>
    public static StructureDecode DecodeStructure(
        double[] face, int dim, IReadOnlyCollection<string>? labelVocab = null)
    {
        var children = new List<string>();
        var label = string.Empty;
        if (face is null || !FaceLayout.IsAddressSpace(dim))
            return new StructureDecode(label, children);

        for (var i = 0; i < FaceLayout.StructureSlots; i++)
        {
            var slotStart = FaceLayout.StructureStart + (i * FaceLayout.StructureSlotDims);
            if (RangeNorm(face, slotStart, slotStart + FaceLayout.StructureChildDigestDims) < 1e-9)
                break; // first empty child slot ends the list

            string child;
            var polyHead = slotStart < face.Length ? face[slotStart] : 0.0;
            if (Math.Abs(polyHead) > 1e-12)
                child = (polyHead * 10.0).ToString("0.####", CultureInfo.InvariantCulture);
            else
                child = DecodeDigestSpelling(face, slotStart + 4);
            children.Add(child);

            if (i == 0 && labelVocab is { Count: > 0 })
                label = NearestLabel(face, slotStart + FaceLayout.StructureChildDigestDims, labelVocab);
        }
        return new StructureDecode(label, children);
    }

    /// <summary>Decode up to 2 spelling digest-slots (20 dims) starting at <paramref name="start"/>.</summary>
    private static string DecodeDigestSpelling(double[] face, int start)
    {
        var sb = new StringBuilder();
        for (var s = 0; s < 2; s++)
        {
            var ss = start + (s * FaceLayout.SpellingSlotDims);
            if (RangeNorm(face, ss, ss + FaceLayout.SpellingSlotDims) < 1e-6)
                break;
            sb.Append(NearestChar(face, ss));
        }
        return sb.ToString();
    }

    /// <summary>Nearest role/label in <paramref name="vocab"/> for the 8-dim code at <paramref name="roleStart"/>.</summary>
    private static string NearestLabel(double[] face, int roleStart, IReadOnlyCollection<string> vocab)
    {
        var best = string.Empty;
        var bestDist = double.MaxValue;
        foreach (var cand in vocab)
        {
            if (string.IsNullOrEmpty(cand)) continue;
            var code = PlatonicFaceComposer.LabelCode(cand);
            var dist = 0.0;
            for (var k = 0; k < FaceLayout.StructureRoleDims && roleStart + k < face.Length; k++)
            {
                var diff = face[roleStart + k] - code[k];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = cand; }
        }
        return best;
    }

    /// <summary>Euclidean norm of <paramref name="vec"/> over [start, end).</summary>
    private static double RangeNorm(double[] vec, int start, int end)
    {
        var sum = 0.0;
        for (var i = start; i < end && i < vec.Length; i++)
            sum += vec[i] * vec[i];
        return Math.Sqrt(sum);
    }
}
