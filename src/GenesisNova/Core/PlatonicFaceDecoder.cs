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
///   <item><see cref="WordSlotDecode"/> ↔ <see cref="PlatonicFaceComposer.GetWordComposedEmbedding"/></item>
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

    /// <summary>
    /// Decode a predicted embedding back to a sentence by reading each word slot independently — the
    /// inverse of <see cref="PlatonicFaceComposer.GetWordComposedEmbedding"/>.
    /// <para>
    /// Word slots live at <see cref="FaceLayout.WordFaceStart"/> (falling back to dim/2 when
    /// <see cref="FaceLayout.WordFaceDims"/> == 0, exactly like the composer). Explicit space tokens
    /// are interleaved at odd slot indices identically to the encode, so they are skipped here and the
    /// decoded words are re-joined with single spaces. Each word slot is matched to the nearest word in
    /// <paramref name="wordVocab"/> by comparing the slot dims against that word's atom
    /// (<see cref="PlatonicFaceComposer.GetChunkComposedEmbedding"/> read from [dim/2..dim)).
    /// </para>
    /// </summary>
    public static string WordSlotDecode(double[] predicted, int dim, IReadOnlyCollection<string> wordVocab)
    {
        if (predicted is null || dim <= 0 || wordVocab is null || wordVocab.Count == 0)
            return string.Empty;

        var wordStart = FaceLayout.WordFaceStart(dim);
        var wordDims = FaceLayout.WordFaceDims(dim);
        var atomStart = dim / 2; // GetChunkComposedEmbedding fills [dim/2..dim)

        // No dedicated word face at this dim → fall back to the full semantic half (composer parity).
        if (wordDims == 0)
        {
            wordStart = dim / 2;
            wordDims = dim - wordStart;
        }

        var wSlotDims = FaceLayout.WordSlotDims(wordDims);
        var maxWSlots = wSlotDims > 0 ? wordDims / wSlotDims : 0;
        if (maxWSlots <= 0)
            return string.Empty;

        // Precompute each candidate word's atom slice [atomStart..atomStart+wSlotDims).
        var spaceAtom = PlatonicFaceComposer.GetChunkComposedEmbedding(" ", dim);

        var decoded = new List<string>();
        for (var i = 0; i < maxWSlots; i++)
        {
            var slotStart = wordStart + (i * wSlotDims);

            // End of sentence: first near-zero slot.
            if (RangeNorm(predicted, slotStart, slotStart + wSlotDims) < 1e-9)
                break;

            // Odd slots ARE the explicit space token (encode interleaves words and spaces). The space
            // is implicit when we re-join, so just skip these positions.
            if (i % 2 == 1)
                continue;

            var best = string.Empty;
            var bestDist = double.MaxValue;
            foreach (var word in wordVocab)
            {
                if (string.IsNullOrEmpty(word))
                    continue;
                var wordAtom = PlatonicFaceComposer.GetChunkComposedEmbedding(word, dim);
                var dist = SlotDistance(predicted, slotStart, wordAtom, atomStart, wSlotDims, dim);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    best = word;
                }
            }

            // Also consider the literal space atom — if the slot is closest to it, treat as a gap.
            var spaceDist = SlotDistance(predicted, slotStart, spaceAtom, atomStart, wSlotDims, dim);
            if (spaceDist < bestDist)
                continue;

            if (best.Length > 0)
                decoded.Add(best);
        }

        return string.Join(' ', decoded);
    }

    private static double SlotDistance(
        double[] predicted, int slotStart, double[] atom, int atomStart, int slotDims, int dim)
    {
        var dist = 0.0;
        for (var k = 0; k < slotDims && atomStart + k < dim && slotStart + k < dim; k++)
        {
            var diff = predicted[slotStart + k] - atom[atomStart + k];
            dist += diff * diff;
        }
        return dist;
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
