using System.Globalization;

namespace GenesisNova.Core;

/// <summary>
/// Canonical platonic face composer — ported from the genesis-engine source of truth
/// (Genesis.Shared.EmbeddingGenerator) and adapted to nova's <see cref="FaceLayout"/>.
/// <para>
/// Single source of truth for turning a token/string into a face embedding, so that BOTH
/// input embeddings (<c>InputEmbeddingComposer</c>) and stored concept faces
/// (<c>PlatonicSpaceMemory.CreateFace</c>) live in the SAME geometry. Previously these used
/// different ad-hoc encodings, which silently broke face distance / KNN between an input and
/// the concept it should match.
/// </para>
/// <para>
/// Layout: numbers → polynomial+logarithmic faces (homomorphic identity), single tokens →
/// clean per-char slots in the char face, whole strings/sentences → word slots in the word
/// face (so large strings live intact instead of being mean-pooled toward zero). Every
/// non-identity dimension is seeded with small deterministic noise so it is free to "wiggle"
/// (learnable) without disturbing the identity dims that dominate exact recall.
/// </para>
/// </summary>
internal static class PlatonicFaceComposer
{
    private static readonly char[] ArithmeticDelimiters =
        { ' ', ',', '(', ')', '+', '-', '*', '/', 'x', '=', '?', ':', ';', '\t', '\n' };

    // Strict numeric classification: only a plain signed decimal is treated as a number and
    // composed into the polynomial/log (arithmetic) face. NumberStyles.Any would accept
    // trailing-sign garbage like "0+"/"5-" (parses as 0/5) plus thousands/exponent/currency/parens,
    // letting a malformed token masquerade as numeric. Stays aligned with
    // PlatonicSpaceMemory.TryParseNumber so a token is numeric in exactly one consistent way.
    private const NumberStyles NumericStyle =
        NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint
        | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite;

    /// <summary>
    /// Compose a face for arbitrary input. Single tokens use the char face; multi-token strings
    /// use the dedicated word face so large strings live intact (no mean-pool washout).
    /// </summary>
    public static double[] Compose(string input, int dim)
    {
        if (dim <= 0)
            return Array.Empty<double>();
        if (string.IsNullOrWhiteSpace(input))
            return new double[dim];

        var tokens = input.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        // GEOMETRIC START (getting back to position-as-identity): a non-numeric concept's SEMANTIC face is a
        // NEUTRAL, whole-string-seeded small start (via SeedLearnableDims) — NOT a lexical chunk-hash — so the
        // message-passing/contrastive geometry (PlatonicSpaceMemory) is what POSITIONS related concepts near
        // each other, instead of spelling similarity ("four"≈"fruit") biasing the VP-Tree. Single tokens keep
        // their char-face spelling identity; multi-word concepts no longer impose the confusable chunk geometry.
        var embedding = tokens.Length >= 2
            ? new double[dim]                        // multi-word → neutral start; geometry does the positioning
            : GetFreshEmbedding(input.Trim(), dim);  // single token → numeric or char-face spelling

        SeedLearnableDims(embedding, input, dim);
        // C1: give every NON-numeric concept a distinct near-orthogonal IDENTITY in the word face [202,dim) — the
        // region FaceAwareDistance actually measures. That region previously held only ±0.01 seed noise
        // (AddWordIdentity existed but was never called), so concepts were born indistinguishable there and ALL
        // separation had to be manufactured by push/pull. Now they are born ~orthogonal; attraction pulls related
        // ones together. Numbers are excluded (their identity is the value face; their word face moves by attraction).
        var single = input.Trim();
        var isSingleNumber = tokens.Length == 1 &&
            double.TryParse(single, NumericStyle, CultureInfo.InvariantCulture, out _);
        if (!isSingleNumber)
            AddWordIdentity(embedding, input, dim);
        return embedding;
    }

    /// <summary>
    /// Write a PER-CONCEPT IDENTITY code into the word face <c>[WordFaceStart, dim)</c>: a deterministic,
    /// whole-string-seeded unit vector (≈orthogonal across distinct concepts in this many dims). It is keyed
    /// by the ENTIRE string, so "four"/"fruit"/"fort" get distinct, non-confusable identities (unlike the old
    /// per-char-chunk hash) and a multi-word concept is distinct from its constituent words. Decoupled from
    /// spelling (the char face) — this is "who is this concept", not "how is it spelled"; meaning lives in
    /// relations. No-op when there is no word face at this dim (dim ≤ 202). OVERWRITES the region (so any
    /// prior seed noise there is replaced by the stable code).
    /// </summary>
    public static void AddWordIdentity(double[] embedding, string concept, int dim)
    {
        var wordStart = FaceLayout.WordFaceStart(dim);
        var wordDims = FaceLayout.WordFaceDims(dim);
        if (wordDims <= 0 || string.IsNullOrEmpty(concept))
            return;
        var rng = new Random(unchecked((int)StableHash("id:" + concept))); // whole-string seed
        var sum = 0.0;
        for (var d = wordStart; d < wordStart + wordDims && d < dim && d < embedding.Length; d++)
        {
            var v = (rng.NextDouble() * 2.0) - 1.0;
            embedding[d] = v;
            sum += v * v;
        }
        var norm = Math.Sqrt(sum);
        if (norm <= 1e-12)
            return;
        var inv = 1.0 / norm;
        for (var d = wordStart; d < wordStart + wordDims && d < dim && d < embedding.Length; d++)
            embedding[d] *= inv;
    }

    /// <summary>Single-token embedding: numeric (poly+log) or char-composed, then seeded.</summary>
    public static double[] GetFreshEmbedding(string symbol, int dim)
    {
        var embedding = double.TryParse(symbol, NumericStyle, CultureInfo.InvariantCulture, out var numeric)
            ? GetFreshNumericEmbedding(numeric, dim)
            : GetCharComposedEmbedding(symbol, dim);
        SeedLearnableDims(embedding, symbol, dim);
        return embedding;
    }

    /// <summary>
    /// Pure numeric embedding. Polynomial face (linear for add/sub) + logarithmic face
    /// (linear for mul/div). embed(a)+embed(b)=embed(a+b); ln|a|+ln|b|=ln|a·b|.
    /// </summary>
    public static double[] GetFreshNumericEmbedding(double value, int dim)
    {
        var embedding = new double[dim];
        var numericDims = FaceLayout.NumericDims(dim);

        for (var i = 0; i < numericDims; i++)
            embedding[i] = value * Math.Pow(10.0, -(i + 1));

        var logStart = numericDims;
        var logDims = Math.Min(numericDims, dim - logStart);
        if (Math.Abs(value) > 1e-12)
        {
            var logValue = Math.Log(Math.Abs(value));
            for (var i = 0; i < logDims; i++)
                embedding[logStart + i] = logValue * Math.Pow(10.0, -(i + 1));
        }
        return embedding;
    }

    /// <summary>Single-character hash-seeded atom, unit-normalized over the semantic half [dim/2..dim).</summary>
    public static double[] GetCharAtomEmbedding(char c, int dim)
    {
        var embedding = new double[dim];
        var halfDim = dim / 2;
        var rng = new Random(c); // char code → deterministic across runs
        for (var d = halfDim; d < dim; d++)
            embedding[d] = (rng.NextDouble() * 2.0) - 1.0;
        NormalizeRange(embedding, halfDim, dim);
        return embedding;
    }

    /// <summary>
    /// Clean slot-encoded character embedding. The char face is divided into equal slots;
    /// slot i holds the first SlotDims components of the atom for s[i], scaled so norm stays ~1.
    /// Trivially separable per character (no hash collisions, unlike the prior bag-of-chars).
    /// </summary>
    public static double[] GetCharComposedEmbedding(string symbol, int dim)
    {
        var charStart = FaceLayout.CharFaceStart(dim);
        var charDims = FaceLayout.CharFaceDims(dim);
        var slotDims = FaceLayout.SlotDims(charDims);
        var maxSlots = slotDims > 0 ? charDims / slotDims : 0;
        var atomStart = dim / 2;

        var result = new double[dim];
        if (maxSlots <= 0)
            return result;

        var chars = Math.Min(symbol.Length, maxSlots);
        var scale = 1.0 / Math.Sqrt(maxSlots);
        for (var i = 0; i < chars; i++)
        {
            var atom = GetCharAtomEmbedding(symbol[i], dim);
            var slotStart = charStart + (i * slotDims);
            for (var k = 0; k < slotDims && atomStart + k < dim && slotStart + k < dim; k++)
                result[slotStart + k] = atom[atomStart + k] * scale;
        }
        return result;
    }

    /// <summary>
    /// Encode a sentence as a word-slot embedding. Tokenises by whitespace, interleaves explicit
    /// space tokens, and writes each word's chunk-composed atom into its dedicated word slot.
    /// Numeric tokens additionally sum into the arithmetic face so the homomorphism is preserved
    /// orthogonally to word routing.
    /// </summary>
    public static double[] GetWordComposedEmbedding(string sentence, int dim)
    {
        var wordStart = FaceLayout.WordFaceStart(dim);
        var wordDims = FaceLayout.WordFaceDims(dim);
        var atomStart = dim / 2;

        // No dedicated word face at this dim → fall back to the full semantic half (compat).
        if (wordDims == 0)
        {
            wordStart = dim / 2;
            wordDims = dim - wordStart;
        }

        var wSlotDims = FaceLayout.WordSlotDims(wordDims);
        var maxWSlots = wSlotDims > 0 ? wordDims / wSlotDims : 0;

        var result = new double[dim];
        var words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (maxWSlots > 0)
        {
            var tokenCount = words.Length == 0 ? 0 : (words.Length * 2) - 1;
            var count = Math.Min(tokenCount, maxWSlots);
            for (var i = 0; i < count; i++)
            {
                var token = (i % 2 == 0) ? words[i / 2] : " ";
                var wordAtom = GetChunkComposedEmbedding(token, dim);
                var slotStart = wordStart + (i * wSlotDims);
                for (var k = 0; k < wSlotDims && atomStart + k < dim && slotStart + k < dim; k++)
                    result[slotStart + k] = wordAtom[atomStart + k];
            }
        }

        // Arithmetic face: sum poly+log for any numeric tokens, preserving the homomorphism.
        if (dim >= 2)
        {
            var numericDims = FaceLayout.NumericDims(dim);
            var numTokens = sentence.Split(ArithmeticDelimiters, StringSplitOptions.RemoveEmptyEntries);
            foreach (var tok in numTokens)
            {
                if (double.TryParse(tok, NumericStyle, CultureInfo.InvariantCulture, out var numVal))
                {
                    var numEmbed = GetFreshNumericEmbedding(numVal, dim);
                    for (var d = 0; d < numericDims * 2 && d < dim; d++)
                        result[d] += numEmbed[d];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Per-word atom generator (5-char chunk hashing). Made public so the inverse decoder
    /// (<see cref="PlatonicFaceDecoder.WordSlotDecode"/>) can regenerate each word-vocabulary
    /// candidate's atom to match the encode geometry exactly.
    /// </summary>
    public static double[] GetChunkComposedEmbedding(string word, int dim)
    {
        var halfDim = dim / 2;
        var semanticDims = dim - halfDim;
        var slotDims = FaceLayout.ChunkSlotDims(semanticDims);
        var maxSlots = slotDims > 0 ? semanticDims / slotDims : 0;

        var result = new double[dim];
        if (maxSlots <= 0)
            return result;

        var chunks = GetWordChunks(word);
        if (chunks.Count == 0)
            return result;

        var norm = 1.0 / chunks.Count;
        for (var ci = 0; ci < chunks.Count; ci++)
        {
            var h = ChunkHashU32(chunks[ci]);
            var slot = ci % maxSlots;
            var slotStart = halfDim + (slot * slotDims);
            for (var k = 0; k < slotDims && slotStart + k < dim; k++)
            {
                // Independent per-dim hash mixing (avoids the geometric decay that collapsed
                // signal into k=0 and made distinct words nearly indistinguishable).
                var dk = h ^ ((uint)k * 2654435761u);
                dk *= 2246822519u;
                dk ^= dk >> 13;
                dk *= 3266489917u;
                dk ^= dk >> 16;
                result[slotStart + k] += (((double)dk / uint.MaxValue) - 0.5) * 2.0 * norm;
            }
        }
        return result;
    }

    private static List<string> GetWordChunks(string word, int chunkSize = 5)
    {
        var chunks = new List<string>();
        if (word.Length <= chunkSize)
        {
            chunks.Add(word);
            return chunks;
        }
        for (var i = 0; i <= word.Length - chunkSize; i++)
            chunks.Add(word.Substring(i, chunkSize));
        return chunks;
    }

    private static uint ChunkHashU32(string chunk)
    {
        var hash = 2166136261u;
        foreach (var c in chunk)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return hash;
    }

    /// <summary>
    /// Seed the learnable (non-identity) dims with small deterministic noise so graph alignment
    /// has signal, without disturbing the identity dims (arithmetic face for numbers, char face
    /// for text) that carry exact recall.
    /// </summary>
    public static void SeedLearnableDims(double[] embedding, string symbol, int dim)
    {
        var isNumeric = double.TryParse(symbol, NumericStyle, CultureInfo.InvariantCulture, out _);
        int identityStart, identityEnd;
        if (isNumeric)
        {
            identityStart = 0;
            identityEnd = FaceLayout.ArithmeticFaceEnd(dim);
        }
        else
        {
            identityStart = FaceLayout.CharFaceStart(dim);
            identityEnd = FaceLayout.WordFaceStart(dim);
        }

        // Deterministic, symbol-stable RNG (not string.GetHashCode, which is randomised per process).
        var rng = new Random(unchecked((int)StableHash(symbol)));
        const double scale = 0.01;
        for (var d = 0; d < dim && d < embedding.Length; d++)
        {
            if (d >= identityStart && d < identityEnd) continue;
            if (Math.Abs(embedding[d]) > 1e-12) continue;
            embedding[d] = ((rng.NextDouble() * 2.0) - 1.0) * scale;
        }
    }

    private static uint StableHash(string value)
    {
        var hash = 2166136261u;
        foreach (var c in value)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return hash;
    }

    private static void NormalizeRange(double[] vec, int start, int end)
    {
        var sum = 0.0;
        for (var i = start; i < end && i < vec.Length; i++)
            sum += vec[i] * vec[i];
        var norm = Math.Sqrt(sum);
        if (norm < 1e-12)
            return;
        var inv = 1.0 / norm;
        for (var i = start; i < end && i < vec.Length; i++)
            vec[i] *= inv;
    }
}
