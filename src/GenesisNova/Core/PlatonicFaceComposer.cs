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

        // ADDRESS-SPACE (dim ≥ 512): the spelling band IS the authoritative, decodable identity — no
        // random AddWordIdentity stamp. GetFreshEmbedding writes the numeric (poly/log) face OR the
        // spelling+kind identity (single OR multi-word: the whole trimmed string up to 16 chars), plus
        // the orbital spawn-seed. Frozen bands [0,416) stay a pure function of the symbol.
        if (FaceLayout.IsAddressSpace(dim))
            return GetFreshEmbedding(input.Trim(), dim);

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
        // ADDRESS-SPACE: disabled. The spelling band [48,208) is the authoritative, decodable identity
        // now, so no random unit-vector is stamped into the (re-purposed) high face — that would corrupt
        // the frozen structure/op/orbital bands. Kept live only for the legacy small-dim layout.
        if (FaceLayout.IsAddressSpace(dim))
            return;

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
        // ADDRESS-SPACE: write the AUTHORITATIVE, decodable spelling band [48,208) — 16 char-slots × 10
        // dims, slot i = deterministic atom of s[i] — plus a best-effort kind code. This is the inverse
        // partner of PlatonicFaceDecoder.CharSlotDecode. The legacy hash-seeded char face is used only
        // for small dims (< 512).
        if (FaceLayout.IsAddressSpace(dim))
            return GetSpellingEmbedding(symbol, dim);

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
        // ADDRESS-SPACE: the frozen address [0,OrbitalStart) is pure codec and must NOT be perturbed
        // (invariant: identical for a blank and a realised coordinate). Only the orbital tail gets the
        // symbol-stable spawn-spread that breaks symmetry for the learned region.
        if (FaceLayout.IsAddressSpace(dim))
        {
            var rngTail = new Random(unchecked((int)StableHash(symbol)));
            const double tailScale = 0.01;
            for (var d = FaceLayout.OrbitalStart; d < dim && d < embedding.Length; d++)
            {
                if (Math.Abs(embedding[d]) > 1e-12) continue;
                embedding[d] = ((rngTail.NextDouble() * 2.0) - 1.0) * tailScale;
            }
            return;
        }

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

    // ============================================================================================
    // ADDRESS-SPACE ENCODERS (active when dim ≥ 512). Each frozen band is written here and read back
    // by the matching decoder in PlatonicFaceDecoder. These generators are the SOURCE OF TRUTH for the
    // deterministic per-kind / per-char / per-op / per-label codes; the decoder regenerates candidates
    // from the same functions (exactly like WordSlotDecode regenerates GetChunkComposedEmbedding).
    // ============================================================================================

    /// <summary>
    /// Best-effort kind from the raw symbol (precise kind stamping is a later layer's job):
    /// numeric → None (read off poly/log); a ⟨…⟩-bracketed symbol → Composition; anything else → Object.
    /// </summary>
    public static PlatonicKind KindForSymbol(string symbol)
    {
        if (string.IsNullOrEmpty(symbol))
            return PlatonicKind.None;
        if (double.TryParse(symbol, NumericStyle, CultureInfo.InvariantCulture, out _))
            return PlatonicKind.None;
        if (symbol[0] == '⟨')
            return PlatonicKind.Composition;
        return PlatonicKind.Object;
    }

    /// <summary>
    /// Write the 6-dim deterministic kind code into the kind band [42,48). None is the all-zero code;
    /// every other kind is one-hot. No-op below address-space dims. Inverse: <c>DecodeKind</c>.
    /// </summary>
    public static void EncodeKind(double[] face, PlatonicKind kind, int dim)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim))
            return;
        for (var k = 0; k < FaceLayout.KindDims && FaceLayout.KindStart + k < face.Length; k++)
            face[FaceLayout.KindStart + k] = 0.0;
        if (kind == PlatonicKind.None)
            return;
        var idx = (int)kind - 1; // Atom→0 … Composition→4
        if (idx >= 0 && idx < FaceLayout.KindDims && FaceLayout.KindStart + idx < face.Length)
            face[FaceLayout.KindStart + idx] = 1.0;
    }

    /// <summary>
    /// Address-space identity face: best-effort kind code + the decodable spelling band (16 char-slots ×
    /// 10 dims, slot i = atom of s[i]). The whole frozen [0,416) region is a pure function of the symbol.
    /// </summary>
    private static double[] GetSpellingEmbedding(string symbol, int dim)
    {
        var result = new double[dim];
        EncodeKind(result, KindForSymbol(symbol), dim);
        if (string.IsNullOrEmpty(symbol))
            return result;
        var chars = Math.Min(symbol.Length, FaceLayout.SpellingSlots);
        for (var i = 0; i < chars; i++)
        {
            var atom = SpellingCharAtom(symbol[i]);
            var slotStart = FaceLayout.SpellingStart + (i * FaceLayout.SpellingSlotDims);
            for (var k = 0; k < FaceLayout.SpellingSlotDims && slotStart + k < dim; k++)
                result[slotStart + k] = atom[k];
        }
        return result;
    }

    /// <summary>
    /// Deterministic unit atom (length <see cref="FaceLayout.SpellingSlotDims"/>) for a character — the
    /// per-slot code the spelling band writes and <c>CharSlotDecode</c> matches against. Public so the
    /// decoder regenerates the exact same atom.
    /// </summary>
    public static double[] SpellingCharAtom(char c)
        => DeterministicUnit(c.ToString(), FaceLayout.SpellingSlotDims, 0xC0DEC0DEu);

    /// <summary>16-dim deterministic op code for an operation token (e.g. "+","-","mul"). Inverse: <c>DecodeOp</c>.</summary>
    public static double[] OpCode(string op)
        => DeterministicUnit(op ?? string.Empty, FaceLayout.OpDims, 0x0F0F0F0Fu);

    /// <summary>8-dim deterministic role/label code for a structure slot. Empty label → all-zero code.</summary>
    public static double[] LabelCode(string label)
        => DeterministicUnit(label ?? string.Empty, FaceLayout.StructureRoleDims, 0x5A5A5A5Au);

    /// <summary>
    /// Deterministic, symbol-stable unit vector of the given length, namespaced by <paramref name="salt"/>
    /// so codes from different bands (chars / ops / labels) never collide. Empty key → all zeros.
    /// </summary>
    public static double[] DeterministicUnit(string key, int len, uint salt)
    {
        var v = new double[Math.Max(0, len)];
        if (string.IsNullOrEmpty(key) || len <= 0)
            return v;
        var rng = new Random(unchecked((int)(StableHash(key) ^ salt)));
        var sum = 0.0;
        for (var i = 0; i < len; i++) { v[i] = (rng.NextDouble() * 2.0) - 1.0; sum += v[i] * v[i]; }
        var inv = sum > 1e-12 ? 1.0 / Math.Sqrt(sum) : 0.0;
        for (var i = 0; i < len; i++) v[i] *= inv;
        return v;
    }

    /// <summary>
    /// Write the 16-dim op code into the op band [400,416). No-op below address-space dims.
    /// Inverse: <c>PlatonicFaceDecoder.DecodeOp</c>.
    /// </summary>
    public static void EncodeOp(double[] face, string op, int dim)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim))
            return;
        var code = OpCode(op);
        for (var k = 0; k < FaceLayout.OpDims && FaceLayout.OpStart + k < face.Length; k++)
            face[FaceLayout.OpStart + k] = code[k];
    }

    /// <summary>
    /// Encode an ordered list of child coordinates (+ a shared role/label) into the structure band
    /// [208,400): up to 6 slots × 32 dims = (child-digest 24 + role/label 8). The child digest stores
    /// the child's numeric head (poly[0..1]+log[0..1], 4 dims) so a NUMERIC child decodes its value
    /// EXACTLY, followed by the child's first 2 spelling slots (20 dims) so a SHORT atom child decodes
    /// its first 2 chars. <b>Boundary:</b> a child longer than 2 chars only resolves to this digest here;
    /// the substrate layer cleans it up against the child's realised coordinate. Inverse: <c>DecodeStructure</c>.
    /// </summary>
    public static void EncodeStructure(double[] face, IReadOnlyList<double[]> childCoords, string label, int dim)
    {
        if (face is null || !FaceLayout.IsAddressSpace(dim) || childCoords is null)
            return;
        var labelCode = LabelCode(label);
        var n = Math.Min(childCoords.Count, FaceLayout.StructureSlots);
        var logStart = FaceLayout.LogFaceStart(dim);
        for (var i = 0; i < n; i++)
        {
            var child = childCoords[i];
            var slotStart = FaceLayout.StructureStart + (i * FaceLayout.StructureSlotDims);
            if (child is not null && child.Length > 0)
            {
                // Numeric head (4 dims): poly[0],poly[1],log[0],log[1].
                if (FaceLayout.PolyFaceStart < child.Length) face[slotStart + 0] = child[FaceLayout.PolyFaceStart];
                if (FaceLayout.PolyFaceStart + 1 < child.Length) face[slotStart + 1] = child[FaceLayout.PolyFaceStart + 1];
                if (logStart < child.Length) face[slotStart + 2] = child[logStart];
                if (logStart + 1 < child.Length) face[slotStart + 3] = child[logStart + 1];
                // First 2 spelling slots (20 dims) so a short atom child decodes its leading chars.
                var spellDigest = 2 * FaceLayout.SpellingSlotDims;
                for (var k = 0; k < spellDigest; k++)
                {
                    var src = FaceLayout.SpellingStart + k;
                    if (src < child.Length && slotStart + 4 + k < face.Length)
                        face[slotStart + 4 + k] = child[src];
                }
            }
            // Role/label code (8 dims) at the tail of the slot.
            for (var k = 0; k < FaceLayout.StructureRoleDims && slotStart + FaceLayout.StructureChildDigestDims + k < face.Length; k++)
                face[slotStart + FaceLayout.StructureChildDigestDims + k] = labelCode[k];
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
