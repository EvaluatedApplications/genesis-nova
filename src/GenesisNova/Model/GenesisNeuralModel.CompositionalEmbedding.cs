using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
    // ============================================================================================
    // BOUNDED-VOCAB + CHAR-FACE COMPOSITION — read ANY token within a FIXED parameter budget.
    //
    // The legacy model grows a learned ROW per token in three tables (_embT/_wOutT/_bOutT), so a large
    // corpus balloons the weights (~32k rows ≈ 535 MB) and load time without bound. Here, when
    // CompositionalTokenEmbedding is on, the learned per-token table is CAPPED at MaxModelVocab and a
    // token id BEYOND the cap (OOV) gets its INPUT embedding COMPOSED from its deterministic spelling
    // band ([48,208) — FaceLayout.SpellingStart..) via a single FIXED-SIZE learned projection
    // (Linear(SpellingDims → hidden)), instead of a fresh row. The table therefore stays bounded (a
    // fixed projection matrix + ≤cap rows) while the model can READ any spelling (every token has a
    // char face). Identity in the frozen char face IS the better basis the substrate already proves.
    //
    // OUTPUT side: _wOutT/_bOutT (the neural decoder) is bounded the same way (EnsureVocabularySizeGpu
    // clamps the table), but it is NOT extended to OOV here. In PRODUCTION the conscious-field path
    // (GenerateFromField → EmitField) answers from the SUBSTRATE and never decodes through _wOutT, so
    // the output table is VESTIGIAL for production generation; OOV neural-decode output is deferred to a
    // later stage (the field path / a char-face decoder). See the task notes / memory.
    // ============================================================================================

    // Learned projection charFace(spelling) → hidden. FIXED [SpellingDims, hidden] (vocab-INDEPENDENT), so reading
    // OOV tokens never grows a per-token row. Lazily created (like the route/trunk heads); autograd-trained.
    private TorchSharp.Modules.Parameter? _charProjWT; // [CharSpellingDims, hidden]
    private TorchSharp.Modules.Parameter? _charProjB;  // [hidden]

    // token id → case-folded spelling, set by the inference engine / trainer from their tokenizer (the model layer
    // has no tokenizer). Consulted ONLY when CompositionalTokenEmbedding is on and a token id is OOV (beyond the cap).
    private Func<int, string?>? _tokenSpelling;

    // Width of the deterministic spelling band fed to the projection (FaceLayout spelling band [48,208) = 160 dims).
    private const int CharSpellingDims = FaceLayout.SpellingDims;

    /// <summary>Inject the token-id → spelling resolver (from the driving tokenizer). Null disables OOV composition.</summary>
    public void SetTokenSpelling(Func<int, string?>? resolver) => _tokenSpelling = resolver;

    /// <summary>True when bounded-vocab + char-face composition is active (flag on AND a positive cap).</summary>
    private bool CompositionalEmbeddingOn => _config.CompositionalTokenEmbedding && _config.MaxModelVocab > 0;

    /// <summary>The bounded per-token table cap when the feature is on.</summary>
    private int BoundedVocabCap => _config.MaxModelVocab;

    private void EnsureCharProjInitialized()
    {
        if (_charProjWT is not null)
            return;
        // Small-uniform init in [-0.05,0.05], matching the reasoning trunk / output table init scale.
        _charProjWT = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { CharSpellingDims, _hiddenSize }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _charProjB = new TorchSharp.Modules.Parameter(
            zeros(new long[] { _hiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        RecreateOptimizer();
    }

    /// <summary>
    /// The deterministic spelling band for a symbol as float[CharSpellingDims]. Composed at an ADDRESS-SPACE face
    /// dim (≥512) so the fixed spelling band [48,208) exists; the band is a pure function of the spelling (the same
    /// decodable identity the substrate stores), so distinct spellings yield distinct bands — a genuine read, never
    /// a shared UNK.
    /// </summary>
    private float[] SpellingBand(string symbol)
    {
        var faceDim = Math.Max(FaceLayout.AddressSpaceDim, _config.FaceDimension);
        var face = PlatonicFaceComposer.GetFreshEmbedding(symbol, faceDim);
        var band = new float[CharSpellingDims];
        for (var i = 0; i < CharSpellingDims && FaceLayout.SpellingStart + i < face.Length; i++)
            band[i] = (float)face[FaceLayout.SpellingStart + i];
        return band;
    }

    /// <summary>
    /// Compose an OOV token's input embedding from its spelling via the learned projection: tanh(spellingBand·W + b),
    /// a [hidden] vector on <paramref name="device"/>. Intermediates are appended to <paramref name="scratch"/> (when
    /// provided) for post-backward disposal — mirroring GruStep's discipline; the returned activation is the live
    /// graph node the caller tracks. During training this is a live autograd path, so the projection LEARNS.
    /// </summary>
    private Tensor ComposeFromCharFace(string symbol, Device device, List<Tensor>? scratch)
    {
        EnsureCharProjInitialized();
        var band = SpellingBand(symbol);
        var vec = tensor(band, dtype: ScalarType.Float32, device: device); scratch?.Add(vec); // [CharSpellingDims]

        Tensor w = _charProjWT!, b = _charProjB!;
        if (device != _trainingDevice)
        {
            w = _charProjWT!.to(device); scratch?.Add(w);
            b = _charProjB!.to(device); scratch?.Add(b);
        }

        var proj = vec.matmul(w); scratch?.Add(proj);   // [hidden]
        var biased = proj + b; scratch?.Add(biased);    // [hidden]
        return biased.tanh();                            // returned; caller tracks it
    }
}
