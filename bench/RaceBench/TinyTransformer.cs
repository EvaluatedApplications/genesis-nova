using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RaceBench;

/// <summary>
/// A standard decoder-only (GPT-style) transformer LM, built best-effort — pre-LN blocks, multi-head
/// causal self-attention, GELU feed-forward (4× expansion), learned positional embeddings. Hand-built from
/// Embedding/Linear/LayerNorm primitives so it is correct on the pinned TorchSharp version. NOT gimped:
/// this is a real, competently-configured baseline. d_model matches the nova controller's hidden size.
/// </summary>
public sealed class TinyTransformer : Module<Tensor, Tensor>
{
    private readonly int _dModel;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly Embedding _tok;
    private readonly Parameter _pos;          // [maxLen, d]
    private readonly ModuleList<TransformerBlock> _blocks;
    private readonly LayerNorm _lnFinal;
    private readonly Linear _head;
    private readonly Device _dev;
    private readonly int _layers;

    public TinyTransformer(int vocab, int dModel, int heads, int layers, int ffMult, int maxLen, Device dev)
        : base(nameof(TinyTransformer))
    {
        _dModel = dModel;
        _heads = heads;
        _headDim = dModel / heads;
        _layers = layers;
        _dev = dev;

        _tok = Embedding(vocab, dModel);
        _pos = Parameter(torch.randn(maxLen, dModel) * 0.02);
        _blocks = new ModuleList<TransformerBlock>();
        for (var i = 0; i < layers; i++)
            _blocks.Add(new TransformerBlock(dModel, heads, ffMult));
        _lnFinal = LayerNorm(dModel);
        _head = Linear(dModel, vocab);

        RegisterComponents();
        this.to(dev);
    }

    /// <summary>idx: [B, T] (long token ids) → logits [B, T, vocab].</summary>
    public override Tensor forward(Tensor idx)
    {
        var t = (int)idx.shape[1];
        var x = _tok.forward(idx) + _pos.narrow(0, 0, t).unsqueeze(0); // [B,T,d]
        using var mask = CausalMask(t);
        for (var i = 0; i < _layers; i++)
            x = _blocks[i].forward(x, mask);
        x = _lnFinal.forward(x);
        return _head.forward(x); // [B,T,vocab]
    }

    // Additive causal mask: 0 on/below the diagonal, -inf above (a position cannot attend to the future).
    private Tensor CausalMask(int t)
        => torch.full(new long[] { t, t }, float.NegativeInfinity, dtype: ScalarType.Float32, device: _dev).triu(1);
}

/// <summary>Pre-LN transformer block: causal multi-head attention + GELU MLP, each with a residual.</summary>
internal sealed class TransformerBlock : Module<Tensor, Tensor, Tensor>
{
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _dModel;
    private readonly LayerNorm _ln1;
    private readonly LayerNorm _ln2;
    private readonly Linear _qkv;
    private readonly Linear _proj;
    private readonly Linear _fc1;
    private readonly Linear _fc2;

    public TransformerBlock(int dModel, int heads, int ffMult) : base(nameof(TransformerBlock))
    {
        _dModel = dModel;
        _heads = heads;
        _headDim = dModel / heads;
        _ln1 = LayerNorm(dModel);
        _ln2 = LayerNorm(dModel);
        _qkv = Linear(dModel, 3 * dModel);
        _proj = Linear(dModel, dModel);
        _fc1 = Linear(dModel, ffMult * dModel);
        _fc2 = Linear(ffMult * dModel, dModel);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor mask)
    {
        var b = x.shape[0];
        var t = x.shape[1];

        // --- Causal multi-head self-attention (pre-LN) ---
        var h = _ln1.forward(x);
        var qkv = _qkv.forward(h).chunk(3, -1); // 3 × [B,T,d]
        Tensor Heads(Tensor v) => v.view(b, t, _heads, _headDim).transpose(1, 2); // [B,heads,T,hd]
        var q = Heads(qkv[0]);
        var k = Heads(qkv[1]);
        var v2 = Heads(qkv[2]);
        var att = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(_headDim); // [B,heads,T,T]
        att = functional.softmax(att + mask, -1);
        var o = att.matmul(v2)                       // [B,heads,T,hd]
            .transpose(1, 2).contiguous().view(b, t, _dModel);
        x = x + _proj.forward(o);

        // --- Feed-forward (pre-LN) ---
        var f = _fc2.forward(functional.gelu(_fc1.forward(_ln2.forward(x))));
        return x + f;
    }
}
