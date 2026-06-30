using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Persistence;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Modules = TorchSharp.Modules;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE QUERY-CONDITIONED NAVIGATOR POLICY  (PLATONIC_NAVIGATOR.md §2/§3/§6; PLATONIC_MIND.md §3 "an answer is wherever
//  a query relaxes to").
//
//  Same recurrent differential recogniser as NavigatorPolicyNet, but goal-conditioning is REPLACED by a query-context
//  the walker has WITHOUT the answer:  (anchorFace, cue).
//    • the self h₀ is SEEDED from  tanh(W·anchorFace + cueEmbedding[cue])  — the walk starts as the self, oriented by
//      the question (the anchor it is asked about) and the target-aspect tension (the cue), not by the answer;
//    • each hop the cue embedding is MIXED into the GRU input (it persists as the question-tension), alongside the
//      masked candidate summary and the (cur − anchor) displacement context;
//    • the HALT and VALUE heads read concat[self, cue], so "am I resolved?" is learned PER CUE — the same node halts
//      for GENUS but steps onward for DOMAIN/ROOT.
//
//  Candidate features are the answer-free differential rows from NavQueryFeatures ([cand−anchor, cand−cur, cand, κ]).
//  The net still never emits an answer from weights: it SCORES substrate-enumerated candidates and the chosen
//  candidate's FACE is the target the lattice lands (§5.1).
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// The query-conditioned recurrent policy/value network (PLATONIC_NAVIGATOR.md §6). Two steps like its goal-conditioned
/// sibling: <see cref="SeedHidden"/> (h₀ from anchor ⊕ cue) and <see cref="Step"/> (one hop: candidate features +
/// (cur−anchor) context + cue + h ⇒ candidate logits, halt logit, value, next h). A hand-rolled GRU carries the self.
/// </summary>
public sealed class NavQueryPolicyNet : Module
{
    private readonly Modules.Linear _candEnc;    // per-candidate differential feature → embedding
    private readonly Modules.Linear _ctxEnc;     // (cur − anchor) displacement context → embedding
    private readonly Modules.Linear _h0Enc;      // anchor face → seed hidden h₀
    private readonly Modules.Linear _selfEnc;    // W_s: the PERSISTENT SELF (meaning-space vector) → seed hidden bias
    private readonly Modules.Linear _kindEnc;    // W_k (M2): the TARGET-KIND face (the category the answer belongs to) →
                                                 // seed/per-hop bias. Goal-EMERGENT: the walk knows the KIND it seeks, not
                                                 // the answer. Null kind ⇒ adds EXACTLY zero (no bias) ⇒ M1-byte-identical.
    private readonly Modules.Embedding _cueEmb;  // learned cue embedding over {GENUS, DOMAIN, ROOT}
    private readonly Modules.Linear _gx;          // GRU input transform: [summ, ctx, cue] = 3H → 3·Hidden (r,z,n gates)
    private readonly Modules.Linear _gh;          // GRU hidden transform → 3·Hidden
    private readonly Modules.Linear _score1;     // score MLP layer 1 over concat[h, candEmb]
    private readonly Modules.Linear _score2;     // score MLP layer 2 → one logit per candidate
    private readonly Modules.Linear _haltHead;   // halt logit from concat[h, cue]  (resolved FOR THIS CUE)
    private readonly Modules.Linear _valueHead;  // value (cost-to-go) from concat[h, cue]

    /// <summary>Face/coordinate dimension (e.g. 1024 — the decodable address space).</summary>
    public int Dim { get; }

    /// <summary>Per-candidate differential feature width (3·dim + 1).</summary>
    public int FeatureLength { get; }

    /// <summary>Hidden trunk / self width (kept thin — a controller, not a store).</summary>
    public int Hidden { get; }

    /// <summary>Number of target-aspect cues the embedding table covers ({GENUS, DOMAIN, ROOT} = 3).</summary>
    public int CueCount { get; }

    /// <summary>Width of the persistent-self vector W_s consumes (the meaning-space semantic length the engine's
    /// <c>SelfField</c> lives in). A self of this length seeds h₀; a null/empty self reduces to the query-only seed.</summary>
    public int SelfLength { get; }

    public NavQueryPolicyNet(int dim, int cueCount = NavQueryFeatures.CueCount, int hidden = 2048, Device? device = null, int? selfDim = null)
        : base(nameof(NavQueryPolicyNet))
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (cueCount <= 0) throw new ArgumentOutOfRangeException(nameof(cueCount));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        Dim = dim;
        Hidden = hidden;
        CueCount = cueCount;
        SelfLength = selfDim ?? FaceCodec.SemanticLength(dim); // the meaning-space self vector width (engine SelfField length)
        FeatureLength = NavQueryFeatures.FeatureLength(dim);

        _candEnc = Linear(FeatureLength, hidden);
        _ctxEnc = Linear(dim, hidden);
        _h0Enc = Linear(dim, hidden);
        _selfEnc = Linear(SelfLength, hidden, hasBias: false); // no bias → a null/zero self adds EXACTLY zero to the seed
        _kindEnc = Linear(dim, hidden, hasBias: false);        // no bias → a null/zero kind adds EXACTLY zero (M1-identical)
        _cueEmb = Embedding(cueCount, hidden);
        _gx = Linear(3 * hidden, 3 * hidden); // GRU input = concat[obsSummary(H), ctxEmb(H), cue(H)] = 3H
        _gh = Linear(hidden, 3 * hidden);
        _score1 = Linear(2 * hidden, hidden);
        _score2 = Linear(hidden, 1);
        _haltHead = Linear(2 * hidden, 1);  // (self ⊕ cue)
        _valueHead = Linear(2 * hidden, 1); // (self ⊕ cue)

        RegisterComponents();
        if (device is not null) _ = this.to(device);
    }

    /// <summary>The learned cue vector for a [B] long index tensor → [B, Hidden].</summary>
    public Tensor CueVector(Tensor cueIdx) => _cueEmb.forward(cueIdx);

    /// <summary>Seed the self h₀ from the query-context (PLATONIC_NAVIGATOR.md §3) AND the PERSISTENT SELF (the vital
    /// loop, PLATONIC_CONSCIOUSNESS.md): h₀ = tanh(W·anchor + cueEmb[cue] + W_s·self). <paramref name="anchor"/> =
    /// [B, dim]; <paramref name="cueIdx"/> = [B] long; <paramref name="selfVec"/> = [B, SelfLength] the engine's
    /// accumulated <c>SelfField</c> (null/empty → the self term vanishes and this reduces EXACTLY to the query-only
    /// seed). The self biases the AMBIGUOUS fork — exactly as it conditions <c>ds.Reason</c> in the field today.</summary>
    public Tensor SeedHidden(Tensor anchor, Tensor cueIdx, Tensor? selfVec = null, Tensor? kindFace = null)
    {
        var pre = _h0Enc.forward(anchor) + CueVector(cueIdx);
        if (selfVec is not null) pre = pre + _selfEnc.forward(selfVec); // W_s·self — the persistent self tilts the seed
        if (kindFace is not null) pre = pre + _kindEnc.forward(kindFace); // W_k·kind (M2) — orient the seed toward the sought KIND
        return tanh(pre);
    }

    /// <summary>
    /// One hop of the recurrent query policy. <paramref name="features"/> = [B, K, F] answer-free differential rows;
    /// <paramref name="mask"/> = [B, K]; <paramref name="context"/> = [B, dim] the (cur−anchor) displacement;
    /// <paramref name="cueIdx"/> = [B] long; <paramref name="h"/> = [B, Hidden]; <paramref name="selfVec"/> =
    /// [B, SelfLength] the PERSISTENT SELF (null → query-only, byte-identical). Returns candidate logits [B, K]
    /// (padding −∞), halt logit [B, 1], value [B, 1], next self [B, Hidden].
    /// </summary>
    public (Tensor candLogits, Tensor haltLogit, Tensor value, Tensor hNext) Step(
        Tensor features, Tensor mask, Tensor context, Tensor cueIdx, Tensor h, Tensor? selfVec = null, Tensor? kindFace = null)
    {
        var candEmb = functional.relu(_candEnc.forward(features));      // [B, K, H]

        var m = mask.unsqueeze(-1);                                    // [B, K, 1]
        var summ = (candEmb * m).sum(new long[] { 1 }) / m.sum(new long[] { 1 }).clamp_min(1e-6); // [B, H]
        var ctx = functional.relu(_ctxEnc.forward(context));           // [B, H]
        // The persistent question-tension. The SELF rides the SAME un-saturated channel as the cue, mixed in every hop
        // (the seed alone is tanh-saturated by the cue, so W_s gets no gradient there) — W_s·self + cueEmb. Null self →
        // exactly the cue (byte-identical to the query-only navigator). The TARGET-KIND (M2) rides the same channel, so
        // the halt/value heads (which read this cue channel) learn "resolved for THIS kind" with no new head wiring.
        var cue = CueVector(cueIdx);                                    // [B, H]
        if (selfVec is not null) cue = cue + _selfEnc.forward(selfVec); // the self conditions the walk, exactly per hop
        if (kindFace is not null) cue = cue + _kindEnc.forward(kindFace); // W_k·kind — the sought KIND conditions every hop + halt

        // GRU cell: thread the self with the new observation AND the cue⊕self.
        var hNext = GruCell(cat(new[] { summ, ctx, cue }, dim: 1), h);  // input 3H → [B, H]

        // Score each candidate from the updated self ⊕ its embedding.
        var k = features.shape[1];
        var hExp = hNext.unsqueeze(1).expand(new long[] { hNext.shape[0], k, Hidden }); // [B, K, H]
        var scoreIn = cat(new[] { hExp, candEmb }, dim: 2);            // [B, K, 2H]
        var logits = _score2.forward(functional.relu(_score1.forward(scoreIn))).squeeze(-1); // [B, K]
        logits = logits + (mask - 1f) * 1e9f;                          // mask padded candidates to −∞

        // HALT / VALUE conditioned on (self, cue) — "resolved for THIS cue".
        var headIn = cat(new[] { hNext, cue }, dim: 1);                // [B, 2H]
        var halt = _haltHead.forward(headIn);                          // [B, 1]
        var value = _valueHead.forward(headIn);                        // [B, 1]
        return (logits, halt, value, hNext);
    }

    // Hand-rolled GRU cell (version-independent). x = [B, 3H], h = [B, H].
    private Tensor GruCell(Tensor x, Tensor h)
    {
        var gi = _gx.forward(x);  // [B, 3H]
        var gh = _gh.forward(h);  // [B, 3H]
        var ix = gi.chunk(3, dim: 1);
        var ih = gh.chunk(3, dim: 1);
        var r = sigmoid(ix[0] + ih[0]);
        var z = sigmoid(ix[1] + ih[1]);
        var n = tanh(ix[2] + r * ih[2]);
        return (1f - z) * n + z * h;
    }

    /// <summary>Total trainable parameter count (diagnostic — proves the net stays a thin controller).</summary>
    public long ParameterCount() => parameters().Sum(p => p.numel());

    /// <summary>EXPORT every named weight tensor for the checkpoint as NATIVE f32 (the net's own dtype — lossless, and
    /// HALF the bytes of the former f64 export, so autosave I/O is halved). The architecture dims ride along so a load
    /// can detect a re-architected navigator and decline rather than crash.</summary>
    public NavigatorSnapshot ExportWeights()
    {
        using var _ = no_grad();
        var list = new List<NavParameterSnapshot>();
        foreach (var (name, p) in named_parameters())
        {
            using var t = p.detach().to(float32).cpu().contiguous();
            list.Add(new NavParameterSnapshot(name, p.shape.ToArray(), t.data<float>().ToArray()));
        }
        return new NavigatorSnapshot(Dim, Hidden, CueCount, SelfLength, list.ToArray());
    }

    /// <summary>RESTORE weights from a checkpoint IN PLACE (copy_ into this net's parameters). No-op when the snapshot
    /// is null/empty or its architecture dims don't match this net (a re-architected navigator stays freshly
    /// initialised — never throws), so an OLD checkpoint without navigator data, or a navigator built at different
    /// dims, both load cleanly. Per-tensor shape is re-checked so a partial/renamed param is skipped, not mis-copied.</summary>
    public void ImportWeights(NavigatorSnapshot? snapshot)
    {
        if (snapshot is null || snapshot.Parameters.Length == 0) return;
        if (snapshot.Dim != Dim || snapshot.Hidden != Hidden || snapshot.CueCount != CueCount || snapshot.SelfLength != SelfLength)
            return; // architecture changed → keep the fresh net rather than mis-load
        using var _ = no_grad();
        var byName = named_parameters().ToDictionary(np => np.name, np => np.parameter);
        foreach (var ps in snapshot.Parameters)
        {
            if (ps.Values.Length == 0) continue;
            if (!byName.TryGetValue(ps.Name, out var dst)) continue;
            if (!dst.shape.SequenceEqual(ps.Shape)) continue;
            using var src = tensor(ps.Values, ps.Shape).to(dst.dtype).to(dst.device);
            dst.copy_(src);
        }
    }
}

/// <summary>
/// THE QUERY-CONDITIONED POLICY as an <see cref="INavPolicy"/> (PLATONIC_NAVIGATOR.md §6) — drops into
/// <see cref="NavigatorWalk"/> unchanged. UNLIKE <see cref="NavNetPolicy"/> it is NOT given the goal: the anchor face
/// and the cue are fixed in the ctor (the query), and <see cref="Decide"/> reads ONLY <c>state.CurrentSymbol/Face</c>
/// plus that stored query-context. Run the walk with <c>goalSymbol=null</c> so the loop relies on this policy's learned
/// HALT (not goal-reached): when the halt head fires the agent stands on the answer the query relaxed to. <see
/// cref="LastHalt"/> records whether the most recent decision was a halt — the caller uses it to tell a confident halt
/// from a budget-exhausted abstain (§8).
/// </summary>
public sealed class QueryNavPolicy : INavPolicy, IDisposable
{
    private readonly NavQueryPolicyNet _net;
    private readonly DialecticalSpace _space;
    private readonly Device _device;
    private readonly double[] _anchorFace;
    private readonly int _cue;
    private readonly float[]? _selfVec; // the PERSISTENT SELF (engine SelfField) that seeds h₀; null → query-only seed
    private readonly float[]? _kindFace; // the TARGET-KIND face (M2) — the category the answer belongs to; null → no kind bias
    private readonly int _k;
    private readonly double _minConfidence;
    private readonly double _haltThreshold;
    private Tensor? _h; // the threaded self (persists across Decide calls within a walk)

    /// <summary>True iff the most recent <see cref="Decide"/> returned a HALT (confident halt OR a structural dead-end).
    /// False after a step move — so budget exhaustion (the loop exits after a non-halt move) leaves this false.</summary>
    public bool LastHalt { get; private set; }

    public QueryNavPolicy(NavQueryPolicyNet net, DialecticalSpace space, double[] anchorFace, int cue,
        Device? device = null, int k = 16, double minConfidence = 0.0, double haltThreshold = 0.5, double[]? selfVec = null,
        double[]? kindFace = null)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        _space = space ?? throw new ArgumentNullException(nameof(space));
        _anchorFace = anchorFace ?? throw new ArgumentNullException(nameof(anchorFace));
        _cue = cue;
        // The persistent self that seeds the walk (null/empty → query-only seed, the un-conditioned default).
        if (selfVec is { Length: > 0 })
        {
            _selfVec = new float[selfVec.Length];
            for (var i = 0; i < selfVec.Length; i++) _selfVec[i] = (float)selfVec[i];
        }
        // The TARGET-KIND face (M2) — full dim, must match the net's face dim; null/empty → no kind bias (M1 walk).
        if (kindFace is { Length: > 0 } && kindFace.Length >= net.Dim)
        {
            _kindFace = new float[net.Dim];
            for (var i = 0; i < net.Dim; i++) _kindFace[i] = (float)kindFace[i];
        }
        _device = device ?? CPU;
        _k = k;
        _minConfidence = minConfidence;
        _haltThreshold = haltThreshold;
    }

    public NavDecision Decide(NavState state)
    {
        var dim = _net.Dim;
        if (state.CurrentFace is null || state.CurrentFace.Length < dim)
        { LastHalt = true; return new NavDecision(Array.Empty<double>(), Halt: true); } // can't sense → abstain

        _net.eval();
        using var _ = no_grad();
        using var scope = NewDisposeScope();

        using var cueT = tensor(new long[] { _cue }, new long[] { 1 }, device: _device);
        // The PERSISTENT SELF (engine SelfField) as a tensor, reused for the seed AND every hop (null → query-only).
        using var selfT = _selfVec is null ? null : tensor(_selfVec, new long[] { 1, _selfVec.Length }, device: _device);
        // The TARGET-KIND face (M2) as a tensor, reused for the seed AND every hop (null → no kind bias = M1 walk).
        using var kindT = _kindFace is null ? null : tensor(_kindFace, new long[] { 1, _kindFace.Length }, device: _device);

        // (Re)seed the self at the start of a walk from the QUERY-CONTEXT (anchor ⊕ cue) AND the PERSISTENT SELF
        // (W_s·SelfField — the vital loop) AND the TARGET-KIND (W_k·kind); otherwise carry it. Null self/kind reduce out.
        if (state.Step == 0 || _h is null)
        {
            using var anchorSeed = ToTensor(_anchorFace, dim);
            ReplaceHidden(_net.SeedHidden(anchorSeed, cueT, selfT, kindT));
        }

        // Answer-free egocentric observation: candidate rows are differentials against the ANCHOR, not the goal.
        var obs = NavQueryFeatures.Build(_space, state.CurrentSymbol, state.CurrentFace, _anchorFace, _k, _minConfidence);
        if (obs.ValidCount == 0)
        { LastHalt = true; return new NavDecision(Array.Empty<double>(), Halt: true); } // dead end → structural abstain (§8)

        var f = _net.FeatureLength;
        using var features = tensor(obs.FeaturesFlat, new long[] { 1, _k, f }, device: _device);
        using var mask = tensor(obs.Mask, new long[] { 1, _k }, device: _device);

        var ctxArr = new float[dim];
        for (var i = 0; i < dim; i++) ctxArr[i] = (float)(state.CurrentFace[i] - _anchorFace[i]); // displacement from start
        using var context = tensor(ctxArr, new long[] { 1, dim }, device: _device);

        var (logits, haltLogit, _, hNext) = _net.Step(features, mask, context, cueT, _h!, selfT, kindT);
        ReplaceHidden(hNext); // thread the self forward

        var haltProb = sigmoid(haltLogit).cpu().item<float>();
        if (haltProb > _haltThreshold)
        { LastHalt = true; return new NavDecision(Array.Empty<double>(), Halt: true); }

        var best = (int)argmax(logits, dim: 1).cpu().item<long>();
        if (best < 0 || best >= obs.ValidCount)
        { LastHalt = true; return new NavDecision(Array.Empty<double>(), Halt: true); }

        // Emit the chosen candidate's FACE; the lattice lands exactly on it (decode-first, §5.1).
        LastHalt = false;
        return new NavDecision((double[])obs.CandidateFaces[best].Clone(), Halt: false);
    }

    private Tensor ToTensor(double[] face, int dim)
    {
        var arr = new float[dim];
        for (var i = 0; i < dim; i++) arr[i] = (float)face[i];
        return tensor(arr, new long[] { 1, dim }, device: _device);
    }

    // Persist the new hidden state across Decide calls: detach + survive the per-call dispose scope, free the old one.
    private void ReplaceHidden(Tensor next)
    {
        var keep = next.detach().clone();
        keep.MoveToOuterDisposeScope();
        _h?.Dispose();
        _h = keep;
    }

    public void Dispose()
    {
        _h?.Dispose();
        _h = null;
    }
}
