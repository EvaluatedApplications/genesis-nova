using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Modules = TorchSharp.Modules;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE NAVIGATOR'S LEARNED POLICY — a LOCAL DIFFERENTIAL RECOGNISER, recurrent  (PLATONIC_NAVIGATOR.md §2 egocentric
//  observation, §3 the self h_t threads the walk, §6 thin recogniser, §7 oracle supervision).
//
//  WHY differential + recurrent (first principles): meaning is DIFFERENTIAL (a concept is its contrasts κ); the NN must
//  RECOGNISE structure, never store it (nova-nn-recognizer-space-structural). So the policy SCORES the local candidate
//  neighbours from their CONTRAST to the goal and to here (NavFeatures) — a rule universal across graphs, so it
//  generalises to held-out chains instead of memorising a per-node lookup (the prior MLP's 10% held-out collapse). A
//  GRU self h_t threads the walk first-person (§3), seeded from the goal, updated from each step's observation.
//
//  The net never emits an answer from weights: it SCORES candidates the substrate enumerated and the chosen candidate's
//  FACE is the emitted target the lattice lands (§5.1). Heads: candidate-logits (softmax = the policy over the k
//  neighbours), HALT logit, VALUE V(state) (cost-to-go, the dense reward field supervises it). Trunk width = Hidden.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// The recurrent differential-recogniser policy/value network (PLATONIC_NAVIGATOR.md §6). It does NOT have a single
/// fixed-arity forward (the observation is a variable candidate set), so it exposes two steps: <see cref="SeedHidden"/>
/// (h₀ from the goal face) and <see cref="Step"/> (one hop: candidate features + (cur−goal) context + h ⇒ candidate
/// logits, halt logit, value, and the next h). A hand-rolled GRU cell carries the self across hops. Float32 throughout.
/// </summary>
public sealed class NavigatorPolicyNet : Module
{
    private readonly Modules.Linear _candEnc;   // per-candidate differential feature → embedding
    private readonly Modules.Linear _ctxEnc;    // (cur − goal) context → embedding
    private readonly Modules.Linear _h0Enc;     // goal face → seed hidden h₀
    private readonly Modules.Linear _gx;         // GRU input transform → 3·Hidden (r,z,n gates)
    private readonly Modules.Linear _gh;         // GRU hidden transform → 3·Hidden
    private readonly Modules.Linear _score1;    // score MLP layer 1 over concat[h, candEmb]
    private readonly Modules.Linear _score2;    // score MLP layer 2 → one logit per candidate
    private readonly Modules.Linear _haltHead;  // halt logit from h
    private readonly Modules.Linear _valueHead; // value (cost-to-go) from h

    /// <summary>Face/coordinate dimension (e.g. 512 — the decodable address space).</summary>
    public int Dim { get; }

    /// <summary>Per-candidate differential feature width (3·dim + 1).</summary>
    public int FeatureLength { get; }

    /// <summary>Hidden trunk / self width (kept thin — a controller, not a store).</summary>
    public int Hidden { get; }

    public NavigatorPolicyNet(int dim, int hidden = 2048, Device? device = null) : base(nameof(NavigatorPolicyNet))
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        Dim = dim;
        Hidden = hidden;
        FeatureLength = NavFeatures.FeatureLength(dim);

        _candEnc = Linear(FeatureLength, hidden);
        _ctxEnc = Linear(dim, hidden);
        _h0Enc = Linear(dim, hidden);
        _gx = Linear(2 * hidden, 3 * hidden); // GRU input = concat[obsSummary(H), ctxEmb(H)] = 2H
        _gh = Linear(hidden, 3 * hidden);
        _score1 = Linear(2 * hidden, hidden);
        _score2 = Linear(hidden, 1);
        _haltHead = Linear(hidden, 1);
        _valueHead = Linear(hidden, 1);

        RegisterComponents();
        if (device is not null) _ = this.to(device);
    }

    /// <summary>Seed the self h₀ from the goal face (PLATONIC_NAVIGATOR.md §3 "the walk starts as the self, oriented by
    /// the question"). <paramref name="goal"/> is [B, dim]; returns h₀ [B, Hidden].</summary>
    public Tensor SeedHidden(Tensor goal) => tanh(_h0Enc.forward(goal));

    /// <summary>
    /// One hop of the recurrent policy. <paramref name="features"/> = [B, K, F] differential candidate rows;
    /// <paramref name="mask"/> = [B, K] (1 valid / 0 pad); <paramref name="context"/> = [B, dim] the (cur−goal) contrast;
    /// <paramref name="h"/> = [B, Hidden] the incoming self. Returns the candidate logits [B, K] (padding masked to
    /// −∞), the halt logit [B, 1], the value [B, 1], and the updated self hₙ [B, Hidden].
    /// </summary>
    public (Tensor candLogits, Tensor haltLogit, Tensor value, Tensor hNext) Step(Tensor features, Tensor mask, Tensor context, Tensor h)
    {
        // Per-candidate embedding of the differential feature rows.
        var candEmb = functional.relu(_candEnc.forward(features));     // [B, K, H]

        // Observation summary = masked mean over valid candidate embeddings.
        var m = mask.unsqueeze(-1);                                     // [B, K, 1]
        var summ = (candEmb * m).sum(new long[] { 1 }) / m.sum(new long[] { 1 }).clamp_min(1e-6); // [B, H]
        var ctx = functional.relu(_ctxEnc.forward(context));           // [B, H]

        // GRU cell: thread the self with the new observation.
        var hNext = GruCell(cat(new[] { summ, ctx }, dim: 1), h);      // [B, H]

        // Score each candidate from the updated self ⊕ its embedding.
        var k = features.shape[1];
        var hExp = hNext.unsqueeze(1).expand(new long[] { hNext.shape[0], k, Hidden }); // [B, K, H]
        var scoreIn = cat(new[] { hExp, candEmb }, dim: 2);            // [B, K, 2H]
        var logits = _score2.forward(functional.relu(_score1.forward(scoreIn))).squeeze(-1); // [B, K]
        // Mask padded candidates to −∞ so softmax/argmax ignore them.
        logits = logits + (mask - 1f) * 1e9f;

        var halt = _haltHead.forward(hNext);                           // [B, 1]
        var value = _valueHead.forward(hNext);                         // [B, 1]
        return (logits, halt, value, hNext);
    }

    // Hand-rolled GRU cell (version-independent; mirrors the repo's preference for raw GRU math). x = [B, 2H], h = [B, H].
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
}

/// <summary>
/// The LEARNED policy as an <see cref="INavPolicy"/> (PLATONIC_NAVIGATOR.md §6) — drops into <see cref="NavigatorWalk"/>
/// unchanged. <see cref="Decide"/> re-derives the local candidate neighbourhood from <c>state.CurrentSymbol</c> via the
/// space (<see cref="NavFeatures.Build"/>), runs one recurrent <see cref="NavigatorPolicyNet.Step"/>, and either HALTs
/// (sigmoid(haltLogit) &gt; threshold, or a dead end with no candidates) or emits the ARGMAX candidate's FACE as the
/// continuous target the lattice lands (§5.1). The self h_t is threaded across the walk: it is reseeded from the goal
/// whenever <c>state.Step == 0</c> (a fresh walk) and carried otherwise. No gradients at inference.
/// </summary>
public sealed class NavNetPolicy : INavPolicy, IDisposable
{
    private readonly NavigatorPolicyNet _net;
    private readonly DialecticalSpace _space;
    private readonly Device _device;
    private readonly int _k;
    private readonly double _minConfidence;
    private readonly double _haltThreshold;
    private Tensor? _h; // the threaded self (persists across Decide calls within a walk)

    public NavNetPolicy(NavigatorPolicyNet net, DialecticalSpace space, Device? device = null,
        int k = 16, double minConfidence = 0.0, double haltThreshold = 0.5)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        _space = space ?? throw new ArgumentNullException(nameof(space));
        _device = device ?? CPU;
        _k = k;
        _minConfidence = minConfidence;
        _haltThreshold = haltThreshold;
    }

    public NavDecision Decide(NavState state)
    {
        var dim = _net.Dim;
        if (state.CurrentFace is null || state.GoalFace is null ||
            state.CurrentFace.Length < dim || state.GoalFace.Length < dim)
            return new NavDecision(Array.Empty<double>(), Halt: true); // can't sense → abstain

        _net.eval();
        using var _ = no_grad();
        using var scope = NewDisposeScope();

        // (Re)seed the self at the start of a walk; otherwise carry it.
        if (state.Step == 0 || _h is null)
        {
            var goalSeed = ToTensor(state.GoalFace, dim);
            ReplaceHidden(_net.SeedHidden(goalSeed));
        }

        // Egocentric differential observation from the substrate.
        var obs = NavFeatures.Build(_space, state.CurrentSymbol, state.CurrentFace, state.GoalFace, _k, _minConfidence);
        if (obs.ValidCount == 0)
            return new NavDecision(Array.Empty<double>(), Halt: true); // dead end → structural abstain (§8)

        var f = _net.FeatureLength;
        using var features = tensor(obs.FeaturesFlat, new long[] { 1, _k, f }, device: _device);
        using var mask = tensor(obs.Mask, new long[] { 1, _k }, device: _device);

        var ctxArr = new float[dim];
        for (var i = 0; i < dim; i++) ctxArr[i] = (float)(state.CurrentFace[i] - state.GoalFace[i]);
        using var context = tensor(ctxArr, new long[] { 1, dim }, device: _device);

        var (logits, haltLogit, _, hNext) = _net.Step(features, mask, context, _h!);
        ReplaceHidden(hNext); // thread the self forward

        var haltProb = sigmoid(haltLogit).cpu().item<float>();
        if (haltProb > _haltThreshold)
            return new NavDecision(Array.Empty<double>(), Halt: true);

        var best = (int)argmax(logits, dim: 1).cpu().item<long>();
        if (best < 0 || best >= obs.ValidCount)
            return new NavDecision(Array.Empty<double>(), Halt: true);

        // Emit the chosen candidate's FACE; the lattice lands exactly on it (decode-first, §5.1).
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
