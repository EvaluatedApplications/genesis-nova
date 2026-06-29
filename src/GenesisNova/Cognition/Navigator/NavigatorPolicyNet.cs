using System;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Modules = TorchSharp.Modules;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE NAVIGATOR'S LEARNED POLICY  (PLATONIC_NAVIGATOR.md §6 thin policy/value over egocentric features, §10 the
//  policy emits a CONTINUOUS target coordinate + the lattice snaps it, §7 trained by behavioural cloning on the
//  flow-field oracle).
//
//  A THIN MLP (recogniser/controller, never a store — nova-nn-recognizer-space-structural): in goes the agent's
//  first-person situation as the concatenation of WHERE IT STANDS (currentFace) and WHERE IT WANTS TO BE (goalFace);
//  out comes the next MOVE — a continuous target coordinate to land on (regression), a value (cost-to-go), and a HALT
//  logit. The net emits a POSITION, never an answer from weights; the substrate's lattice (TryLand) decodes where the
//  foot falls. It is the learned successor to FlowFieldPolicy: same INavPolicy.Decide seam, a learned target instead
//  of an oracle read.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Thin policy/value MLP for the navigator (PLATONIC_NAVIGATOR.md §6). Input = concat(currentFace, goalFace) of
/// 2·<see cref="Dim"/> floats; a 2-layer ReLU trunk feeds three heads: <c>target</c> (regression — the emitted target
/// coordinate, <see cref="Dim"/> dims), <c>value</c> (scalar cost-to-go), <c>halt</c> (scalar logit). Forward returns
/// (target, value, haltLogit). Float32 throughout (TorchSharp default; faces are double[] → cast at the seam).
/// </summary>
public sealed class NavigatorPolicyNet : Module<Tensor, (Tensor target, Tensor value, Tensor haltLogit)>
{
    private readonly Modules.Linear _fc1;
    private readonly Modules.Linear _fc2;
    private readonly Modules.Linear _targetHead;
    private readonly Modules.Linear _valueHead;
    private readonly Modules.Linear _haltHead;

    /// <summary>Coordinate dimension (the face width, e.g. 512 — the decodable address space).</summary>
    public int Dim { get; }

    /// <summary>Hidden trunk width (kept THIN — a controller, not a store).</summary>
    public int Hidden { get; }

    public NavigatorPolicyNet(int dim, int hidden = 256, Device? device = null) : base(nameof(NavigatorPolicyNet))
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        Dim = dim;
        Hidden = hidden;

        // Trunk: concat(currentFace, goalFace) [2·dim] → hidden → hidden (1–2 thin ReLU layers).
        _fc1 = Linear(2 * dim, hidden);
        _fc2 = Linear(hidden, hidden);
        // Three heads off the shared trunk.
        _targetHead = Linear(hidden, dim); // the emitted target coordinate (regression)
        _valueHead = Linear(hidden, 1);    // cost-to-go
        _haltHead = Linear(hidden, 1);     // halt logit

        RegisterComponents();
        if (device is not null) _ = this.to(device);
    }

    public override (Tensor target, Tensor value, Tensor haltLogit) forward(Tensor x)
    {
        var h1 = functional.relu(_fc1.forward(x));
        var h2 = functional.relu(_fc2.forward(h1));
        var target = _targetHead.forward(h2);
        var value = _valueHead.forward(h2);
        var halt = _haltHead.forward(h2);
        return (target, value, halt);
    }

    /// <summary>Total trainable parameter count (diagnostic — proves the net stays thin).</summary>
    public long ParameterCount()
    {
        long total = 0;
        foreach (var p in parameters()) total += p.numel();
        return total;
    }
}

/// <summary>
/// The LEARNED policy as an <see cref="INavPolicy"/> (PLATONIC_NAVIGATOR.md §6) — the seam the oracle's
/// <c>FlowFieldPolicy</c> occupied, now driven by a trained <see cref="NavigatorPolicyNet"/>. <see cref="Decide"/>
/// builds the input from <c>state.CurrentFace ⊕ state.GoalFace</c>, runs a no-grad forward, halts when
/// <c>sigmoid(haltLogit) &gt; threshold</c>, and otherwise emits the target head as the continuous target coordinate
/// the lattice will land (§10). No gradients at inference.
/// </summary>
public sealed class NavNetPolicy : INavPolicy
{
    private readonly NavigatorPolicyNet _net;
    private readonly Device _device;
    private readonly double _haltThreshold;

    public NavNetPolicy(NavigatorPolicyNet net, Device? device = null, double haltThreshold = 0.5)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        _device = device ?? CPU;
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

        var input = new float[2 * dim];
        for (var i = 0; i < dim; i++)
        {
            input[i] = (float)state.CurrentFace[i];
            input[dim + i] = (float)state.GoalFace[i];
        }

        using var x = tensor(input, new long[] { 1, 2 * dim }, device: _device);
        var (target, _, haltLogit) = _net.forward(x);

        var haltProb = sigmoid(haltLogit).cpu().item<float>();
        if (haltProb > _haltThreshold)
            return new NavDecision(Array.Empty<double>(), Halt: true);

        var tf = target.cpu().reshape(dim).data<float>().ToArray();
        var coord = new double[dim];
        for (var i = 0; i < dim; i++) coord[i] = tf[i];
        return new NavDecision(coord, Halt: false);
    }
}
