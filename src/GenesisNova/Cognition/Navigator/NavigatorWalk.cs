using GenesisNova.Cognition.Platonic;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE NAVIGATOR WALK LOOP  (PLATONIC_NAVIGATOR.md §2 the agent's loop, §5.1 one motion primitive, §5.2 materialise,
//  §8 step budget + abstain).
//
//  This is the ENVIRONMENT a policy drives to walk the platonic address space to an answer. Reasoning is a SITUATED
//  WALK (§2): the agent stands at a coordinate, senses (NavState), the policy emits a TARGET coordinate (NavDecision),
//  the lattice LANDS the step (DialecticalSpace.TryLand, §5.1 "continuous intent, discrete decodable landing"), and it
//  repeats until it stands on the answer or halts. The answer is WHERE THE WALK ENDS.
//
//  The policy is a SEAM (INavPolicy). Here it is driven by the deterministic FLOW-FIELD ORACLE (FlowFieldPolicy, §7) —
//  no NN — to prove the loop. The future learned policy implements the SAME INavPolicy.Decide(NavState)->NavDecision:
//  a net that emits a continuous target face + halt. Nothing in the loop is oracle-specific.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>What the agent senses from where it stands (first person, §2). The policy decides from this alone.</summary>
public readonly record struct NavState(
    string CurrentSymbol,   // decoded identity of the current coordinate
    double[] CurrentFace,   // the current coordinate itself (canonical face)
    double[] GoalFace,      // the encoded goal/question coordinate
    string? GoalSymbol,     // the known answer symbol when training/proving (null at open inference)
    int Step);              // how many hops taken so far (against the budget / light-cone)

/// <summary>The policy's move (§5.1): emit a continuous TARGET coordinate to land on, and/or HALT. Every action type
/// reduces to this one primitive — the lattice resolves where the foot actually falls.</summary>
public readonly record struct NavDecision(double[] Target, bool Halt);

/// <summary>The seam a policy implements. The oracle implements it here; the learned net implements it later.</summary>
public interface INavPolicy
{
    NavDecision Decide(NavState state);
}

/// <summary>Outcome of one walk. <see cref="Reached"/>=false at budget / dead-end / cycle is the STRUCTURAL ABSTAIN
/// (§8): no confident halt inside the cognitive light-cone → "I don't know".</summary>
public readonly record struct NavWalkResult(
    bool Reached,
    IReadOnlyList<string> Trajectory,
    int Steps,
    string FinalSymbol);

/// <summary>Walk knobs. <see cref="MaxSteps"/> is the cognitive light-cone (§8). <see cref="MaterialiseOnSuccess"/>
/// commits every coordinate the successful trail blazed (genesis-tick growth, §5.2).</summary>
public sealed record NavWalkOptions(int MaxSteps = 32, bool MaterialiseOnSuccess = false);

/// <summary>
/// The walk loop: a policy drives the agent hop-by-hop through the address space until it stands on the answer or
/// halts/abstains. Each hop: build <see cref="NavState"/> → <c>policy.Decide</c> → land the emitted target on the
/// lattice (<see cref="DialecticalSpace.TryLand"/>) → step. Guards: a STEP BUDGET (<see cref="NavWalkOptions.MaxSteps"/>)
/// and a CYCLE / NO-PROGRESS guard (landing on the current node, or revisiting a node, with progress unproven → abstain).
/// </summary>
public sealed class NavigatorWalk
{
    public NavWalkResult Walk(
        DialecticalSpace space,
        string startSymbol,
        double[] goalFace,
        string? goalSymbol,
        INavPolicy policy,
        NavWalkOptions options)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentNullException.ThrowIfNull(options);

        // The agent must be able to STAND where it starts — read the start coordinate. If we can't even decode a face
        // for the start, there is nowhere to begin: abstain immediately (no exception, no loop).
        if (string.IsNullOrWhiteSpace(startSymbol) || !space.TryGetConceptFace(startSymbol, out var currentFace))
            return new NavWalkResult(false, new[] { startSymbol ?? string.Empty }, 0, startSymbol ?? string.Empty);

        var current = startSymbol;
        var trajectory = new List<string> { current };
        var coords = new List<double[]> { currentFace };           // parallel: the FACE of each trajectory node (for §5.2)
        var visited = new HashSet<string>(StringComparer.Ordinal) { current };

        var reached = goalSymbol != null && string.Equals(current, goalSymbol, StringComparison.Ordinal);
        var step = 0;

        while (!reached && step < options.MaxSteps)
        {
            var decision = policy.Decide(new NavState(current, currentFace, goalFace, goalSymbol, step));

            // HALT, or nothing to emit (dead end / void) → stop. The current coordinate is the agent's final answer;
            // whether that COUNTS as reached is decided by the goal check below (a halt off the goal is an abstain).
            if (decision.Halt || decision.Target is null || decision.Target.Length == 0)
                break;

            // §5.1 ONE MOTION PRIMITIVE: the lattice lands the continuous target on the nearest decodable coordinate.
            // No landing = undecodable void with no incident structure → abstain (§8), no exception.
            if (!space.TryLand(decision.Target, out var landed, out _, out var landedFace, out _))
                break;

            step++;

            // Standing on the goal — success, regardless of whether it is also a "revisit" (the goal IS the destination).
            if (goalSymbol != null && string.Equals(landed, goalSymbol, StringComparison.Ordinal))
            {
                current = landed; currentFace = landedFace;
                trajectory.Add(landed); coords.Add(landedFace);
                reached = true;
                break;
            }

            // CYCLE / NO-PROGRESS GUARD: visited.Add is false if we landed on a node we have already stood on — which
            // includes landing on the CURRENT node (no progress). Either way the walk is not advancing toward an
            // unseen answer → abstain rather than loop forever.
            if (!visited.Add(landed))
                break;

            current = landed; currentFace = landedFace;
            trajectory.Add(landed); coords.Add(landedFace);
        }

        // §5.2 GENESIS-TICK GROWTH: on a confident successful trail, materialise every passed-through coordinate so the
        // trail the walk blazed becomes durable structure. Relevance-decay eviction prunes trails that never pay off.
        if (reached && options.MaterialiseOnSuccess)
            foreach (var coord in coords)
                space.Materialise(coord);

        return new NavWalkResult(reached, trajectory, step, current);
    }
}

/// <summary>
/// THE ORACLE POLICY (§7) — proves the walk loop with NO NN. It reads the precomputed flow field (backward Dijkstra
/// from the answer) and emits, at each node, the FACE of the field's optimal next node — a continuous target the
/// lattice lands exactly on that node. Halts on the answer; halts (→ abstain) at a dead end with no field entry.
/// This is the seam the learned policy replaces: same <see cref="INavPolicy.Decide"/>, learned target instead of read.
/// </summary>
public sealed class FlowFieldPolicy : INavPolicy
{
    private readonly PlatonicFlowField _field;
    private readonly IPlatonicSpace _space;

    public FlowFieldPolicy(PlatonicFlowField field, IPlatonicSpace space)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _space = space ?? throw new ArgumentNullException(nameof(space));
    }

    public NavDecision Decide(NavState state)
    {
        // Standing on the answer → HALT (this coordinate is the answer).
        if (string.Equals(state.CurrentSymbol, _field.Target, StringComparison.Ordinal))
            return new NavDecision(Array.Empty<double>(), Halt: true);

        // Emit the FACE of the expert next node; the lattice lands exactly on it (§5.1).
        if (_field.TryNext(state.CurrentSymbol, out var next) && _space.TryGetConceptFace(next, out var face))
            return new NavDecision(face, Halt: false);

        // No field entry here (unreachable / off the flow) → dead end → HALT → structural abstain (§8).
        return new NavDecision(Array.Empty<double>(), Halt: true);
    }
}
