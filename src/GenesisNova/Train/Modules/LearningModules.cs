using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// CORRECTNESS-GATED training (the anti-erosion gate). Predict each generated example and keep ONLY the ones the
/// model currently gets WRONG; already-correct examples are skipped (and re-checked next cycle from fresh draws, so
/// a regression auto-re-admits them). When disabled, passes the batch through unchanged. See
/// nova-correctness-gated-training / nova-erosion-under-long-training.
/// </summary>
public sealed class CorrectnessGateModule : IBatchFilterModule
{
    private readonly ILearningRuntime _runtime;
    private readonly bool _enabled;
    private readonly int _retries;
    private readonly int _gateWaitMs;
    private long _generated, _trained;

    public CorrectnessGateModule(ILearningRuntime runtime, bool enabled, int retries, int gateWaitMs)
    {
        _runtime = runtime;
        _enabled = enabled;
        _retries = retries;
        _gateWaitMs = gateWaitMs;
    }

    public string Name => "correctness-gate";

    public async Task<IReadOnlyList<(string Input, string Output)>> FilterAsync(
        IReadOnlyList<(string Input, string Output)> generated, CancellationToken ct)
    {
        _generated += generated.Count;
        if (!_enabled)
        {
            _trained += generated.Count;
            return generated;
        }

        var wrong = new List<(string Input, string Output)>(generated.Count);
        foreach (var ex in generated)
        {
            if (ct.IsCancellationRequested) break;
            GenesisPredictTaskData? r = null;
            for (var i = 0; i < _retries && r is null; i++)
                r = await _runtime.TryPredictAsync(ex.Input, gateWaitMilliseconds: _gateWaitMs);
            var got = r?.Result?.Output ?? string.Empty;
            if (GenesisGrader.Quality(got, new[] { ex.Output }, 1, false, false) > 0)
                continue; // value-correct (route-agnostic) → already mastered → don't reinforce
            wrong.Add(ex);
        }
        _trained += wrong.Count;
        return wrong;
    }

    public IReadOnlyDictionary<string, double> Metrics() => new Dictionary<string, double>
    {
        ["generated"] = _generated,
        ["trained"] = _trained,
    };
}

/// <summary>CREDIT ASSIGNMENT: feed the graded outcome back onto the edges the answer USED — a correct platonic
/// answer strengthens its edges, a wrong one weakens them (the utility pruner detaches the persistently-bad ones).
/// Neural answers carry no edges → no-op.</summary>
public sealed class CreditAssignmentModule : IProbeOutcomeModule
{
    private long _reinforced;
    public string Name => "credit-assignment";

    public void OnGradedProbe(in ProbeOutcome o)
    {
        if (o.Evidence is { Count: > 0 } ev)
            try { o.Runtime.ReinforceEvidence(ev, o.Quality > 0); _reinforced++; } catch { }
    }

    public IReadOnlyDictionary<string, double> Metrics() => new Dictionary<string, double> { ["reinforced"] = _reinforced };
}

/// <summary>RUNG 1 — task-outcome disruption (PLATONIC_BACKPROP.md): a value-WRONG answer means the geometry made
/// the wrong concept retrievable, so repel it off the anchor. Self-gated downstream (numbers/reserved skipped, and
/// the engine's FunctionDisruptionEnabled flag controls effect).</summary>
public sealed class DisruptionModule : IProbeOutcomeModule
{
    private long _disrupted;
    public string Name => "disruption";

    public void OnGradedProbe(in ProbeOutcome o)
    {
        if (!o.ValueCorrect)
            try { o.Runtime.DisruptWrongAnswer(o.Probe.Query, o.Output); _disrupted++; } catch { }
    }

    public IReadOnlyDictionary<string, double> Metrics() => new Dictionary<string, double> { ["disrupted"] = _disrupted };
}

/// <summary>SELF-HEAL a CUE MISROUTE — the missing "learn from a WRONG ROUTE" signal. On a value-WRONG probe whose
/// numeric answer was produced as a compare WORD (an arithmetic query hijacked to the compare route), contradict the
/// operator/cue that selected it so training UNLEARNS the bad cue→∘cmp relation and the route stops hijacking. Gated by
/// the engine's SelfHealMisroutedCues flag (off = no-op). Without it a corpus-contaminated "-"→∘cmp is IMMORTAL and
/// focused training can never recover the skill (the failure lives where no gradient/curriculum reaches it). See
/// nova-subtract-stuck-compare-hijack.</summary>
public sealed class CueSelfHealModule : IProbeOutcomeModule
{
    private long _healed;
    public string Name => "cue-self-heal";

    public void OnGradedProbe(in ProbeOutcome o)
    {
        if (!o.ValueCorrect)
            try { o.Runtime.HealMisroutedCue(o.Probe.Query, o.Probe.Allowed, o.Output, o.DecisionPath); _healed++; } catch { }
    }

    public IReadOnlyDictionary<string, double> Metrics() => new Dictionary<string, double> { ["healed"] = _healed };
}

/// <summary>RUNG 2 — function gradient (PLATONIC_BACKPROP.md, dormant unless FunctionGradientEnabled): descend the
/// softmax-CE function gradient so the anchor retrieves a valid task answer (arithmetic anchors are frozen numbers
/// → no-op). The engine self-gates when the flag is off.</summary>
public sealed class FunctionGradientModule : IProbeOutcomeModule
{
    private long _steps;
    public string Name => "function-gradient";

    public void OnGradedProbe(in ProbeOutcome o)
    {
        try { o.Runtime.TrainRetrievalToward(o.Probe.Query, o.Probe.Allowed); _steps++; } catch { }
    }

    public IReadOnlyDictionary<string, double> Metrics() => new Dictionary<string, double> { ["steps"] = _steps };
}
