using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Cognition;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// A named, individually-measurable LEARNING MECHANISM. The orchestrator iterates the REGISTERED set at each hook
/// point instead of hardcoding each mechanism inline (Open/Closed) — adding a mechanism is adding a module, never
/// editing the training loop. Stage-specific hooks are split into sub-interfaces (ISP): a module implements only
/// the stage it participates in. <see cref="Metrics"/> is what the module DID (it feeds the telemetry surface).
/// </summary>
public interface ILearningModule
{
    string Name { get; }
    IReadOnlyDictionary<string, double> Metrics();
}

/// <summary>PRE-TRAIN batch transform — e.g. correctness-gating (predict each example, keep only the wrong ones).</summary>
public interface IBatchFilterModule : ILearningModule
{
    Task<IReadOnlyList<(string Input, string Output)>> FilterAsync(
        IReadOnlyList<(string Input, string Output)> generated, CancellationToken ct);
}

/// <summary>Reacts to ONE graded probe — credit assignment, Rung-1 disruption, Rung-2 function gradient.</summary>
public interface IProbeOutcomeModule : ILearningModule
{
    void OnGradedProbe(in ProbeOutcome outcome);
}

/// <summary>Everything a probe-outcome module needs about one graded probe (built once, passed to each module).</summary>
public readonly record struct ProbeOutcome(
    ILearningRuntime Runtime,
    TrainingProbe Probe,
    string Output,
    bool Neural,
    bool ValueCorrect,
    double Quality,
    IReadOnlyList<PlatonicEvidence> Evidence,
    string DecisionPath);
