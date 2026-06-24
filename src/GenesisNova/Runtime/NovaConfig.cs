using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

/// <summary>
/// THE single control surface for the ML MECHANISMS — one place to set (and, with <see cref="NovaTelemetry"/>,
/// observe) every toggle that used to be scattered across ~7 files (model <c>internal</c>s, the inference engine,
/// the platonic space, the trainer, the orchestrator options). Grouped by layer; <see cref="StableDefault"/> is the
/// single source of truth for "what is on by default" — the known-stable baseline. <see cref="GenesisRuntimeState"/>
/// calls <see cref="ApplyTo"/> ONCE to push these onto the live subsystems, replacing the old per-field assignments.
///
/// Scope: this owns the mechanism toggles only. Pure infrastructure (VRAM, parallelism, persistence, dimensions)
/// stays on <see cref="GenesisNovaConfig"/>; <see cref="FromLegacy"/> carries across the few mechanism flags that
/// historically lived there so existing call sites keep their exact behaviour during the migration.
/// </summary>
public sealed record NovaConfig(
    SubstrateOptions Substrate,
    ControllerOptions Controller,
    RoutingOptions Routing,
    LearningOptions Learning,
    bool KeepCoreControl = false, // PLATONIC_RECKONING.md keep-core: substrate-confidence routing + seam fix + abstention
    bool ConsciousField = false,  // PLATONIC_MIND.md: think by field-relaxation, bypass the route/plan/op classifier
    bool FieldTicks = false,       // the genesis tick cascade (numeric + meaning) runs queries as multi-step derivations
    bool MeaningOps = false)       // generative compose/analogy in the large (meaning) face
{
    /// <summary>The default-on/off profile — every field is set to today's live value, so applying it is a no-op
    /// against the historical scattered defaults. Change a default HERE, not in seven places.</summary>
    public static NovaConfig StableDefault { get; } = new(
        new SubstrateOptions(),
        new ControllerOptions(),
        new RoutingOptions(),
        new LearningOptions());

    /// <summary>Carry across the mechanism flags that historically lived on <see cref="GenesisNovaConfig"/> so the
    /// migration is behaviour-identical (the app still drives them through the legacy config for now).</summary>
    public static NovaConfig FromLegacy(GenesisNovaConfig c) => StableDefault with
    {
        Routing = StableDefault.Routing with { EdgeRoutingEnabled = c.EdgeRoutingEnabled },
        Learning = StableDefault.Learning with { FunctionGradientEnabled = c.FunctionGradientEnabled },
        Controller = StableDefault.Controller with { SelfConditioned = c.LivingSelf },
        KeepCoreControl = c.KeepCoreControl,
        ConsciousField = c.ConsciousField,
        FieldTicks = c.FieldTicks,
        MeaningOps = c.MeaningOps,
    };

    /// <summary>Push every mechanism toggle onto the live subsystems in ONE place (replaces the old scattered
    /// per-field assignments in <see cref="GenesisRuntimeState"/>).</summary>
    public void ApplyTo(GenesisNeuralModel model, IPlatonicSpace memory, GenesisInferenceEngine inference, GenesisTrainer trainer)
    {
        // Controller (GRU heads)
        model.PerceptionRouting = Controller.PerceptionRouting;
        model.TransformReliabilityRouting = Controller.TransformReliabilityRouting;
        model.PerceptionQuery = Controller.PerceptionQuery;
        model.PerceptionPlan = Controller.PerceptionPlan;
        model.RouteClassBalanceEnabled = Controller.ClassBalanceRoute;
        model.QueryOpClassBalanceEnabled = Controller.ClassBalanceOp;
        model.PlanClassBalanceEnabled = Controller.ClassBalancePlan;
        inference.SelfConditionsCognition = Controller.SelfConditioned; // the meaning-space self conditions reasoning (PLATONIC_CONSCIOUSNESS.md)

        // Substrate (platonic space)
        memory.UseInfoNceRepulsion = Substrate.UseInfoNceRepulsion;
        memory.DimensionalContradiction = Substrate.DimensionalContradiction;

        // Reasoning (routing)
        inference.EdgeRoutingEnabled = Routing.EdgeRoutingEnabled;

        // Keep-core control path (PLATONIC_RECKONING.md) — pushed onto BOTH sides so train and infer perceive alike.
        inference.KeepCoreControl = KeepCoreControl;
        trainer.KeepCoreControl = KeepCoreControl;

        // Conscious field (PLATONIC_MIND.md) — inference thinks by relaxation; the classifier path is bypassed.
        inference.ConsciousField = ConsciousField;

        // Generative field routes — the genesis tick cascade + large-face meaning ops (compose/analogy). The field
        // REASONS over the substrate (manufactures intermediate elements, operates in meaning-space), not just retrieves.
        inference.FieldTicksEnabled = FieldTicks;
        inference.MeaningOpsEnabled = MeaningOps;

        // Learning (task→space mechanisms + edit head)
        inference.FunctionDisruptionEnabled = Learning.FunctionDisruptionEnabled; // Rung 1
        inference.FunctionGradientEnabled = Learning.FunctionGradientEnabled;     // Rung 2
        trainer.RequirePlatonicForCorrect = Learning.RequirePlatonicForCorrect;
        trainer.SpaceAwareEdit = Learning.SpaceAwareEdit;
        trainer.PerceptionEdit = Learning.PerceptionEdit;
        trainer.RepelNeighbors = Learning.RepelNeighbors;
    }
}

/// <summary>Platonic-substrate mechanism toggles. (Repulsion/maintenance rates remain consts inside the space for
/// now; wiring those through here is a follow-on — the toggle that matters today is the repulsion MODE.)</summary>
public sealed record SubstrateOptions(
    bool UseInfoNceRepulsion = false,    // false = manual constant-step repulsion (live); true = InfoNCE push
    bool DimensionalContradiction = true); // Phase 1 dialectic: per-dimension agreement/contradiction (false = legacy scalar)

/// <summary>GRU controller / decision-head toggles.</summary>
public sealed record ControllerOptions(
    bool PerceptionRouting = true,
    bool TransformReliabilityRouting = true,
    bool PerceptionQuery = true,
    bool PerceptionPlan = true,
    bool ClassBalanceRoute = true,
    bool ClassBalanceOp = true,
    bool ClassBalancePlan = true,
    bool SelfConditioned = false); // the meaning-space self conditions cognition (PLATONIC_CONSCIOUSNESS.md) — off by default; the app turns it on

/// <summary>Inference routing toggles.</summary>
public sealed record RoutingOptions(
    bool EdgeRoutingEnabled = true);     // false = proximity-kNN-only retrieval (edge-following routes dropped)

/// <summary>Task→space learning mechanisms + edit-head knobs. (TrainOnFailureOnly lives on the orchestrator's
/// Options for now and is surfaced here for completeness; it is wired through the orchestrator, not ApplyTo.)</summary>
public sealed record LearningOptions(
    bool FunctionDisruptionEnabled = true,   // Rung 1 — repel a value-wrong answer from the anchor
    bool FunctionGradientEnabled = false,    // Rung 2 — softmax-CE function gradient (app overrides to true)
    bool RequirePlatonicForCorrect = true,
    bool SpaceAwareEdit = true,
    bool PerceptionEdit = true,
    int RepelNeighbors = 3);
