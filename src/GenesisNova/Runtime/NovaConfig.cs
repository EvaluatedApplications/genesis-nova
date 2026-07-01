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
    bool MeaningOps = false,       // generative compose/analogy in the large (meaning) face
    bool DeHardcodedDispatch = false, // number↔word + compare/format/question/retrieval answered by LEARNED lexicon/cues
    bool BridgeReasoning = false,  // bridge-dimensions: when all routes abstain, infer a lacked property from embedding-neighbours
    bool RelationalFold = false,   // relational fold: multi-hop chain over fact edges (apple→fruit→food), a derivation single-hop recall can't make
    bool GeometricReasoning = false,// geometry-native derivation: read the answer from the latent geometry (edge-free, survives eviction, self-precision abstain)
    bool DirectionalReasoning = false,// trained-direction derivation: compose the trained relation direction to reach ancestors (fold-faithful)
    bool SelfReinforcement = false,// outcome-reinforce the self: at the grade stage pull the self toward correct answers, push it away from wrong ones
    bool ChunkTraversalSpeech = false)// generative speech: compose a reply by repeated-query chunk traversal (ALSO gated by TalkEnabled → non-talk paths unaffected)
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
        Learning = StableDefault.Learning with { FunctionGradientEnabled = c.FunctionGradientEnabled, SelfHealMisroutedCues = c.SelfHealMisroutedCues },
        Controller = StableDefault.Controller with { SelfConditioned = c.LivingSelf },
        Substrate = StableDefault.Substrate with { BatchedCloudGpu = c.BatchedCloudGpu, DecodeFromVoidRecovery = c.DecodeFromVoidRecovery, DerivabilityGate = c.DerivabilityGate, SelfDiscriminatedIngestion = c.SelfDiscriminatedIngestion, RelationDirectionTraining = c.RelationDirectionTraining },
        KeepCoreControl = c.KeepCoreControl,
        ConsciousField = c.ConsciousField,
        FieldTicks = c.FieldTicks,
        MeaningOps = c.MeaningOps,
        DeHardcodedDispatch = c.DeHardcodedDispatch,
        BridgeReasoning = c.BridgeReasoning,
        RelationalFold = c.RelationalFold,
        GeometricReasoning = c.GeometricReasoning,
        DirectionalReasoning = c.DirectionalReasoning,
        SelfReinforcement = c.SelfReinforcement,
        ChunkTraversalSpeech = c.ChunkTraversalSpeech,
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
        memory.GenerativeAtoms = Substrate.GenerativeAtoms; // token-as-atom + on-demand decompose/recognise (off = legacy chars)
        memory.BatchedCloudGpu = Substrate.BatchedCloudGpu; // deferred batched-GPU cloud recompute (off = per-observation scalar)
        memory.RecoverFromVoid = Substrate.DecodeFromVoidRecovery; // decode-from-the-void recovery (off = miss on an evicted/latent coordinate)
        memory.DerivabilityGate = Substrate.DerivabilityGate; // evict relation-edges derivable from a stronger path (off = keep every observed edge)
        memory.SelfDiscriminatedIngestion = Substrate.SelfDiscriminatedIngestion; // attenuate all-pairs edge formation by endpoint generality (off = flat all-pairs)
        // auto-train TransE relation-directions from real ingestion (off = inert; on = DirectionalReasoning gets live
        // directions). Cast rather than an interface member to avoid touching IPlatonicSpace (only the dialectical core trains).
        if (memory is GenesisNova.Cognition.Platonic.DialecticalSpace rdSpace) rdSpace.RelationDirectionTraining = Substrate.RelationDirectionTraining;

        // Reasoning (routing)
        inference.EdgeRoutingEnabled = Routing.EdgeRoutingEnabled;

        // Keep-core control path (PLATONIC_RECKONING.md) — pushed onto BOTH sides so train and infer perceive alike.
        inference.KeepCoreControl = KeepCoreControl;
        trainer.KeepCoreControl = KeepCoreControl;

        // Conscious field (PLATONIC_MIND.md) — inference thinks by relaxation; the classifier path is bypassed.
        inference.ConsciousField = ConsciousField;

        // De-hardcoded dispatch — number↔word via the LEARNED lexicon (no codec), and compare/to-word/to-digit/
        // question/retrieval via LEARNED cues (no word-lists). Production-on; the gym bootstraps them (warm-start).
        inference.LearnedNumberWordsOnly = DeHardcodedDispatch;
        inference.LearnedCuesOnly = DeHardcodedDispatch;

        // Generative field routes — the genesis tick cascade + large-face meaning ops (compose/analogy). The field
        // REASONS over the substrate (manufactures intermediate elements, operates in meaning-space), not just retrieves.
        inference.FieldTicksEnabled = FieldTicks;
        inference.MeaningOpsEnabled = MeaningOps;
        inference.BridgeReasoning = BridgeReasoning;   // bridge-dimensions reasoning (last-resort embedding inference)
        inference.RelationalFold = RelationalFold;     // multi-hop relational chain fold (derives super-categories by chaining fact edges)
        inference.GeometricReasoning = GeometricReasoning; // read the answer from the latent geometry (edge-free, survives eviction)
        inference.DirectionalReasoning = DirectionalReasoning; // compose the trained relation direction to reach ancestors (fold-faithful)
        inference.SelfReinforcement = SelfReinforcement; // outcome-reinforce the self at the grade stage (off = self is a passive accumulator)
        inference.ChunkTraversalSpeech = ChunkTraversalSpeech; // generative speech by repeated-query chunk traversal (also gated by TalkEnabled; non-talk unaffected)
        // The LEARNED DIRECTOR gates the risky meaning routes (compose / meaning-tick). Attached with a conservative
        // prior so it DEFAULTS to retrieval (safe — no free-form misfire) and opens the gate only as it learns.
        inference.Director = MeaningOps ? new FieldDirector(featureCount: 5, initialBias: -3.0) : null;

        // Learning (task→space mechanisms + edit head)
        inference.FunctionDisruptionEnabled = Learning.FunctionDisruptionEnabled; // Rung 1
        inference.FunctionGradientEnabled = Learning.FunctionGradientEnabled;     // Rung 2
        inference.SelfHealMisroutedCues = Learning.SelfHealMisroutedCues;         // learn from a wrong route (unlearn a hijacking cue→∘cmp edge)
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
    bool DimensionalContradiction = true, // Phase 1 dialectic: per-dimension agreement/contradiction (false = legacy scalar)
    bool GenerativeAtoms = false,         // false = legacy eager char-atoms; true = token-as-atom + decompose/recognise via ticks
    bool BatchedCloudGpu = false,         // false = per-observation scalar cloud recompute; true = deferred batched-GPU (Cloud = A·T)
    bool DecodeFromVoidRecovery = false,  // false = miss on an evicted/latent coordinate; true = re-materialise it from the codec on demand
    bool DerivabilityGate = false,        // false = keep every observed edge; true = evict edges derivable from a stronger path (transitive reduction)
    bool SelfDiscriminatedIngestion = false, // false = flat all-pairs edge formation; true = attenuate by endpoint generality (hub/glue pairs write weakly)
    bool RelationDirectionTraining = false); // false = inert; true = auto-train TransE relation-directions from observed couplings (makes DirectionalReasoning live)

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
    int RepelNeighbors = 3,
    bool SelfHealMisroutedCues = false); // learn from a WRONG ROUTE — unlearn a cue→route edge that hijacked a numeric example to compare
