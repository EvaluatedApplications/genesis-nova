namespace GenesisNova.Axioms;

public static class GenesisAxiomBridge
{
    public static IReadOnlyList<AxiomBridgeEntry> Entries { get; } =
    [
        new(
            GenesisAxiom.G1ConsciousSelection,
            MlInterpretation: "Learnable control policy over compute.",
            ArchitectureConsequence: "Router head chooses expert/tool path.",
            ObjectiveSignal: "Route policy loss.",
            EvaluationMetric: "Route entropy and route-task alignment"),
        new(
            GenesisAxiom.G2NonContradiction,
            MlInterpretation: "Outputs and latent states must remain logically consistent.",
            ArchitectureConsequence: "Consistency checks and contradiction-aware training pairs.",
            ObjectiveSignal: "Contradiction penalty.",
            EvaluationMetric: "Contradiction rate on paired prompts"),
        new(
            GenesisAxiom.G3GenerativeObservation,
            MlInterpretation: "Observing context should generate new reusable structure.",
            ArchitectureConsequence: "Autoregressive decoding + optional world-state update.",
            ObjectiveSignal: "Token likelihood and novelty regularization.",
            EvaluationMetric: "Task success plus valid novel generations"),
        new(
            GenesisAxiom.G4Conservation,
            MlInterpretation: "Generated distinctions preserve balanced representational mass.",
            ArchitectureConsequence: "Paired latent channels with anti-symmetric regularization.",
            ObjectiveSignal: "Conservation drift penalty.",
            EvaluationMetric: "Average conservation drift"),
        new(
            GenesisAxiom.G5RecursiveAvailability,
            MlInterpretation: "Any learned concept can be revisited by future reasoning.",
            ArchitectureConsequence: "Memory is re-addressable through retrieval/attention.",
            ObjectiveSignal: "Recall and retrieval consistency loss.",
            EvaluationMetric: "Recall hit-rate over stored concepts"),
        new(
            GenesisAxiom.G6Irreversibility,
            MlInterpretation: "Knowledge history is monotonic; corrections supersede, not erase.",
            ArchitectureConsequence: "Append-only memory/event stream with versioning.",
            ObjectiveSignal: "Monotonic update regularization.",
            EvaluationMetric: "Overwrite rate and supersession integrity")
    ];
}

