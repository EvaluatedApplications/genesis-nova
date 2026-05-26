namespace GenesisNova.Axioms;

public sealed record AxiomBridgeEntry(
    GenesisAxiom Axiom,
    string MlInterpretation,
    string ArchitectureConsequence,
    string ObjectiveSignal,
    string EvaluationMetric);

