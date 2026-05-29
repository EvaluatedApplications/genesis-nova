namespace GenesisNova.Runtime;

public sealed record PlatonicActivatedNode(
    string Name,
    double Score,
    int ObservationCount,
    bool IsAnchor);

public sealed record PlatonicActivatedEdge(
    string Left,
    string Right,
    double Score,
    double Contradiction,
    int ObservationCount);

public sealed record PlatonicActivationView(
    string Input,
    string[] InputTokens,
    string[] Anchors,
    PlatonicActivatedNode[] Nodes,
    PlatonicActivatedEdge[] Edges);
