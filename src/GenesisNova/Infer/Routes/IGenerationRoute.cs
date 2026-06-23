namespace GenesisNova.Infer;

/// <summary>
/// One arm of the platonic reasoning ladder (Open/Closed). Each route ATTEMPTS to answer and ABSTAINS — returns
/// false — when it can't, so the engine simply tries the routes in order. Adding, removing, or reordering a route
/// is a change to the registry LIST, never to the dispatch loop. <see cref="Name"/> is the DecisionPath label the
/// route emits; <see cref="EdgeFollowing"/> marks routes that traverse relation EDGES (dropped when the engine is
/// in proximity-kNN-only mode, i.e. EdgeRoutingEnabled = false).
/// </summary>
public interface IGenerationRoute
{
    string Name { get; }
    bool EdgeFollowing { get; }
    bool TryGenerate(GenerationRequest request, out GenerationResult result);
}

/// <summary>
/// Adapter that exposes an existing engine route METHOD as an <see cref="IGenerationRoute"/>. This is the strangler
/// seam: the route method bodies stay on <see cref="GenesisInferenceEngine"/> for now, while the LADDER itself
/// becomes a declarative, ordered, filterable registry. A later pass can relocate each body into its own route type
/// without touching the engine's dispatch.
/// </summary>
internal sealed class DelegateRoute : IGenerationRoute
{
    public delegate bool RouteAttempt(GenerationRequest request, out GenerationResult result);

    private readonly RouteAttempt _attempt;

    public DelegateRoute(string name, RouteAttempt attempt, bool edgeFollowing = false)
    {
        Name = name;
        _attempt = attempt;
        EdgeFollowing = edgeFollowing;
    }

    public string Name { get; }
    public bool EdgeFollowing { get; }
    public bool TryGenerate(GenerationRequest request, out GenerationResult result) => _attempt(request, out result);
}
