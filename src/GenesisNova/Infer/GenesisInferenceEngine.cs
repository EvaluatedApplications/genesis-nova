using GenesisNova.Model;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Core;

namespace GenesisNova.Infer;

public sealed partial class GenesisInferenceEngine
{
    private const double DefaultNeuralBiasScale = 1.4;
    private const double MinAdaptiveBiasScale = 0.8;
    private const double MaxAdaptiveBiasScale = 2.0;
    private const double TelemetryEmaAlpha = 0.15;

    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly IPlatonicSpace _memory;
    // Learned-function store: functions learned as transform vectors T(f)=avg(embed(out)-embed(in)),
    // applied by composition (embed(x)+T(f)) in the learned-function route. Optional (null → route off).
    private readonly TransformAccumulator? _transformAccumulator;
    // Learned BINARY ops: a discovered fold/log-linear structure (e.g. mul = fold of add; c = a^α·b^β),
    // applied in the learned-function route's two-operand case. Optional (null → binary route off).
    private readonly FoldPathDiscovery? _foldPathDiscovery;
    // The learned composer's executor: the GRU's plan head selects a composition shape and this runs the
    // corresponding block tree (Compare/Branch/...) on the substrate. The blocks are the vocabulary; the
    // GRU is the composer; nothing is hardcoded per token. The interpreter carries the SHAPE REGISTRY'S
    // library so Ref blocks resolve recursively against shapes-as-Function-elements (no inline gliders).
    private readonly PlatonicGliderInterpreter _glider;
    // Named, reusable shapes as Function elements of the space (the Ref vocabulary). Built from the block
    // alphabet, not premade answers; the GRU selects which shape, the substrate executes it.
    private readonly PlatonicShapeRegistry _shapeRegistry;
    private readonly int _maxPlatonicAssistInvocations;
    private readonly object _telemetryLock = new();
    private readonly object _routeTelemetryLock = new();
    private readonly List<RouteDecisionTelemetry> _lastRouteDecisions = new();
    private double _adaptiveBiasScale = DefaultNeuralBiasScale;
    // EDGE-FOLLOWING RETRIEVAL toggle (experiment, 2026-06-22): when false, the two routes that ANSWER a query by
    // traversing relation EDGES — relation-first ("follow the strongest edge") and the multi-hop concept-chain
    // beam walk — are skipped, leaving proximity kNN (geometric retrieval, "position IS identity") as the only
    // retrieval mechanism. Relation edges are STILL created/observed during training (attraction/repulsion that
    // SHAPES the geometry is untouched); this only stops them being followed at inference. Default true (no change);
    // the live app flips it off via GenesisNovaConfig.EdgeRoutingEnabled to test whether the geometry alone routes.
    public bool EdgeRoutingEnabled { get; set; } = true;
    // Rung 1 (PLATONIC_BACKPROP.md): on a VALUE-WRONG answer, repel the produced answer from the query anchor in the
    // space — the task outcome reaching the geometry to disrupt the element that created the wrong answer. Default on.
    public bool FunctionDisruptionEnabled { get; set; } = true;
    // Rung 2 (PLATONIC_BACKPROP.md): descend the softmax-CE function gradient so the anchor's nearest neighbour
    // becomes the TASK target (pull positive, push confusers, self-scaled). Default OFF — enable to A/B vs Rung 1.
    public bool FunctionGradientEnabled { get; set; }
    // KEEP-CORE control path (PLATONIC_RECKONING.md). When true the substrate's OWN confidence drives retrieval:
    // the RELAXATION route (`reason`) becomes the primary retrieval path, the route/plan/op heads perceive the
    // DISCRIMINATIVE anchor (matching how the trainer now reinforces them), and a non-arithmetic query that no
    // platonic route can settle ABSTAINS instead of emitting a neural-decoder hallucination ("speak only from a
    // settled state"). Default OFF → byte-identical to the classifier-gated path; the desktop app turns it on.
    public bool KeepCoreControl { get; set; }
    // CONSCIOUS FIELD (PLATONIC_MIND.md / PLATONIC_CONSCIOUSNESS.md). When true the model thinks by the field
    // RELAXING to a settled state (compute → relax → abstain), with NO route/plan/op classifier — the GRU stops
    // being the boss of "neural vs platonic". This is the real architecture the docs describe; the route-ladder
    // classifier path is bypassed entirely (GenerateSingle delegates to GenerateFromField). Default off so existing
    // tests keep the classifier path; the desktop app turns it on. See GenesisInferenceEngine.Field.cs.
    public bool ConsciousField { get; set; }
    private double _telemetrySuccessEma = 0.5;
    private double _checkpointConceptEfficacyEma = 0.5;
    private InferenceTelemetryHint _trainerHint = InferenceTelemetryHint.Default;
    // The ORDERED platonic reasoning ladder (OCP). Built once in the constructor; the dispatch loop just iterates it.
    private readonly IReadOnlyList<IGenerationRoute> _platonicRoutes;

    public GenesisInferenceEngine(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        IPlatonicSpace memory,
        Func<string?>? platonicFilePathProvider = null,
        TransformAccumulator? transformAccumulator = null,
        FoldPathDiscovery? foldPathDiscovery = null,
        int maxPlatonicAssistInvocations = 3)
    {
        _tokenizer = tokenizer;
        _model = model;
        _memory = memory;
        _shapeRegistry = new PlatonicShapeRegistry(memory);
        _glider = new PlatonicGliderInterpreter(memory, _shapeRegistry.Library);
        _ = platonicFilePathProvider; // checkpoint-context concept loading retired (the path was inert); param kept for signature compatibility
        _transformAccumulator = transformAccumulator;
        _foldPathDiscovery = foldPathDiscovery;
        _maxPlatonicAssistInvocations = Math.Clamp(maxPlatonicAssistInvocations, 0, 16);

        // LEGACY CLASSIFIER FALLBACK ladder (reached only when ConsciousField is OFF; production thinks by the field —
        // see Generate/GenerateSingle). In priority order; each route abstains when it can't answer; edge-following
        // routes are dropped automatically when EdgeRoutingEnabled is false. The glider-plan/gru-query routes consult
        // the GRU plan/op heads — that classifier machinery is what the field path replaces and is NOT in production.
        _platonicRoutes = new IGenerationRoute[]
        {
            new DelegateRoute("glider-plan",         TryGenerateFromGliderPlan),         // GRU plan head → composition shape
            new DelegateRoute("expression-chain",    TryGenerateFromExpressionChain),    // multi-operator expression
            new DelegateRoute("gru-query",           TryGenerateFromGruQuery),           // single-op arithmetic homomorphism
            new DelegateRoute("learned-function",    TryGenerateFromLearnedFunction),    // transform-element by composition
            new DelegateRoute("reason",              TryGenerateFromReason),             // relaxation retrieval (keep-core; inert unless KeepCoreControl)
            new DelegateRoute("geometric-retrieval", TryGenerateFromGeometricRetrieval), // nearest concept (position = identity)
            new DelegateRoute("relation-edge",       TryGenerateFromRelationEdge, edgeFollowing: true), // strongest relation edge
            new DelegateRoute("concept-chain",       TryGenerateFromConceptChain, edgeFollowing: true), // multi-hop beam walk
        };
    }

    /// <summary>
    /// When false (the default), inference performs NO writes to any learned store — the transform
    /// accumulator is left untouched. The shared, persistent learned state is mutated ONLY by the
    /// training loop, which scopes this true around its own training-context generations. This keeps
    /// REPL/inference queries from drifting the store that every future training session loads (and
    /// from learning transforms off the model's own, possibly-wrong, output).
    /// </summary>
    public bool LearningEnabled { get; set; }

    public IReadOnlyList<RouteDecisionTelemetry> LastRouteDecisions
    {
        get
        {
            lock (_routeTelemetryLock)
                return _lastRouteDecisions.ToArray();
        }
    }

    /// <summary>The platonic reasoning ladder in priority order (DecisionPath labels) — for diagnostics and as the
    /// characterization anchor that pins the OCP route registry's order.</summary>
    public IReadOnlyList<string> PlatonicRouteOrder => _platonicRoutes.Select(r => r.Name).ToList();

    // The conversation threads ONE self — but in MEANING-space, inside conscious-field cognition (the persistent
    // _selfField, which conditions reasoning and is shaped by learning). There is no separate GRU-hidden self.
    /// <summary>The single public generation entry point. In PRODUCTION (<see cref="ConsciousField"/> = true, set by
    /// WithProductionMechanisms) this routes to <see cref="GenerateFromField"/> — field relaxation + the navigator in
    /// the ambiguous branch, with NO route/plan/op classifier. The legacy classifier ladder (<see cref="GenerateSingle"/>
    /// → <see cref="GenerateSinglePass"/>) is the FALLBACK reached ONLY when ConsciousField is off.</summary>
    public GenerationResult Generate(GenerationRequest request) => GenerateSingle(request);

    /// <summary>The PRODUCTION primary path is <see cref="GenerateFromField"/> (conscious-field cognition); everything
    /// below the ConsciousField guard — the route ladder (glider-plan / gru-query / …), the GRU plan/op heads and the
    /// label resolvers they feed — is the LEGACY CLASSIFIER FALLBACK. It runs ONLY when <see cref="ConsciousField"/> is
    /// off (default-off / A-B characterization tests), never in production. Do not extend it (CLAUDE.md: the
    /// task-classifier over the gym taxonomy is the thing to SUBTRACT); it is kept reachable purely for reversibility.</summary>
    private GenerationResult GenerateSingle(GenerationRequest request)
    {
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);

        // CONSCIOUS-FIELD COGNITION (PRODUCTION PRIMARY): bypass the entire route/plan/op classifier — the field relaxes
        // to its answer or abstains (PLATONIC_MIND.md). This IS the model's thinking when alive. Everything BELOW this
        // guard is the LEGACY CLASSIFIER FALLBACK, reached only when ConsciousField is off (default-off / A-B case).
        if (ConsciousField)
            return GenerateFromField(request);

        ResetRouteTelemetry();
        var chunkBudget = Math.Max(1, request.ChunkTokenBudget);

        if (request.MaxNewTokens <= chunkBudget)
        {
            var singlePassTokens = _tokenizer.Encode(request.Input);
            return GenerateSinglePass(request, singlePassTokens);
        }

        var generatedTokens = new List<int>();
        var decisionPaths = new List<string>();
        var usedPlatonicQuery = false;
        var usedNeuralFallback = false;
        var totalBiasCount = 0;
        var totalBiasMagnitude = 0.0;
        var totalPlatonicConfidence = 0.0;
        var platonicConfidenceCount = 0;
        var totalPlatonicHops = 0;
        string? routedTransform = null;
        string? transformIntercept = null;
        var evidence = new List<PlatonicEvidence>();

        while (generatedTokens.Count < request.MaxNewTokens)
        {
            var remainingTokens = request.MaxNewTokens - generatedTokens.Count;
            var currentBudget = Math.Min(chunkBudget, remainingTokens);
            var contextInput = BuildChunkContext(request.Input, generatedTokens, _tokenizer);
            var contextTokens = _tokenizer.Encode(contextInput);
            _model.EnsureVocabularySize(_tokenizer.VocabularySize);
            var chunkRequest = request with { Input = contextInput, MaxNewTokens = currentBudget };
            var chunkResult = GenerateSinglePass(chunkRequest, contextTokens);

            if (chunkResult.GeneratedTokens.Length == 0)
                break;

            generatedTokens.AddRange(chunkResult.GeneratedTokens);
            decisionPaths.Add(chunkResult.DecisionPath);
            usedPlatonicQuery |= chunkResult.UsedPlatonicQuery;
            usedNeuralFallback |= chunkResult.UsedNeuralFallback;
            totalBiasCount += chunkResult.AppliedBiasCount;
            totalBiasMagnitude += chunkResult.AverageBiasMagnitude * Math.Max(1, chunkResult.AppliedBiasCount);
            if (chunkResult.UsedPlatonicQuery)
            {
                totalPlatonicConfidence += chunkResult.PlatonicConfidence;
                platonicConfidenceCount++;
                totalPlatonicHops += chunkResult.PlatonicHopCount;
            }

            routedTransform ??= chunkResult.RoutedTransform;
            transformIntercept ??= chunkResult.TransformIntercept;
            if (chunkResult.Evidence is { Count: > 0 })
                evidence.AddRange(chunkResult.Evidence);

            if (chunkResult.GeneratedTokens[^1] == _tokenizer.EosTokenId)
                break;
        }

        if (generatedTokens.Count == 0)
        {
            return new GenerationResult(
                Output: string.Empty,
                GeneratedTokens: Array.Empty<int>(),
                UsedPlatonicQuery: false,
                UsedNeuralFallback: false,
                DecisionPath: "neural-token",
                PlatonicConfidence: 0.0,
                AppliedBiasCount: 0,
                AverageBiasMagnitude: 0.0,
                ChunksGenerated: 0,
                PlatonicHopCount: 0);
        }

        var decisionPath = decisionPaths.Count switch
        {
            0 => "neural-token",
            1 => decisionPaths[0],
            _ => $"chunked[{decisionPaths.Count}]: {string.Join(" -> ", decisionPaths)}"
        };

        return new GenerationResult(
            Output: _tokenizer.Decode(generatedTokens),
            GeneratedTokens: generatedTokens.ToArray(),
            UsedPlatonicQuery: usedPlatonicQuery,
            UsedNeuralFallback: usedNeuralFallback,
            DecisionPath: decisionPath,
            PlatonicConfidence: platonicConfidenceCount > 0 ? totalPlatonicConfidence / platonicConfidenceCount : 0.0,
            AppliedBiasCount: totalBiasCount,
            AverageBiasMagnitude: totalBiasCount > 0 ? totalBiasMagnitude / totalBiasCount : 0.0,
            ChunksGenerated: Math.Max(1, decisionPaths.Count),
            PlatonicHopCount: totalPlatonicHops,
            RoutedTransform: routedTransform,
            TransformIntercept: transformIntercept,
            Evidence: CollapseEvidence(evidence));
    }

    private GenerationResult GenerateSinglePass(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens)
    {
        // PERCEPTION ROUTING (SPACE_AWARE_GRU.md §I): when enabled, let the route head also read a TARGET-AGNOSTIC
        // perception of the query anchor ("does the space look like it can answer this?") so platonic-vs-neural is
        // decided from perceived retrievability, not tokens alone. Default-off → unchanged token-only routing.
        double[]? routePerception = null;
        if (_model.PerceptionRouting && inputTokens.Count > 0)
        {
            var toks = _tokenizer.Decode(inputTokens).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (toks.Length > 0)
            {
                // Bubble EARNED transform reliability up to the route head (gated by TransformReliabilityRouting):
                // a proven transform tilts the router toward the function/platonic route; a noisy one doesn't.
                var transformReliability = _model.TransformReliabilityRouting && _transformAccumulator is not null
                    ? _transformAccumulator.BestReliabilityUcb()
                    : 0.0;
                // KEEP-CORE: perceive the DISCRIMINATIVE anchor (the content cue, not the last surface token) so the
                // route head reads at decode time the same region the trainer reinforced it on. Falls back to the
                // last token when no known concept is present (early training / numeric inputs).
                var anchorTok = KeepCoreControl ? (DiscriminativeAnchorToken(inputTokens) ?? toks[^1]) : toks[^1];
                routePerception = _memory.ComputeRoutePerception(anchorTok, transformReliability);
            }
        }
        var (routeId, routeConfidence) = _model.PredictRoute(inputTokens, routePerception);

        // Mode 2 (platonic-assisted reasoning): generate neurally but invoke the platonic space
        // mid-generation as an internal scratchpad. Falls back to pure neural if no sub-step fires.
        if (routeId == 2)
        {
            // Arithmetic is resolved by the GRU-CONSTRUCTED query (op CLASSIFIED from context + operand
            // selection), not a hardcoded symbol→op parser. Removing the compact regex (2026-06-14): it
            // forced ambiguous tokens to a fixed math meaning ("x"⇒multiply always), which is wrong when
            // "x" is a variable/word; the op head decides from context (digits flanking "x" ⇒ multiply;
            // otherwise abstain) and the homomorphism computes.
            if (TryGenerateFromGruQuery(request, out var gruQuery))
            {
                RecordRouteDecision(routeId, 1, true, true, true, 1, gruQuery.DecisionPath, routeConfidence);
                return gruQuery;
            }

            var assisted = GenerateNeuralWithPlatonicAssist(request, inputTokens);
            RecordRouteDecision(
                routeId, 2,
                platonicAttempted: assisted.PlatonicAssistInvocations > 0,
                platonicSucceeded: assisted.PlatonicAssistFired > 0,
                predictedRouteProducedFinalAnswer: assisted.PlatonicAssistFired > 0,
                supervisionLabel: 2,
                assisted.DecisionPath, routeConfidence,
                assisted.PlatonicAssistInvocations, assisted.PlatonicAssistFired);
            return assisted;
        }

        // Mode 1 (platonic-direct): answer straight from a platonic transform/query.
        if (routeId == 1)
        {
            if (TryGenerateFromPlatonicPlan(request, out var platonic))
            {
                RecordRouteDecision(routeId, 1, true, true, true, 1, platonic.DecisionPath, routeConfidence);
                return platonic;
            }
            // KEEP-CORE: every platonic route abstained and this is the non-arithmetic branch (arithmetic is
            // routeId 2) — the field has nothing settled to say. ABSTAIN rather than invent via the neural decoder
            // (PLATONIC_RECKONING.md: the neural decoder as a primary answer path is a throw; speak only from a
            // settled state). The substrate still LEARNS this example through training's observation writes.
            if (KeepCoreControl)
            {
                var abstain = EmptyAbstention();
                RecordRouteDecision(routeId, 0, true, false, false, 0, abstain.DecisionPath, routeConfidence);
                return abstain;
            }
            // Platonic route attempted but no tool fired — fall through to neural.
            var fallback = GenerateNeuralTokens(request, inputTokens, neuralFallback: true);
            RecordRouteDecision(routeId, 0, true, false, false, 0, fallback.DecisionPath, routeConfidence);
            return fallback;
        }

        // Mode 0 (neural-only).
        var neural = GenerateNeuralTokens(request, inputTokens, neuralFallback: false);
        RecordRouteDecision(routeId, 0, false, false, true, 0, neural.DecisionPath, routeConfidence);
        return neural;
    }

    /// <summary>
    /// Rung 1 task-outcome disruption (PLATONIC_BACKPROP.md): given a VALUE-WRONG (query → output), repel the
    /// produced answer concept from the query's discriminative anchor(s) in the space. General over routes — the
    /// answer may have come from geometric retrieval, a relation edge, or the neural fallback; either way the
    /// geometry that made it retrievable gets disrupted. The caller MUST gate on value-incorrectness (a neural but
    /// value-correct answer must not be disrupted). Single-token answers only; numbers/reserved are skipped in the
    /// space. No-op when disabled.
    /// </summary>
    public void DisruptWrongAnswer(string query, string output)
    {
        if (!FunctionDisruptionEnabled || string.IsNullOrWhiteSpace(query) || string.IsNullOrWhiteSpace(output))
            return;
        var answer = output.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (answer.Length != 1) // a single emitted concept is what got mis-retrieved; multi-token outputs aren't a retrieval
            return;
        foreach (var anchor in ExtractConceptAnchors(query))
            _memory.DisruptAssociation(anchor, answer[0]);
    }

    /// <summary>
    /// Rung 2 function gradient (PLATONIC_BACKPROP.md): descend softmax-CE so the query anchor's nearest neighbour
    /// becomes a valid TASK answer — pull the target toward the anchor, push the current confusers away, self-scaled.
    /// Runs on EVERY graded retrieval probe (right or wrong: it reinforces a correct target and corrects a wrong one);
    /// arithmetic anchors/targets are numbers (frozen) so they no-op. Distractors = the anchor's live nearest
    /// neighbours. No-op when disabled.
    /// </summary>
    public void TrainRetrievalToward(string query, IReadOnlyList<string> allowedAnswers)
    {
        if (!FunctionGradientEnabled || string.IsNullOrWhiteSpace(query) || allowedAnswers is null)
            return;
        var anchors = ExtractConceptAnchors(query);
        if (anchors.Count == 0)
            return;
        string? target = null;
        foreach (var ans in allowedAnswers)
        {
            var t = ans?.Trim();
            if (!string.IsNullOrEmpty(t) && _memory.ContainsConcept(t)) { target = t; break; } // first answer that's a real concept
        }
        if (target is null)
            return;
        foreach (var anchor in anchors)
        {
            var distractors = _memory.GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 8)
                .Select(x => x.Symbol)
                .Where(s => !string.Equals(s, target, StringComparison.OrdinalIgnoreCase))
                .ToList();
            _memory.FunctionGradientStep(anchor, target, distractors);
        }
    }

    private static string BuildChunkContext(string baseInput, IReadOnlyList<int> generatedTokens, IGenesisTokenizer tokenizer)
    {
        if (generatedTokens.Count == 0)
            return baseInput;

        var generatedText = tokenizer.Decode(generatedTokens);
        return $"{baseInput}\n{generatedText}";
    }

    // NOTE: the symbol→op COMPACT ARITHMETIC parser (TryGenerateFromDiscoveredTransform,
    // TryGenerateFromArithmeticQuery, TryResolveArithmeticQuery, TryParseArithmeticInput) was removed
    // 2026-06-14. It hardcoded ambiguous operator tokens to a single math meaning ("x"⇒multiply always),
    // which is wrong when "x" is a variable/word. Arithmetic now flows through TryGenerateFromGruQuery:
    // the GRU op head CLASSIFIES the operation from context and the homomorphism (TryFaceArithmetic)
    // computes — so "x" only means multiply when context (flanking operands) says so. See PLATONIC_SPACE.md.

    // RELATION-FIRST retrieval. For a single-concept query, return the strongest learned relational
    // neighbour ("1"→"one", "apple"→"fruit") — the STABLE substrate fact the lesson established. (A
    // geometric-first variant was tried and empirically refuted: the migrated numeric face is unstable
    // under contrastive repulsion and the semantic face is lexical, so geometry mis-retrieved; the edge
    // is the robust mechanism.) Single-concept only; arithmetic / glider capabilities are handled earlier.
    private const double RelationFirstMinConfidence = 0.5;

    // GEOMETRIC retrieval gate: confidence = 1/(1+faceDistance) in the semantic face. Above this, position
    // alone is a trustworthy answer (content addressing); below, defer to the relation edge.
    private const double GeometricMinConfidence = 0.55;

    // REASON (relaxation) retrieval gate (keep-core): confidence is the cosine overlap of the relaxed query
    // field with the winning concept cloud. Lower scale than the geometric 1/(1+dist), so a distinct threshold;
    // the route ALSO requires Thought.Settled (the raw query had a near basin) before this even applies.
    private const double ReasonMinConfidence = 0.42;

    // KEEP-CORE abstention: an honest "nothing settled" result — no neural hallucination. Mirrors the empty
    // generation result; the DecisionPath names it so telemetry/tests can see the substrate declined to answer.
    private static GenerationResult EmptyAbstention() => new(
        Output: string.Empty,
        GeneratedTokens: Array.Empty<int>(),
        UsedPlatonicQuery: false,
        UsedNeuralFallback: false,
        DecisionPath: "platonic-abstain",
        PlatonicConfidence: 0.0,
        AppliedBiasCount: 0,
        AverageBiasMagnitude: 0.0,
        ChunksGenerated: 0,
        PlatonicHopCount: 0);

    // Per-generation IMMUTABLE bias context. The query bias derives ONLY from the input tokens
    // (BuildContextConcepts excludes generated tokens) and the vocabulary's concept-bearing tokens —
    // both invariant across a single generation's decode loop — so they are computed ONCE here and
    // reused every step. Only the repetition guard (drop the positive boost for already-emitted tokens)
    // is per-step. Token ids are stored in ASCENDING order to match the old full-vocab scan exactly.
    private sealed record TokenBiasContext(
        IReadOnlyList<string> ContextConcepts,
        IReadOnlyList<(int Token, string Candidate)> ConceptTokens);
}

public sealed record RouteDecisionTelemetry(
    int PredictedRoute,
    int FinalRoute,
    bool PlatonicRouteAttempted,
    bool PlatonicRouteSucceeded,
    bool PredictedRouteProducedFinalAnswer,
    int? SupervisionLabel,
    string DecisionPath,
    double RouteConfidence,
    // Introspective telemetry (additive). For the platonic-assisted reasoning route these record
    // how many mid-generation platonic tool calls were attempted and how many fired.
    int PlatonicAssistInvocations = 0,
    int PlatonicAssistFired = 0);
