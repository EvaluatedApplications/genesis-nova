using GenesisNova.Model;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Core;
using System.Text.Json;

namespace GenesisNova.Infer;

public sealed class GenesisInferenceEngine
{
    private const double DefaultNeuralBiasScale = 1.4;
    private const double MinAdaptiveBiasScale = 0.8;
    private const double MaxAdaptiveBiasScale = 2.0;
    private const double TelemetryEmaAlpha = 0.15;
    private const double CheckpointDisableThreshold = 0.30;
    private const double CheckpointEnableThreshold = 0.52;

    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly PlatonicSpaceMemory _memory;
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
    private readonly Func<string?>? _platonicFilePathProvider;
    private readonly int _maxPlatonicAssistInvocations;
    private readonly object _platonicFileCacheLock = new();
    private string? _cachedPlatonicFilePath;
    private DateTime _cachedPlatonicFileWriteUtc;
    private IReadOnlyCollection<string>? _cachedPlatonicFileConcepts;
    private readonly object _telemetryLock = new();
    private readonly object _routeTelemetryLock = new();
    private readonly List<RouteDecisionTelemetry> _lastRouteDecisions = new();
    private double _adaptiveBiasScale = DefaultNeuralBiasScale;
    private double _telemetrySuccessEma = 0.5;
    private double _checkpointConceptEfficacyEma = 0.5;
    private bool _checkpointContextConceptsEnabled = true;
    private InferenceTelemetryHint _trainerHint = InferenceTelemetryHint.Default;

    public GenesisInferenceEngine(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicSpaceMemory memory,
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
        _platonicFilePathProvider = platonicFilePathProvider;
        _transformAccumulator = transformAccumulator;
        _foldPathDiscovery = foldPathDiscovery;
        _maxPlatonicAssistInvocations = Math.Clamp(maxPlatonicAssistInvocations, 0, 16);
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

    public GenerationResult Generate(GenerationRequest request)
        => GenerateSingle(request);

    public void ReportTelemetryOutcome(bool isSuccessful, GenerationResult result)
    {
        lock (_telemetryLock)
        {
            _telemetrySuccessEma = BlendEma(_telemetrySuccessEma, isSuccessful ? 1.0 : 0.0, TelemetryEmaAlpha);

            var confidenceSignal = Math.Clamp(result.PlatonicConfidence, 0.0, 1.0);
            var fallbackSignal = result.UsedNeuralFallback ? 0.0 : 1.0;
            var biasSignal = result.AppliedBiasCount > 0
                ? Math.Clamp(result.AverageBiasMagnitude / 0.9, 0.0, 1.0)
                : 0.5;
            var routeSignal = result.UsedPlatonicQuery ? confidenceSignal : 0.5;
            var efficacySignal = Math.Clamp(
                (_telemetrySuccessEma * 0.60) +
                (fallbackSignal * 0.20) +
                (routeSignal * 0.10) +
                (biasSignal * 0.10),
                0.0,
                1.0);

            _checkpointConceptEfficacyEma = BlendEma(_checkpointConceptEfficacyEma, efficacySignal, TelemetryEmaAlpha);

            var scaleTarget = DefaultNeuralBiasScale * (0.75 + (0.55 * _checkpointConceptEfficacyEma));
            _adaptiveBiasScale = Math.Clamp(scaleTarget, MinAdaptiveBiasScale, MaxAdaptiveBiasScale);

            if (_checkpointContextConceptsEnabled && _checkpointConceptEfficacyEma < CheckpointDisableThreshold)
                _checkpointContextConceptsEnabled = false;
            else if (!_checkpointContextConceptsEnabled && _checkpointConceptEfficacyEma > CheckpointEnableThreshold)
                _checkpointContextConceptsEnabled = true;
        }
    }

    public void ApplyTelemetryHint(InferenceTelemetryHint hint)
    {
        lock (_telemetryLock)
            _trainerHint = new InferenceTelemetryHint(
                BiasScale: Math.Clamp(hint.BiasScale, 0.7, 1.4),
                EnableContextBias: hint.EnableContextBias);
    }

    private GenerationResult GenerateSingle(GenerationRequest request)
    {
        ResetRouteTelemetry();
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
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
                routePerception = _memory.ComputeRoutePerception(toks[^1], transformReliability); // operand = last token
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

    private void ResetRouteTelemetry()
    {
        lock (_routeTelemetryLock)
            _lastRouteDecisions.Clear();
    }

    private void RecordRouteDecision(
        int predictedRoute,
        int finalRoute,
        bool platonicAttempted,
        bool platonicSucceeded,
        bool predictedRouteProducedFinalAnswer,
        int? supervisionLabel,
        string decisionPath,
        double routeConfidence,
        int platonicAssistInvocations = 0,
        int platonicAssistFired = 0)
    {
        lock (_routeTelemetryLock)
        {
            _lastRouteDecisions.Add(new RouteDecisionTelemetry(
                PredictedRoute: predictedRoute,
                FinalRoute: finalRoute,
                PlatonicRouteAttempted: platonicAttempted,
                PlatonicRouteSucceeded: platonicSucceeded,
                PredictedRouteProducedFinalAnswer: predictedRouteProducedFinalAnswer,
                SupervisionLabel: supervisionLabel,
                DecisionPath: decisionPath,
                RouteConfidence: routeConfidence,
                PlatonicAssistInvocations: platonicAssistInvocations,
                PlatonicAssistFired: platonicAssistFired));
        }
    }

    /// <summary>
    /// Mode 2: platonic-assisted reasoning. Generates neurally in bounded segments and, between
    /// segments, may invoke the platonic space as an internal scratchpad: it scans the working
    /// context (prompt + tokens so far) for a platonic-resolvable arithmetic sub-step, resolves it
    /// EXACTLY via the log/poly face homomorphism (TryResolveAssistSubResult), tokenizes the
    /// sub-result, INJECTS it back into the working sequence, then continues neural decode
    /// conditioned on it.
    ///
    /// Bounded by <see cref="_maxPlatonicAssistInvocations"/> tool calls per generation. If nothing
    /// fires (no resolvable sub-step, or low decode confidence) it degrades to pure neural output —
    /// never hangs and never injects garbage.
    /// </summary>
    private GenerationResult GenerateNeuralWithPlatonicAssist(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens)
    {
        var (adaptiveBiasScale, checkpointContextEnabled) = GetAdaptiveTelemetryState();
        var checkpointContextConcepts = checkpointContextEnabled
            ? LoadCheckpointContextConcepts()
            : Array.Empty<string>();

        var generated = new List<int>(Math.Max(1, request.MaxNewTokens));
        var evidence = new List<PlatonicEvidence>();
        var totalBiasCount = 0;
        var biasMagnitudeSum = 0.0;
        var prev = _tokenizer.BosTokenId;

        var assistInvocations = 0;
        var assistFired = 0;
        var injectedAny = false;
        var totalAssistConfidence = 0.0;
        var alreadyResolved = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        // Re-invoke the platonic tool roughly every (budget / (cap+1)) decoded tokens so calls are
        // spread across the generation rather than all bunched at the start.
        var maxInvocations = _maxPlatonicAssistInvocations;
        var segment = maxInvocations > 0
            ? Math.Max(1, request.MaxNewTokens / (maxInvocations + 1))
            : int.MaxValue;
        var nextAssistAt = segment;

        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            // Platonic tool-invocation point (bounded): try to resolve a sub-step from the current
            // working context and inject its exact result before continuing neural decode.
            if (maxInvocations > 0 && assistInvocations < maxInvocations && generated.Count >= nextAssistAt)
            {
                nextAssistAt += segment;
                assistInvocations++;
                var workingText = BuildChunkContext(request.Input, generated, _tokenizer);
                if (TryResolveAssistSubResult(workingText, alreadyResolved, out var subResult, out var subConfidence))
                {
                    var injectTokens = _tokenizer.Encode(subResult, addEos: false);
                    var room = request.MaxNewTokens - generated.Count;
                    if (injectTokens.Length > 0 && room > 0)
                    {
                        foreach (var t in injectTokens.Take(room))
                        {
                            generated.Add(t);
                            prev = t;
                        }
                        assistFired++;
                        injectedAny = true;
                        totalAssistConfidence += subConfidence;
                        // Re-align the loop counter with the now-larger sequence.
                        i = generated.Count - 1;
                        if (generated.Count >= request.MaxNewTokens)
                            break;
                    }
                }
            }

            double biasMagnitude;
            IReadOnlyDictionary<int, double>? biases;
            IReadOnlyList<PlatonicEvidence> stepEvidence;
            if (checkpointContextConcepts.Count > 0)
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, checkpointContextConcepts, out biasMagnitude, out stepEvidence);
            else
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, out biasMagnitude, out stepEvidence);

            var next = _model.PredictNextToken(
                inputTokens,
                prev,
                stepIndex: i,
                disallowToken: i == 0 ? _tokenizer.EosTokenId : null,
                penalizedTokens: generated,
                repetitionPenalty: 0.35,
                tokenBiases: biases,
                stopToken: _tokenizer.EosTokenId);
            if (biases is not null)
            {
                totalBiasCount += biases.Count;
                biasMagnitudeSum += biasMagnitude;
                if (stepEvidence.Count > 0)
                    evidence.AddRange(stepEvidence);
            }
            generated.Add(next);
            if (next == _tokenizer.EosTokenId)
                break;
            prev = next;
        }

        var avgAssistConfidence = assistFired > 0 ? totalAssistConfidence / assistFired : 0.0;
        return new GenerationResult(
            Output: _tokenizer.Decode(generated),
            GeneratedTokens: generated.ToArray(),
            UsedPlatonicQuery: injectedAny,
            UsedNeuralFallback: !injectedAny,
            DecisionPath: injectedAny
                ? $"neural+platonic-assist[{assistFired}/{assistInvocations}]"
                : "neural-token+platonic-bias(assist-miss)",
            PlatonicConfidence: avgAssistConfidence,
            AppliedBiasCount: totalBiasCount,
            AverageBiasMagnitude: totalBiasCount > 0 ? biasMagnitudeSum / Math.Max(1, generated.Count) : 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: assistFired,
            Evidence: CollapseEvidence(evidence),
            PlatonicAssistInvocations: assistInvocations,
            PlatonicAssistFired: assistFired);
    }

    /// <summary>
    /// Resolves the arithmetic sub-step in <paramref name="workingText"/> through the SAME
    /// GRU-constructed query path the direct route uses (<see cref="TryGenerateFromGruQuery"/>): the GRU
    /// op head classifies the operation from the scratchpad's CONTEXT and the face homomorphism computes
    /// it exactly. NO hardcoded symbol parser (the regex / "x"⇒multiply switch was removed 2026-06-14) —
    /// the operator is never read off a pattern, so the assist disambiguates the op the same learned way
    /// as every other arithmetic path. Returns the injected sub-result (" = N ") and a decode-consistency
    /// confidence; false when the GRU abstains, nothing resolves, or this value was already injected.
    /// </summary>
    private bool TryResolveAssistSubResult(
        string workingText,
        HashSet<string> alreadyResolved,
        out string subResult,
        out double confidence)
    {
        subResult = string.Empty;
        confidence = 0.0;
        if (string.IsNullOrWhiteSpace(workingText))
            return false;

        // The whole working context IS the query: the GRU selects the operands + op and the homomorphism
        // computes. Abstains (false) on non-arithmetic scratchpads or low decode quality — same gates as
        // the direct route, so the assist injects only what the learned query path would itself produce.
        if (!TryGenerateFromGruQuery(new GenerationRequest(workingText, 6), out var resolved))
            return false;

        var value = resolved.Output.Trim();
        if (value.Length == 0 || !alreadyResolved.Add(value))
            return false; // nothing resolved, or this value was already injected this generation

        // Inject as " = <result> " so the neural decoder continues conditioned on the resolved value.
        subResult = $" = {value} ";
        confidence = resolved.PlatonicConfidence;
        return true;
    }

    private GenerationResult GenerateNeuralTokens(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens,
        bool neuralFallback)
    {
        var (adaptiveBiasScale, checkpointContextEnabled) = GetAdaptiveTelemetryState();
        var checkpointContextConcepts = checkpointContextEnabled
            ? LoadCheckpointContextConcepts()
            : Array.Empty<string>();
        var generated = new List<int>(Math.Max(1, request.MaxNewTokens));
        var evidence = new List<PlatonicEvidence>();
        var totalBiasCount = 0;
        var biasMagnitudeSum = 0.0;
        var prev = _tokenizer.BosTokenId;
        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            double biasMagnitude;
            IReadOnlyDictionary<int, double>? biases;
            IReadOnlyList<PlatonicEvidence> stepEvidence;
            if (checkpointContextConcepts.Count > 0)
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, checkpointContextConcepts, out biasMagnitude, out stepEvidence);
            else
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, out biasMagnitude, out stepEvidence);

            var next = _model.PredictNextToken(
                inputTokens,
                prev,
                stepIndex: i,
                disallowToken: i == 0 ? _tokenizer.EosTokenId : null,
                penalizedTokens: generated,
                repetitionPenalty: 0.35,
                tokenBiases: biases,
                stopToken: _tokenizer.EosTokenId);
            if (biases is not null)
            {
                totalBiasCount += biases.Count;
                biasMagnitudeSum += biasMagnitude;
                if (stepEvidence.Count > 0)
                    evidence.AddRange(stepEvidence);
            }
            generated.Add(next);
            if (next == _tokenizer.EosTokenId)
                break;
            prev = next;
        }

        return new GenerationResult(
            Output: _tokenizer.Decode(generated),
            GeneratedTokens: generated.ToArray(),
            UsedPlatonicQuery: false,
            UsedNeuralFallback: neuralFallback,
            DecisionPath: "neural-token+platonic-bias",
            PlatonicConfidence: 0.0,
            AppliedBiasCount: totalBiasCount,
            AverageBiasMagnitude: totalBiasCount > 0 ? biasMagnitudeSum / Math.Max(1, generated.Count) : 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 0,
            Evidence: CollapseEvidence(evidence));
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

    /// <summary>
    /// The GRU-CONSTRUCTED platonic query: the model itself selects the face operation (learned op
    /// head, face-derived vocabulary, 0 = abstain) and the operand tokens (learned per-token head),
    /// and the platonic faces execute the query exactly. This is the learned successor to the compact
    /// arithmetic grammar — it handles framed natural-language arithmetic ("what is 1 + 1") because
    /// the GRU LEARNED which tokens are operands and which are framing, from the examples' own
    /// numeric structure. Ordered AFTER the exact compact parser (which stays authoritative for bare
    /// expressions) and BEFORE the concept-chain. Abstains — returns false — when the op head says
    /// none, operands don't resolve, or face-decode quality is poor.
    /// </summary>
    private bool TryGenerateFromGruQuery(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;

        var tokenIds = _tokenizer.Encode(request.Input);
        var (opId, opConfidence, flags) = _model.PredictQuery(tokenIds, AnchorPerception(tokenIds));
        if (opId <= 0 || opId >= GenesisNova.Model.GenesisNeuralModel.QueryOpCount || flags.Length != tokenIds.Length)
            return false;

        // Group the SELECTED tokens into operands, in order: adjacent selected digit tokens merge
        // into one number (the tokenizer's own digit-run convention); selected word tokens resolve
        // through the LEARNED relation carrier (one -> 1). Unselected tokens are the framing the GRU
        // learned to ignore.
        var vocab = _tokenizer.Vocabulary;
        string TokenText(int index)
        {
            var id = tokenIds[index];
            return id >= 0 && id < vocab.Count ? vocab[id] : string.Empty;
        }
        static bool IsDigitText(string s) => s.Length == 1 && s[0] is >= '0' and <= '9';

        var operands = new List<double>();
        var i = 0;
        while (i < flags.Length)
        {
            if (!flags[i]) { i++; continue; }
            var text = TokenText(i);
            // Unary minus: a selected '-' not preceded by a digit, followed by a selected digit, is
            // the operand's sign (mirrors the signed-run convention the label derivation supervises).
            var negative = false;
            if (text == "-"
                && i + 1 < flags.Length && flags[i + 1] && IsDigitText(TokenText(i + 1))
                && (i == 0 || !IsDigitText(TokenText(i - 1))))
            {
                negative = true;
                i++;
                text = TokenText(i);
            }
            if (IsDigitText(text))
            {
                var runValue = 0.0;
                while (i < flags.Length && flags[i] && IsDigitText(TokenText(i)))
                {
                    runValue = (runValue * 10.0) + (TokenText(i)[0] - '0');
                    i++;
                }
                operands.Add(negative ? -runValue : runValue);
                continue;
            }
            if (text.Any(char.IsLetter) &&
                TryResolveTokenToNumber(text, out var digit) &&
                double.TryParse(digit, System.Globalization.NumberStyles.AllowLeadingSign
                        | System.Globalization.NumberStyles.AllowDecimalPoint,
                    System.Globalization.CultureInfo.InvariantCulture, out var wordValue))
            {
                operands.Add(wordValue);
            }
            i++;
        }
        if (operands.Count < 2)
            return false;

        var left = operands[0];
        var right = operands[1];
        var additive = opId is 1 or 2;          // poly face: add/sub
        var sign = opId is 2 or 4 ? -1.0 : 1.0; // negative face for sub/div
        if (!TryFaceArithmetic(left, right, additive, sign, out var faceValue, out var quality))
            return false;
        if (quality <= 0.50)
            return false; // decode self-consistency gate — same bar as the assist sub-step path.

        var outTokens = _tokenizer.Encode(FormatNumber(faceValue), addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;

        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-gru-query",
            PlatonicConfidence: Math.Min(quality, opConfidence),
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1);
        return true;
    }

    /// <summary>
    /// LEARNED-OPERATION route. An operation LEARNED from examples (not a fixed op vocabulary, not parsed
    /// by name) is a first-class element of the space, SELECTED by following a learned RELATION from a cue
    /// concept (so "twice 7" reaches "double" through the learned twice↔double edge), and APPLIED:
    ///  • UNARY (one operand) — a transform vector T(f)=avg(embed(out)−embed(in)) applied BY COMPOSITION
    ///    (predicted = embed(x)+T(f), a Sum-composition of the argument with the function element), decoded
    ///    in the function's own face (PreferredFace: poly +k / log ×k; the other face holds a spurious but
    ///    clean reading, so decoding the wrong one would silently mislead).
    ///  • BINARY (two operands) — a discovered fold / log-linear STRUCTURE (mul = fold of add; c = a^α·b^β),
    ///    evaluated by <see cref="FoldPathDiscovery.TryPredict"/>.
    /// Both generalize to UNSEEN operands from a few examples (measured). Abstains when no store is wired,
    /// the operand count isn't 1 or 2, no cue resolves to a learned op, or the decode is low-quality.
    /// </summary>
    private bool TryGenerateFromLearnedFunction(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if ((_transformAccumulator is null && _foldPathDiscovery is null) || string.IsNullOrWhiteSpace(request.Input))
            return false;

        var tokens = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 2)
            return false;

        const System.Globalization.NumberStyles numericStyle =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;

        var operandTokens = tokens.Where(t => double.TryParse(t, numericStyle, inv, out _)).ToArray();
        if (operandTokens.Length is not (1 or 2)) // unary or binary learned op (binary arithmetic with an
            return false;                          // operator token is handled by the GRU-query route first)
        var cues = tokens.Where(t => t.Any(char.IsLetter)).ToArray();
        if (cues.Length == 0)
            return false;

        var dim = _memory.FaceDimension;

        foreach (var cue in cues)
        {
            // The op element is the cue itself OR a relational neighbour of it (learned edge) — retrieval
            // from the space, not a name lookup. First candidate carrying a learned op wins.
            var candidates = new List<string>(5) { cue };
            candidates.AddRange(_memory
                .GetNeighbors(cue, PlatonicNeighborhoodType.Relational, maxNeighbors: 4, minConfidence: 0.35)
                .Select(n => n.Concept));

            foreach (var fn in candidates)
            {
                // UNARY — learned transform applied by composition.
                if (operandTokens.Length == 1 && _transformAccumulator is not null
                    && _transformAccumulator.TryGetTransform(fn, out var transform))
                {
                    var predicted = _transformAccumulator.Apply(fn, InputEmbeddingComposer.GetInputEmbedding(operandTokens[0], dim));
                    if (predicted is null)
                        continue;
                    var (value, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(predicted, dim, transform.PreferredFace);
                    if (face != "none" && quality > 0.50
                        && TryBuildNumericResult(value, quality, $"platonic-learned-function:{fn}", request, out result))
                        return true;
                    continue;
                }

                // BINARY — learned fold / log-linear structure evaluated on the operands.
                if (operandTokens.Length == 2 && _foldPathDiscovery is not null && _foldPathDiscovery.HasOperation(fn)
                    && double.TryParse(operandTokens[0], numericStyle, inv, out var a)
                    && double.TryParse(operandTokens[1], numericStyle, inv, out var b)
                    && _foldPathDiscovery.TryPredict(fn, a, b, out var predValue, out _)
                    && TryBuildNumericResult(predValue, 0.85, $"platonic-learned-op:{fn}", request, out result))
                    return true;
            }
        }
        return false;
    }

    // Encode a numeric value into a platonic-credited GenerationResult; false if it doesn't tokenize.
    private bool TryBuildNumericResult(double value, double confidence, string decisionPath,
        GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var outTokens = _tokenizer.Encode(FormatNumber(value), addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;
        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: decisionPath,
            PlatonicConfidence: confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1);
        return true;
    }

    /// <summary>
    /// LEARNED-COMPOSER route. The GRU's plan head (<see cref="GenesisNeuralModel.PredictPlan"/>) selects a
    /// block-composition SHAPE; this assembles the corresponding glider and runs it on the substrate. The
    /// blocks are the vocabulary, the GRU is the composer — no hardcoded per-token resolver. Increment 1
    /// wires PlanKind=Predicate: a Compare→Branch tree over the two numeric operands → greater/less/equal,
    /// computed element-natively (the difference's sign) and generalizing to unseen operands. Other plan
    /// kinds (arithmetic/retrieval) are served by their existing routes. Abstains otherwise.
    /// </summary>
    private bool TryGenerateFromGliderPlan(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;
        var tokenIds = _tokenizer.Encode(request.Input);
        var (planKind, conf) = _model.PredictPlan(tokenIds, AnchorPerception(tokenIds));
        // Execute the shapes the dedicated routes DON'T serve: predicate (2), arithmetic→word (4),
        // fold-sum (5), fold-product (6), seq (7). Digit-arithmetic (1) defers to GruQuery, retrieval (3)
        // to relation-first, and the MULTI-operator expression-chain (8) to its own route below — keeping
        // their decision paths intact. Each shape's WORK is on the substrate.
        if (planKind is not (2 or 4 or 5 or 6 or 7))
            return false;

        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var operands = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Where(t => double.TryParse(t, ns, inv, out _)).ToArray();
        if (operands.Length < 2)
            return false;

        // The interpreter carries the shape registry's library, so Ref blocks resolve recursively against
        // the shapes-as-Function-elements — no per-call inline glider construction.
        var interp = _glider;
        GliderBlock root;
        switch (planKind)
        {
            case 2: // Predicate: Compare→Branch over the two operands (element-native difference-sign).
                root = new Branch(
                    new Compare(CompareOp.Greater, new Operand(0), new Operand(1)),
                    new Literal("greater"),
                    new Branch(new Compare(CompareOp.Less, new Operand(0), new Operand(1)),
                        new Literal("less"), new Literal("equal")));
                break;
            case 5: // Fold-sum: variadic reduce of + over ALL operands (one N-way R2 compose).
                root = new Fold(GliderOp.Add, 0);
                break;
            case 6: // Fold-product: variadic reduce of × over ALL operands.
                root = new Fold(GliderOp.Multiply, 0);
                break;
            case 7: // SEQ — Concatenate-Composition: a scaffold chunk RETRIEVED from the chunk-element store
                // (mined from graded-correct outputs — NOT a literal baked in here) bound to a substrate-
                // computed value. The COMPUTE part is substrate-native (Fold(Add) → one R2 compose over all
                // operands / the homomorphism); the assembly is the interpreter's Seq block (concatenation =
                // CompositionMode.Concatenate). Abstain until a scaffold has been learned.
                if (!_memory.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out var scaffold))
                    return false;
                root = new Seq(new GliderBlock[]
                {
                    new Literal(scaffold),
                    new Fold(GliderOp.Add, 0),
                });
                break;
            default: // planKind == 4: arithmetic FORMATTED AS A WORD — Hop(Compute(op,operands), Word).
                var (opId, _, _) = _model.PredictQuery(tokenIds, AnchorPerception(tokenIds));
                var gop = opId switch
                {
                    1 => GliderOp.Add,
                    2 => GliderOp.Subtract,
                    3 => GliderOp.Multiply,
                    4 => GliderOp.Divide,
                    _ => (GliderOp?)null
                };
                if (gop is null)
                    return false;
                root = new Hop(new Compute(gop.Value, new GliderBlock[] { new Operand(0), new Operand(1) }), HopTarget.Word);
                break;
        }

        string answer;
        try
        {
            answer = interp.Execute(new PlatonicGlider("plan", root), operands);
        }
        catch (InvalidOperationException)
        {
            return false; // unresolvable on the substrate (e.g. no learned digit→word edge) — fall through
        }
        if (string.IsNullOrEmpty(answer))
            return false;

        var outTokens = _tokenizer.Encode(answer, addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;
        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-glider-plan:" + planKind switch
            {
                2 => "predicate",
                4 => "arith-word",
                5 => "fold-sum",
                6 => "fold-product",
                7 => "seq",
                _ => "shape" + planKind
            },
            PlatonicConfidence: conf,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1);
        return true;
    }

    /// <summary>TARGET-AGNOSTIC perception of the input's anchor region (SPACE_AWARE_GRU.md §A), shared by the
    /// space-aware plan/op heads — the same construction the route head uses. Null when perception is off or
    /// there is no anchor. Numeric/arithmetic inputs have no relational anchor → ≈0 vector → graceful no-op.</summary>
    private double[]? AnchorPerception(IReadOnlyList<int> tokenIds)
    {
        if ((!_model.PerceptionPlan && !_model.PerceptionQuery) || tokenIds.Count == 0)
            return null;
        var toks = _tokenizer.Decode(tokenIds).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (toks.Length == 0)
            return null;
        var transformReliability = _model.TransformReliabilityRouting && _transformAccumulator is not null
            ? _transformAccumulator.BestReliabilityUcb()
            : 0.0;
        return _memory.ComputeRoutePerception(toks[^1], transformReliability);
    }

    /// <summary>
    /// EXPRESSION-CHAIN route (plan-kind 8). The GRU plan head selects "this is a multi-operator expression";
    /// this evaluates it by CHAINING compute-elements on the substrate. Each operator is classified from
    /// CONTEXT by the learned op head (the op head runs on the operator's local binary window) — there is NO
    /// hardcoded symbol→op map, so "x" is multiply only when the surrounding tokens make it so. Evaluation
    /// uses standard precedence (× ÷ before + −, each pass left-to-right); every binary step is one substrate
    /// R2 compose + homomorphic decode (PlatonicGliderInterpreter), so the answer GENERALISES to any operands.
    /// Control flow lives here; the compute is on the substrate. Abstains unless the plan head picked kind 8
    /// AND the input parses as a ≥2-operator expression AND the op head classifies every operator.
    /// </summary>
    private bool TryGenerateFromExpressionChain(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;
        var tokenIds = _tokenizer.Encode(request.Input);
        var (planKind, conf) = _model.PredictPlan(tokenIds, AnchorPerception(tokenIds));
        if (planKind != 8)
            return false; // the GRU plan head must SELECT the expression-chain shape

        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;

        // Parse the (space-separated) expression into alternating operand values and operator tokens.
        var raw = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var operands = new List<double>();
        var opTokens = new List<string>();
        var expectOperand = true;
        foreach (var tok in raw)
        {
            if (expectOperand)
            {
                if (!double.TryParse(tok, ns, inv, out var v))
                    return false; // framing/garbage where an operand was expected → abstain
                operands.Add(v);
                expectOperand = false;
            }
            else { opTokens.Add(tok); expectOperand = true; }
        }
        if (!expectOperand || opTokens.Count < 2 || operands.Count != opTokens.Count + 1)
            return false; // MULTI-operator only; single binary defers to GruQuery

        // Classify EACH operator from CONTEXT via the learned op head — its own local binary window. No
        // symbol→op lookup: the head decides add/sub/mul/div from the operand+operator+operand context.
        var ops = new List<GliderOp>(opTokens.Count);
        for (var i = 0; i < opTokens.Count; i++)
        {
            var window = _tokenizer.Encode($"{FormatNumber(operands[i])} {opTokens[i]} {FormatNumber(operands[i + 1])}");
            var (opId, _, _) = _model.PredictQuery(window);
            GliderOp? g = opId switch
            {
                1 => GliderOp.Add,
                2 => GliderOp.Subtract,
                3 => GliderOp.Multiply,
                4 => GliderOp.Divide,
                _ => (GliderOp?)null
            };
            if (g is null)
                return false; // op head couldn't classify this operator from context → abstain
            ops.Add(g.Value);
        }

        // One binary step = one substrate R2 compose + homomorphic decode (a chained compute-element).
        double Step(double l, GliderOp op, double r)
        {
            var glider = new PlatonicGlider("expr-step",
                new Compute(op, new GliderBlock[] { new Const(l), new Const(r) }));
            var txt = _glider.Execute(glider, Array.Empty<string>());
            return double.TryParse(txt, ns, inv, out var res) ? res : double.NaN;
        }

        // Precedence: resolve × ÷ left-to-right, then + − left-to-right — chaining the compute-elements.
        var vals = operands.ToList();
        var oo = ops.ToList();
        for (var i = 0; i < oo.Count;)
        {
            if (oo[i] is GliderOp.Multiply or GliderOp.Divide)
            {
                var r = Step(vals[i], oo[i], vals[i + 1]);
                if (double.IsNaN(r)) return false;
                vals[i] = r;
                vals.RemoveAt(i + 1);
                oo.RemoveAt(i);
            }
            else i++;
        }
        var acc = vals[0];
        for (var i = 0; i < oo.Count; i++)
        {
            acc = Step(acc, oo[i], vals[i + 1]);
            if (double.IsNaN(acc)) return false;
        }

        var outTokens = _tokenizer.Encode(FormatNumber(acc), addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;
        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-expression-chain",
            PlatonicConfidence: conf,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: opTokens.Count); // chain length = number of chained compute-elements
        return true;
    }

    private bool TryGenerateFromPlatonicPlan(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        // Arithmetic = the GRU-constructed query (op classified from context + operand selection), then
        // the homomorphism computes. No hardcoded compact symbol parser (removed 2026-06-14). Runs before
        // the concept-chain so framed/compact arithmetic doesn't fall into a relational text walk.
        // LEARNED-COMPOSER first: the GRU plan head selects a block-composition shape (predicate; or
        // arithmetic→word, which the digit-only GruQuery below cannot format). It ABSTAINS (returns false)
        // whenever the plan head is untrained or the shape isn't a wired one, so every route below — and
        // every existing test — is unaffected.
        if (TryGenerateFromGliderPlan(request, out result))
            return true;
        // EXPRESSION-CHAIN (plan-kind 8): a MULTI-operator expression evaluated by chaining compute-elements
        // on the substrate, each operator classified from context by the op head. Runs before the single-op
        // GruQuery (which assumes one operator); abstains on single-op / non-expression inputs.
        if (TryGenerateFromExpressionChain(request, out result))
            return true;
        if (TryGenerateFromGruQuery(request, out result))
            return true;
        // LEARNED-FUNCTION: apply a function learned as a transform-element by COMPOSITION, the function
        // selected from the space by relation (not parsed by name). Handles unary learned functions the
        // homomorphism can't (it computes a SPECIFIED op; this LEARNED which function from examples).
        if (TryGenerateFromLearnedFunction(request, out result))
            return true;
        // RELATION-FIRST retrieval: for a single-concept query, follow the strongest learned RELATION edge
        // before the geometric concept-chain. MEASURED (2026-06-14): pure geometric retrieval was refuted
        // as a replacement — the migrated numeric face is unstable under contrastive repulsion and the
        // semantic face is lexical (confuses "four"≈"fruit"), so the relation edge is the robust mechanism.
        if (TryGenerateFromRelationEdge(request, out result))
            return true;

        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count == 0)
            return false;

        var maxHops = ResolveMaxPlatonicHops(anchors.Count);
        var beamWidth = ResolvePlatonicBeamWidth(anchors.Count);
        var conceptResult = _memory.QueryConceptChain(
            anchors,
            maxHops: maxHops,
            beamWidth: beamWidth,
            evidence: out var evidence);
        if (string.IsNullOrWhiteSpace(conceptResult.Text))
            return false;

        // The chain is a SEARCH WALK, not the surface answer: its text concatenates every beam
        // candidate across every hop ("fruit orange banana grape" for 'apple', "1 0 2 3" for 'one'),
        // which was the root of the answer-wrapped-in-noise outputs. The platonic-direct ANSWER is
        // the TOP retrieval — the first selected concept, i.e. the highest-scoring hop-1 candidate —
        // with the rest of the walk retained as evidence, not emitted as text.
        var topConcept = conceptResult.Text.Split(
            (char[]?)null, StringSplitOptions.RemoveEmptyEntries)[0];
        var tokens = _tokenizer.Encode(topConcept, addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (tokens.Length == 0)
            return false;

        result = new GenerationResult(
            Output: _tokenizer.Decode(tokens),
            GeneratedTokens: tokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-query-concept-chain",
            PlatonicConfidence: conceptResult.Confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: conceptResult.Hops,
            Evidence: evidence);
        return true;
    }

    // RELATION-FIRST retrieval. For a single-concept query, return the strongest learned relational
    // neighbour ("1"→"one", "apple"→"fruit") — the STABLE substrate fact the lesson established. (A
    // geometric-first variant was tried and empirically refuted: the migrated numeric face is unstable
    // under contrastive repulsion and the semantic face is lexical, so geometry mis-retrieved; the edge
    // is the robust mechanism.) Single-concept only; arithmetic / glider capabilities are handled earlier.
    private const double RelationFirstMinConfidence = 0.5;

    private bool TryGenerateFromRelationEdge(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count != 1)
            return false;

        var neighbours = _memory.GetNeighbors(
            anchors[0],
            PlatonicNeighborhoodType.Relational,
            maxNeighbors: 5,
            minConfidence: RelationFirstMinConfidence);

        // Pick the strongest neighbour that is a real, user-facing concept — never a RESERVED internal marker
        // (face:poly/face:log) or an op-token. Those are routing affinity, not answers.
        string? answer = null;
        var confidence = 0.0;
        foreach (var n in neighbours)
        {
            if (PlatonicSpaceMemory.IsReservedConcept(n.Concept) || _memory.IsOperationToken(n.Concept))
                continue;
            answer = n.Concept;
            confidence = n.Confidence;
            break;
        }
        if (answer is null)
            return false;

        var tokens = _tokenizer.Encode(answer, addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (tokens.Length == 0)
            return false;

        result = new GenerationResult(
            Output: _tokenizer.Decode(tokens),
            GeneratedTokens: tokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-relation-edge",
            PlatonicConfidence: confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1);
        return true;
    }

    // Resolves a single token to a digit string using ONLY the learned relation graph — no hardcoded
    // word→number table. A token that already strict-parses as a number is returned unchanged. Otherwise
    // the strongest NUMERIC relational neighbour (highest ObservationCount, tiebreak higher Confidence)
    // is used. Returns false when no numeric relation exists, so unlearned words stay verbatim.
    private bool TryResolveTokenToNumber(string token, out string digit)
    {
        digit = token;
        if (string.IsNullOrEmpty(token))
            return false;

        // Strict numeric parse: deliberately NOT NumberStyles.Any (which would accept noise like "0+").
        const System.Globalization.NumberStyles strictStyles =
            System.Globalization.NumberStyles.AllowLeadingSign |
            System.Globalization.NumberStyles.AllowDecimalPoint |
            System.Globalization.NumberStyles.AllowLeadingWhite |
            System.Globalization.NumberStyles.AllowTrailingWhite;

        if (double.TryParse(token, strictStyles, System.Globalization.CultureInfo.InvariantCulture, out _))
            return true; // already a number — leave unchanged

        string? bestConcept = null;
        var bestObs = int.MinValue;
        var bestConf = double.MinValue;
        foreach (var neighbor in _memory.GetNeighbors(
                     token,
                     GenesisNova.Cognition.PlatonicNeighborhoodType.Relational,
                     maxNeighbors: 16,
                     minConfidence: 0.4))
        {
            if (!double.TryParse(neighbor.Concept, strictStyles, System.Globalization.CultureInfo.InvariantCulture, out _))
                continue; // only numeric relational neighbours qualify
            if (neighbor.ObservationCount > bestObs ||
                (neighbor.ObservationCount == bestObs && neighbor.Confidence > bestConf))
            {
                bestObs = neighbor.ObservationCount;
                bestConf = neighbor.Confidence;
                bestConcept = neighbor.Concept;
            }
        }

        if (bestConcept is null)
            return false;

        digit = bestConcept;
        return true;
    }

    /// <summary>
    /// Performs arithmetic in face embedding space by BLENDING the operand faces (preserving the
    /// homomorphism poly(a)±poly(b)=poly(a±b), log(a)±log(b)=log(a*b or a/b)) and then decoding the
    /// resulting predicted embedding through the source-of-truth
    /// <see cref="PlatonicFaceDecoder.DecodeNumericFromPrediction"/> — the exact algebraic inverse of
    /// the composer. The decode formula AND the self-consistency quality come from the decoder, so
    /// they match the canonical genesis-engine codec rather than an ad-hoc CV metric.
    /// <para>
    /// preferFace hint: 1=poly for add/sub, 2=log for mul/div. The log face decodes positive, so the
    /// operand sign is reapplied here (sign lives in the poly face / operands, not the log magnitude).
    /// </para>
    /// </summary>
    private bool TryFaceArithmetic(double left, double right, bool additiveFace, double rightFaceSign, out double result, out double quality)
    {
        result = 0;
        quality = 0;

        var leftKey = FormatNumber(left);
        var rightKey = FormatNumber(right);

        if (!_memory.TryGetConceptFace(leftKey, out var faceA) ||
            !_memory.TryGetConceptFace(rightKey, out var faceB))
            return false;

        var dim = _memory.FaceDimension;
        if (faceA.Length < dim || faceB.Length < dim)
            return false;

        if (!additiveFace)
        {
            // Zero is outside log-space; preserve exact arithmetic behavior for zero cases
            // (mul by zero = 0; div of zero by nonzero = 0). Both faces would otherwise be NaN/Inf.
            if (Math.Abs(left) < 1e-12 || (rightFaceSign > 0.0 && Math.Abs(right) < 1e-12))
            {
                result = 0;
                quality = 1.0;
                return true;
            }
        }

        // Blend the operand faces element-wise. The poly block [0..numericDims) carries the additive
        // homomorphism and the log block [logStart..) carries the multiplicative one, so a single
        // signed sum yields the predicted embedding for the result on BOTH faces simultaneously.
        var blended = new double[dim];
        for (var i = 0; i < dim; i++)
            blended[i] = faceA[i] + rightFaceSign * faceB[i];

        // Decode via the canonical codec, hinting the face the operation actually evaluates on.
        var preferFace = additiveFace ? 1 : 2; // 1=poly (add/sub), 2=log (mul/div)
        var (value, decodeQuality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(blended, dim, preferFace);
        if (face == "none")
            return false;

        // The log face always decodes a positive magnitude; reapply the operand sign for mul/div.
        if (!additiveFace && face == "log")
            value *= Math.Sign(left) * Math.Sign(right);

        result = value;
        quality = decodeQuality;
        return !double.IsNaN(result) && !double.IsInfinity(result) && !double.IsNaN(quality) && !double.IsInfinity(quality);
    }

    private static int ResolveMaxPlatonicHops(int anchorCount)
        => Math.Clamp(1 + anchorCount / 2, 2, 4);

    private static int ResolvePlatonicBeamWidth(int anchorCount)
        => Math.Clamp(1 + anchorCount / 3, 2, 4);

    private static string FormatNumber(double value)
    {
        if (Math.Abs(value - Math.Round(value)) <= 1e-9)
            return ((long)Math.Round(value)).ToString(System.Globalization.CultureInfo.InvariantCulture);

        return value.ToString("0.##########", System.Globalization.CultureInfo.InvariantCulture);
    }

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        double strength,
        out double averageBiasMagnitude)
        => BuildTokenBiases(inputTokens, generatedTokens, strength, null, out averageBiasMagnitude);

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        double strength,
        out double averageBiasMagnitude,
        out IReadOnlyList<PlatonicEvidence> evidence)
        => BuildTokenBiases(inputTokens, generatedTokens, strength, null, out averageBiasMagnitude, out evidence);

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        double strength,
        IReadOnlyCollection<string>? checkpointContextConcepts,
        out double averageBiasMagnitude)
        => BuildTokenBiases(inputTokens, generatedTokens, strength, checkpointContextConcepts, out averageBiasMagnitude, out _);

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        double strength,
        IReadOnlyCollection<string>? checkpointContextConcepts,
        out double averageBiasMagnitude,
        out IReadOnlyList<PlatonicEvidence> evidence)
    {
        averageBiasMagnitude = 0.0;
        evidence = Array.Empty<PlatonicEvidence>();
        var contextConcepts = BuildContextConcepts(inputTokens, generatedTokens, checkpointContextConcepts);
        if (contextConcepts.Count == 0)
            return null;

        var biases = new Dictionary<int, double>();
        var evidenceItems = new List<PlatonicEvidence>();
        for (var token = 0; token < _tokenizer.VocabularySize; token++)
        {
            if (token == _tokenizer.PadTokenId || token == _tokenizer.BosTokenId || token == _tokenizer.EosTokenId)
                continue;

            var candidate = _tokenizer.Vocabulary[token];
            if (string.IsNullOrWhiteSpace(candidate))
                continue;
            if (!_memory.ContainsConcept(candidate))
                continue;

            var total = 0.0;
            var count = 0;
            var applied = new List<PlatonicEvidence>();
            foreach (var concept in contextConcepts)
            {
                var contradiction = _memory.GetContradiction(candidate, concept);
                total += contradiction;
                count++;
                applied.Add(new PlatonicEvidence(candidate, concept, 0.5 - contradiction, 1));
            }

            if (count == 0)
                continue;

            var avg = total / count;
            var bias = Math.Clamp((0.5 - avg) * (0.9 * Math.Max(0.0, strength)), -0.9, 0.9);
            if (Math.Abs(bias) > 1e-6)
            {
                biases[token] = bias;
                if (evidenceItems.Count < 512)
                {
                    evidenceItems.AddRange(applied
                        .OrderByDescending(e => Math.Abs(e.Contribution))
                        .Take(2)
                        .Select(e => e with { Contribution = e.Contribution * (0.9 * Math.Max(0.0, strength)) }));
                }
            }
        }

        // REPETITION GUARD: a token already emitted THIS generation must not keep its positive platonic BOOST
        // on later steps. The query bias is static (e.g. `find X` boosts X at every step), so without this the
        // boosted answer re-wins each step and the decoder repeats it to maxTokens instead of taking the LEARNED
        // EOS ("ObserveTrainingPair" ×16). Dropping the boost for emitted tokens lets termination (or a genuinely
        // next token) win — same principle as the rule that the bias layer must not override the learned stop.
        // Only the POSITIVE boost is removed; the NN's own logit (which legitimately allows repeats like "100")
        // is untouched, and numeric tokens aren't concepts so they never carried a bias here anyway.
        for (var g = 0; g < generatedTokens.Count; g++)
            if (biases.TryGetValue(generatedTokens[g], out var prior) && prior > 0)
                biases.Remove(generatedTokens[g]);

        if (biases.Count == 0)
            return null;

        averageBiasMagnitude = biases.Values.Average(v => Math.Abs(v));
        evidence = CollapseEvidence(evidenceItems);
        return biases;
    }

    private HashSet<string> BuildContextConcepts(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        IReadOnlyCollection<string>? checkpointContextConcepts = null)
    {
        var concepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        AddTokens(inputTokens, concepts);
        // GENERATED tokens are deliberately NOT added. The platonic bias must represent the QUERY's
        // evidence, not compound on the model's own output: feeding the emitted answer back made its
        // platonic SIBLINGS the most-boosted tokens (answer "fruit" → boost grape/banana/orange) while
        // EOS is never biased, so the boosted siblings always outscored the LEARNED stop and decode
        // cascaded through the neighbourhood instead of terminating ("fruit grape banana orange",
        // "paris ratio-7", "1023"). The NN learns termination from data (every target ends in EOS);
        // the bias layer must not override it. (generatedTokens is retained in the signature because
        // the working-context callers still pass it — it is simply no longer a bias source.)
        if (checkpointContextConcepts is not null)
        {
            foreach (var concept in checkpointContextConcepts)
            {
                if (!string.IsNullOrWhiteSpace(concept))
                    concepts.Add(concept);
            }
        }
        return concepts;
    }

    private static IReadOnlyList<PlatonicEvidence> CollapseEvidence(IEnumerable<PlatonicEvidence> evidence)
    {
        var collapsed = evidence
            .Where(e => !string.IsNullOrWhiteSpace(e.Concept))
            .GroupBy(e => (
                Concept: e.Concept.Trim().ToLowerInvariant(),
                Related: string.IsNullOrWhiteSpace(e.RelatedConcept) ? null : e.RelatedConcept.Trim().ToLowerInvariant(),
                e.Hop))
            .Select(g => new PlatonicEvidence(
                g.Key.Concept,
                g.Key.Related,
                g.Sum(e => e.Contribution),
                g.Key.Hop))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(32)
            .ToArray();
        return collapsed;
    }

    private void AddTokens(IReadOnlyList<int> tokens, HashSet<string> concepts)
    {
        foreach (var token in tokens)
        {
            if (token < 0 || token >= _tokenizer.VocabularySize)
                continue;
            if (token == _tokenizer.PadTokenId || token == _tokenizer.BosTokenId || token == _tokenizer.EosTokenId)
                continue;

            var value = _tokenizer.Vocabulary[token];
            if (!string.IsNullOrWhiteSpace(value))
                concepts.Add(value.ToLowerInvariant());
        }
    }

    private IReadOnlyList<string> ExtractConceptAnchors(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return Array.Empty<string>();

        return input
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim('?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '"', '\''))
            .Select(t => t.ToLowerInvariant())
            // Keep single-char DIGIT tokens as concept anchors: a bare number ("3") is a legitimate
            // concept with a learned relation to its word, and excluding it sent digit→word to the
            // neural path. (Earlier reverted because at the DEGENERATE test dim — face 32, no free
            // region — the concept-chain returned a numeric neighbour; being re-validated at production
            // face dim where the free region exists.) Stray single letters still dropped; ContainsConcept gates.
            .Where(t => t.Length > 1 || (t.Length == 1 && char.IsDigit(t[0])))
            // An op-token (e.g. "find") is a ROUTE TRIGGER, not a retrieval anchor — excluding it lets the GRU
            // route on the verb while the OPERAND anchors retrieval (so "find <topic>" is a single-anchor
            // relation-first query, not a collapse onto the verb's edges). See PlatonicSpaceMemory op-tokens.
            .Where(t => !_memory.IsOperationToken(t))
            .Where(t => _memory.ContainsConcept(t))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(8)
            .ToArray();
    }

    private IReadOnlyCollection<string> LoadCheckpointContextConcepts()
    {
        if (_platonicFilePathProvider is null)
            return Array.Empty<string>();

        var path = _platonicFilePathProvider();
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return Array.Empty<string>();

        var lastWriteUtc = File.GetLastWriteTimeUtc(path);
        lock (_platonicFileCacheLock)
        {
            if (_cachedPlatonicFileConcepts is not null &&
                string.Equals(_cachedPlatonicFilePath, path, StringComparison.OrdinalIgnoreCase) &&
                _cachedPlatonicFileWriteUtc == lastWriteUtc)
            {
                return _cachedPlatonicFileConcepts;
            }
        }

        var concepts = ParseConceptsFromCheckpoint(path);

        lock (_platonicFileCacheLock)
        {
            _cachedPlatonicFilePath = path;
            _cachedPlatonicFileWriteUtc = lastWriteUtc;
            _cachedPlatonicFileConcepts = concepts;
        }

        return concepts;
    }

    private static IReadOnlyCollection<string> ParseConceptsFromCheckpoint(string checkpointPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(checkpointPath));
        if (!doc.RootElement.TryGetProperty("PlatonicSpace", out var platonicSpace) ||
            platonicSpace.ValueKind != JsonValueKind.Object)
            return Array.Empty<string>();

        var concepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (platonicSpace.TryGetProperty("Nodes", out var nodes) && nodes.ValueKind == JsonValueKind.Array)
        {
            foreach (var node in nodes.EnumerateArray())
            {
                if (!node.TryGetProperty("Name", out var nameElement) || nameElement.ValueKind != JsonValueKind.String)
                    continue;
                AddConceptFragments(nameElement.GetString(), concepts);
            }
        }

        if (platonicSpace.TryGetProperty("Relations", out var relations) && relations.ValueKind == JsonValueKind.Array)
        {
            foreach (var relation in relations.EnumerateArray())
            {
                if (relation.TryGetProperty("Left", out var left) && left.ValueKind == JsonValueKind.String)
                    AddConceptFragments(left.GetString(), concepts);
                if (relation.TryGetProperty("Right", out var right) && right.ValueKind == JsonValueKind.String)
                    AddConceptFragments(right.GetString(), concepts);
            }
        }

        return concepts;
    }

    private static void AddConceptFragments(string? concept, HashSet<string> output)
    {
        if (string.IsNullOrWhiteSpace(concept))
            return;

        output.Add(concept.ToLowerInvariant());
        foreach (var fragment in concept.Split([' ', '\t', '\r', '\n', '-', '_', ':', '/', '\\'],
                     StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            output.Add(fragment.ToLowerInvariant());
        }
    }

    private (double BiasScale, bool CheckpointContextEnabled) GetAdaptiveTelemetryState()
    {
        lock (_telemetryLock)
        {
            var biasScale = Math.Clamp(_adaptiveBiasScale * _trainerHint.BiasScale, MinAdaptiveBiasScale, MaxAdaptiveBiasScale);
            var checkpointEnabled = _checkpointContextConceptsEnabled && _trainerHint.EnableContextBias;
            return (biasScale, checkpointEnabled);
        }
    }

    private static double BlendEma(double current, double sample, double alpha)
        => (current * (1.0 - alpha)) + (sample * alpha);
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
