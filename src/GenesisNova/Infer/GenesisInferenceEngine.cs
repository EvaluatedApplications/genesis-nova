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
    private readonly FoldPathDiscovery? _foldPathDiscovery;
    private readonly TransformLibrary? _transformLibrary;
    private readonly TransformAccumulator? _transformAccumulator;
    private readonly Func<string?>? _platonicFilePathProvider;
    private readonly bool _enableDiagnosticFaceArithmeticShortcut;
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
        FoldPathDiscovery? foldPathDiscovery = null,
        TransformLibrary? transformLibrary = null,
        TransformAccumulator? transformAccumulator = null,
        bool enableDiagnosticFaceArithmeticShortcut = true,
        int maxPlatonicAssistInvocations = 3)
    {
        _tokenizer = tokenizer;
        _model = model;
        _memory = memory;
        _platonicFilePathProvider = platonicFilePathProvider;
        _foldPathDiscovery = foldPathDiscovery;
        _transformLibrary = transformLibrary;
        _transformAccumulator = transformAccumulator;
        _enableDiagnosticFaceArithmeticShortcut = enableDiagnosticFaceArithmeticShortcut;
        _maxPlatonicAssistInvocations = Math.Clamp(maxPlatonicAssistInvocations, 0, 16);
    }

    public bool DiagnosticFaceArithmeticShortcutEnabled => _enableDiagnosticFaceArithmeticShortcut;

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
        var (routeId, routeConfidence) = _model.PredictRoute(inputTokens);

        // Mode 2 (platonic-assisted reasoning): generate neurally but invoke the platonic space
        // mid-generation as an internal scratchpad. Falls back to pure neural if no sub-step fires.
        if (routeId == 2)
        {
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
            if (TryGenerateFromDiscoveredTransform(request, out var discovered))
            {
                RecordRouteDecision(routeId, 1, true, true, true, 1, discovered.DecisionPath, routeConfidence);
                return discovered;
            }
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
                tokenBiases: biases);
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
    /// Identifies a platonic-resolvable arithmetic sub-step inside <paramref name="workingText"/>
    /// and resolves it EXACTLY through the face homomorphism (log/poly faces) or a discovered
    /// transform. Returns the formatted sub-result and a decode-consistency confidence.
    /// Returns false when no resolvable sub-step exists or confidence is too low.
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

        // Scan for the LAST compact arithmetic sub-expression (e.g. "12+7", "8*3") anywhere in the
        // working text — the most recent sub-step is the one the model is currently working toward.
        var matches = System.Text.RegularExpressions.Regex.Matches(
            workingText,
            @"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)");
        for (var m = matches.Count - 1; m >= 0; m--)
        {
            var match = matches[m];
            var key = match.Value.Trim();
            if (alreadyResolved.Contains(key))
                continue;

            if (!double.TryParse(match.Groups[1].Value, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var left) ||
                !double.TryParse(match.Groups[3].Value, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out var right))
                continue;

            var op = match.Groups[2].Value switch
            {
                "+" => ArithmeticOperation.Add,
                "-" => ArithmeticOperation.Subtract,
                "*" or "x" => ArithmeticOperation.Multiply,
                "/" => ArithmeticOperation.Divide,
                _ => ArithmeticOperation.Add
            };
            if (op == ArithmeticOperation.Divide && Math.Abs(right) < 1e-12)
                continue; // division by zero — skip

            // Prefer a learned/discovered transform when the capability exists; otherwise resolve
            // via the exact face homomorphism. Both keep the arithmetic exact.
            double value;
            double quality;
            var operationName = ToOperationName(op);
            if (_foldPathDiscovery is not null &&
                _foldPathDiscovery.HasOperation(operationName) &&
                _foldPathDiscovery.TryPredict(operationName, left, right, out var predicted, out _))
            {
                value = predicted;
                quality = 0.85;
            }
            else
            {
                var additiveFace = op is ArithmeticOperation.Add or ArithmeticOperation.Subtract;
                var rightFaceSign = op is ArithmeticOperation.Subtract or ArithmeticOperation.Divide ? -1.0 : 1.0;
                if (!TryFaceArithmetic(left, right, additiveFace, rightFaceSign, out value, out quality))
                    continue;
            }

            if (LooksLikeIntegerOperands(left, right) && Math.Abs(value - Math.Round(value)) > 0.25)
                continue;
            if (quality <= 0.50)
                continue; // low decode confidence — fall back to pure neural for this step

            alreadyResolved.Add(key);
            // Inject as " = <result> " so the neural decoder can continue conditioned on the
            // resolved scratchpad value and transform it back into words.
            subResult = $" = {FormatNumber(value)} ";
            confidence = quality;
            return true;
        }

        return false;
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
                tokenBiases: biases);
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

    private bool TryGenerateFromDiscoveredTransform(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (_foldPathDiscovery is null)
            return false;

        if (!TryParseArithmeticInput(request.Input, out var left, out var right, out var operation, out var isContextWrapped, out _))
            return false;
        if (isContextWrapped)
            return false;
        var operationName = ToOperationName(operation);
        var hasCapability = _foldPathDiscovery.HasOperation(operationName) ||
                            (_transformLibrary?.HasCapability(operationName) ?? false);
        if (!hasCapability)
            return false;

        if (!_foldPathDiscovery.TryPredict(operationName, left, right, out var prediction, out var route))
        {
            _transformLibrary?.RecordFailure(operationName);
            return false;
        }

        if (LooksLikeIntegerOperands(left, right) &&
            Math.Abs(prediction - Math.Round(prediction)) > 0.25)
            return false;

        var response = FormatNumber(prediction);
        var tokens = _tokenizer
            .Encode(response, addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (tokens.Length == 0)
            return false;

        _transformLibrary?.RecordSuccess(operationName);
        UpdateAccumulatorFromInference(operationName, request.Input, response);

        // Confidence from decode self-consistency of the learned transform's PREDICTED embedding
        // (TransformAccumulator.Apply yields input+transform; the canonical codec decodes it and
        // scores how well the predicted face matches the value it claims) — not a hardcoded floor.
        var confidence = DecodeTransformConfidence(operationName, request.Input, operation, prediction);

        result = new GenerationResult(
            Output: _tokenizer.Decode(tokens),
            GeneratedTokens: tokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-discovered-transform",
            PlatonicConfidence: confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1,
            RoutedTransform: operationName,
            TransformIntercept: route);
        return true;
    }

    private bool TryGenerateFromPlatonicPlan(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (TryResolveArithmeticQuery(request.Input, out var arithmetic))
        {
            var queryTokens = _tokenizer.Encode(arithmetic.ResponseText, addEos: true);
            var bounded = queryTokens.Take(Math.Max(1, request.MaxNewTokens)).ToArray();
            result = new GenerationResult(
                Output: _tokenizer.Decode(bounded),
                GeneratedTokens: bounded,
                UsedPlatonicQuery: true,
                UsedNeuralFallback: false,
                DecisionPath: "platonic-query-slot-decode",
                PlatonicConfidence: arithmetic.Confidence,
                AppliedBiasCount: 0,
                AverageBiasMagnitude: 0.0,
                ChunksGenerated: 1,
                PlatonicHopCount: 1);
            return true;
        }

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

        var tokens = _tokenizer.Encode(conceptResult.Text, addEos: true)
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

    private bool TryResolveArithmeticQuery(string input, out PlatonicQueryResolution resolution)
    {
        resolution = default;
        if (!TryParseArithmeticInput(input, out var left, out var right, out var op, out var likelyContextWrapped, out var isExplicitArithmeticExpression))
            return false;
        if (likelyContextWrapped)
            return false;

        // The exact direct face-arithmetic path is available by default in production: when the
        // 3-mode router selects platonic-direct (route 1) and the input is a compact arithmetic
        // expression, we resolve it exactly here rather than silently falling through to neural.
        // (_enableDiagnosticFaceArithmeticShortcut is retained for telemetry/diagnostics but no
        // longer gates this production route — it defaults true and the gate is removed so a
        // disabled flag can never drop a platonic-direct answer to neural.)

        var additiveFace = op is ArithmeticOperation.Add or ArithmeticOperation.Subtract;
        var rightFaceSign = op is ArithmeticOperation.Subtract or ArithmeticOperation.Divide ? -1.0 : 1.0;

        // Compute via face arithmetic — preserves the homomorphism poly(a)±poly(b)=poly(a±b),
        // log(a)±log(b)=log(a*b or a/b). No plain C# arithmetic shortcut.
        if (!TryFaceArithmetic(left, right, additiveFace, rightFaceSign, out var faceValue, out var faceQuality))
            return false;

        // Confidence comes from decode consistency across face dimensions (no hardcoded floor).
        var confidence = faceQuality;
        if (!isExplicitArithmeticExpression && confidence <= 0.50)
            return false;

        resolution = new PlatonicQueryResolution(
            ResponseText: FormatNumber(faceValue),
            Confidence: confidence,
            Operation: op,
            IsExplicitExpression: isExplicitArithmeticExpression);

        var operationName = ToOperationName(op);
        _transformLibrary?.RecordSuccess(operationName);
        UpdateAccumulatorFromInference(operationName, input, resolution.ResponseText);
        return true;
    }

    // Parses ONLY compact arithmetic expressions (e.g. "4+5", "3*2", "10-1").
    // Natural language forms ("what is 4 plus 5") are NOT handled here — they flow through
    // the ML path so routing is informed by learned structure, not keyword heuristics.
    private bool TryParseArithmeticInput(
        string input,
        out double left,
        out double right,
        out ArithmeticOperation operation,
        out bool likelyContextWrapped,
        out bool isExplicitArithmeticExpression)
    {
        left = 0d;
        right = 0d;
        operation = ArithmeticOperation.Add;
        likelyContextWrapped = false;
        isExplicitArithmeticExpression = false;
        if (string.IsNullOrWhiteSpace(input))
            return false;

        var normalized = input.ToLowerInvariant();
        likelyContextWrapped =
            normalized.Contains("context:") ||
            normalized.Contains("\nuser:") ||
            normalized.Contains("\nassistant:");
        if (likelyContextWrapped)
            return false;

        // Relational token canonicalization (NOTHING hardcoded): before the compact regex, rewrite
        // each whitespace-delimited NON-numeric operand token into the digit of its strongest numeric
        // relational neighbour, learned from the platonic relation graph (e.g. "one" -> "1"). Operators
        // and unresolvable tokens (e.g. "zarp") are left verbatim, so the EXISTING compact regex below
        // is the sole arithmetic grammar — words with no learned numeric relation simply never resolve.
        var tokens = normalized.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length > 1)
        {
            var rewritten = false;
            for (var i = 0; i < tokens.Length; i++)
            {
                // Only WORD-LIKE tokens (containing a letter) are candidates for number-word resolution.
                // Operator symbols (+, -, *, /) have no letters, so they pass through untouched — without
                // this guard an operator that co-occurs with numbers in training would itself resolve to a
                // digit and corrupt the expression ("1 + 1" -> "1 5 1"). Not an operator table: just "only
                // resolve words, not symbols".
                if (!tokens[i].Any(char.IsLetter))
                    continue;
                if (TryResolveTokenToNumber(tokens[i], out var digit) && !string.Equals(digit, tokens[i], StringComparison.Ordinal))
                {
                    tokens[i] = digit;
                    rewritten = true;
                }
            }
            if (rewritten)
                normalized = string.Join(" ", tokens);
        }

        // Only handle compact symbol-based expressions: "4+5", "3*2", "10-1", "8/2", "6x3"
        var compact = System.Text.RegularExpressions.Regex.Match(
            normalized,
            @"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)\s*$");
        if (!compact.Success)
            return false;

        if (!double.TryParse(compact.Groups[1].Value, System.Globalization.NumberStyles.Any,
                System.Globalization.CultureInfo.InvariantCulture, out left) ||
            !double.TryParse(compact.Groups[3].Value, System.Globalization.NumberStyles.Any,
                System.Globalization.CultureInfo.InvariantCulture, out right))
            return false;

        isExplicitArithmeticExpression = true;
        operation = compact.Groups[2].Value switch
        {
            "+" => ArithmeticOperation.Add,
            "-" => ArithmeticOperation.Subtract,
            "*" or "x" => ArithmeticOperation.Multiply,
            _ => ArithmeticOperation.Add
        };

        if (compact.Groups[2].Value is "/" && Math.Abs(right) > 1e-12)
        {
            operation = ArithmeticOperation.Divide;
        }
        else if (compact.Groups[2].Value is "/")
        {
            return false; // division by zero
        }

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

    private static string ToOperationName(ArithmeticOperation operation)
        => operation switch
        {
            ArithmeticOperation.Add => "add",
            ArithmeticOperation.Subtract => "sub",
            ArithmeticOperation.Multiply => "mul",
            ArithmeticOperation.Divide => "div",
            _ => "add"
        };

    private static bool LooksLikeIntegerOperands(double left, double right)
        => Math.Abs(left - Math.Round(left)) <= 1e-6 && Math.Abs(right - Math.Round(right)) <= 1e-6;

    private void UpdateAccumulatorFromInference(string operation, string input, string output)
    {
        // Training-only: inference must not mutate the shared persistent transform store.
        if (_transformAccumulator is null || !LearningEnabled)
            return;

        var dim = _transformAccumulator.EmbeddingDimension;
        var mode = _foldPathDiscovery?.GetComposition(operation) ?? CompositionMode.Sum;
        var inputEmbedding = InputEmbeddingComposer.ComposeInput(input, mode, dim);
        var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(output, dim);
        _transformAccumulator.Learn(operation, inputEmbedding, outputEmbedding);
    }

    /// <summary>
    /// Derives a principled confidence for a discovered-transform answer by decoding the learned
    /// transform's PREDICTED embedding through the canonical codec. Composes the input embedding the
    /// same way <see cref="UpdateAccumulatorFromInference"/> does, applies the accumulated transform
    /// (predicted = input + transform vector), then decodes that predicted face with the operation's
    /// preferred face (1=poly for add/sub, 2=log for mul/div). The decode Quality measures how
    /// self-consistent the predicted face is; we additionally require the decoded value to agree with
    /// the discovered prediction, otherwise the transform face and the fold route disagree and the
    /// answer is less trustworthy. Falls back to a moderate confidence when no transform is available
    /// to decode (e.g. answer came purely from the fold/log-linear route before any accumulation).
    /// </summary>
    private double DecodeTransformConfidence(
        string operationName,
        string input,
        ArithmeticOperation operation,
        double prediction)
    {
        const double NoDecodeFallbackConfidence = 0.75;
        if (_transformAccumulator is null)
            return NoDecodeFallbackConfidence;

        var dim = _transformAccumulator.EmbeddingDimension;
        var mode = _foldPathDiscovery?.GetComposition(operationName) ?? CompositionMode.Sum;
        var inputEmbedding = InputEmbeddingComposer.ComposeInput(input, mode, dim);
        var predicted = _transformAccumulator.Apply(operationName, inputEmbedding);
        if (predicted is null)
            return NoDecodeFallbackConfidence;

        var preferFace = operation is ArithmeticOperation.Multiply or ArithmeticOperation.Divide ? 2 : 1;
        var (decodedValue, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(predicted, dim, preferFace);
        if (face == "none" || quality <= 0.0)
            return NoDecodeFallbackConfidence;

        // The log face decodes a positive magnitude; compare on magnitude so a sign mismatch from the
        // unsigned log face does not spuriously deflate agreement.
        var agreementBase = Math.Max(1.0, Math.Abs(prediction));
        var disagreement = face == "log"
            ? Math.Abs(Math.Abs(decodedValue) - Math.Abs(prediction)) / agreementBase
            : Math.Abs(decodedValue - prediction) / agreementBase;
        var agreement = Math.Clamp(1.0 - disagreement, 0.0, 1.0);

        return Math.Clamp(quality * agreement, 0.0, 1.0);
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
        AddTokens(generatedTokens, concepts);
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
            .Where(t => t.Length > 1)
            .Where(t => _memory.ContainsConcept(t))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(8)
            .ToArray();
    }

    private readonly record struct PlatonicQueryResolution(
        string ResponseText,
        double Confidence,
        ArithmeticOperation Operation,
        bool IsExplicitExpression);

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

    private enum ArithmeticOperation
    {
        Add,
        Subtract,
        Multiply,
        Divide
    }
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
