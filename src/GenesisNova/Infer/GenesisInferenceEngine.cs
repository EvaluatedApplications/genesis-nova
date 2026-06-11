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
    private readonly object _platonicFileCacheLock = new();
    private string? _cachedPlatonicFilePath;
    private DateTime _cachedPlatonicFileWriteUtc;
    private IReadOnlyCollection<string>? _cachedPlatonicFileConcepts;
    private readonly object _telemetryLock = new();
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
        TransformAccumulator? transformAccumulator = null)
    {
        _tokenizer = tokenizer;
        _model = model;
        _memory = memory;
        _platonicFilePathProvider = platonicFilePathProvider;
        _foldPathDiscovery = foldPathDiscovery;
        _transformLibrary = transformLibrary;
        _transformAccumulator = transformAccumulator;
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
            TransformIntercept: transformIntercept);
    }

    private GenerationResult GenerateSinglePass(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens)
    {
        var (routeId, _) = _model.PredictRoute(inputTokens);
        if (routeId == 1)
        {
            if (TryGenerateFromDiscoveredTransform(request, out var discovered))
                return discovered;
            if (TryGenerateFromPlatonicPlan(request, out var platonic))
                return platonic;
            // Platonic route attempted but no tool fired — fall through to neural.
        }

        return GenerateNeuralTokens(request, inputTokens, neuralFallback: routeId == 1);
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
        var totalBiasCount = 0;
        var biasMagnitudeSum = 0.0;
        var prev = _tokenizer.BosTokenId;
        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            double biasMagnitude;
            IReadOnlyDictionary<int, double>? biases;
            if (checkpointContextConcepts.Count > 0)
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, checkpointContextConcepts, out biasMagnitude);
            else
                biases = BuildTokenBiases(inputTokens, generated, adaptiveBiasScale, out biasMagnitude);

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
            PlatonicHopCount: 0);
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

        result = new GenerationResult(
            Output: _tokenizer.Decode(tokens),
            GeneratedTokens: tokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: "platonic-discovered-transform",
            PlatonicConfidence: 0.92,
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
            beamWidth: beamWidth);
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
            PlatonicHopCount: conceptResult.Hops);
        return true;
    }

    private bool TryResolveArithmeticQuery(string input, out PlatonicQueryResolution resolution)
    {
        resolution = default;
        if (!TryParseArithmeticInput(input, out var left, out var right, out var op, out var likelyContextWrapped, out var isExplicitArithmeticExpression))
            return false;
        if (likelyContextWrapped)
            return false;

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
        if (_transformAccumulator is null)
            return;

        var dim = _transformAccumulator.EmbeddingDimension;
        var mode = _foldPathDiscovery?.GetComposition(operation) ?? CompositionMode.Sum;
        var inputEmbedding = InputEmbeddingComposer.ComposeInput(input, mode, dim);
        var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(output, dim);
        _transformAccumulator.Learn(operation, inputEmbedding, outputEmbedding);
    }

    /// <summary>
    /// Performs arithmetic in face embedding space.
    /// Add/sub: sum poly sub-faces → decode result = blended[0] * 10.
    /// Mul/div: sum log sub-faces → decode result = sign * exp(blended[logStart] * 10).
    /// This is not a learned behaviour — it exploits the homomorphic structure of seeded faces.
    /// </summary>
    private bool TryFaceArithmetic(double left, double right, bool additiveFace, double rightFaceSign, out double result, out double quality)
    {
        result = 0;
        quality = 0;
        var numericDims = _memory.NumericDimensions;
        var logStart = _memory.LogFaceStart;

        var leftKey = FormatNumber(left);
        var rightKey = FormatNumber(right);

        if (!_memory.TryGetConceptFace(leftKey, out var faceA) ||
            !_memory.TryGetConceptFace(rightKey, out var faceB))
            return false;

        if (additiveFace)
        {
            if (faceA.Length < numericDims || faceB.Length < numericDims)
                return false;
            var reconstructed = new List<double>(numericDims);
            for (var i = 0; i < numericDims; i++)
            {
                var blended = faceA[i] + rightFaceSign * faceB[i];
                reconstructed.Add(blended * Math.Pow(10, i + 1));
            }
            result = reconstructed[0];
            quality = ComputeFaceDecodeQuality(reconstructed);
        }
        else
        {
            if (faceA.Length <= logStart || faceB.Length <= logStart)
                return false;
            // Zero is outside log-space; preserve exact arithmetic behavior for zero cases.
            if (Math.Abs(left) < 1e-12 || (rightFaceSign > 0.0 && Math.Abs(right) < 1e-12))
            {
                result = 0;
                quality = 1.0;
                return true;
            }

            var logDims = Math.Min(numericDims, Math.Min(faceA.Length - logStart, faceB.Length - logStart));
            if (logDims <= 0)
                return false;

            var magnitudes = new List<double>(logDims);
            for (var i = 0; i < logDims; i++)
            {
                var blendedLog = (faceA[logStart + i] + rightFaceSign * faceB[logStart + i]) * Math.Pow(10, i + 1);
                magnitudes.Add(Math.Exp(blendedLog));
            }

            var sign = Math.Sign(left) * Math.Sign(right);
            result = sign * magnitudes[0];
            quality = ComputeFaceDecodeQuality(magnitudes);
        }

        return !double.IsNaN(result) && !double.IsInfinity(result) && !double.IsNaN(quality) && !double.IsInfinity(quality);
    }

    private static double ComputeFaceDecodeQuality(IReadOnlyList<double> values)
    {
        if (values.Count <= 1)
            return 1.0;
        var mean = values.Average();
        var meanAbs = Math.Max(Math.Abs(mean), 1e-12);
        var variance = values.Sum(v => Math.Pow(v - mean, 2)) / values.Count;
        var std = Math.Sqrt(variance);
        var cv = std / meanAbs;
        return Math.Clamp(1.0 - cv, 0.0, 1.0);
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
        => BuildTokenBiases(inputTokens, generatedTokens, strength, checkpointContextConcepts: null, out averageBiasMagnitude);

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> generatedTokens,
        double strength,
        IReadOnlyCollection<string>? checkpointContextConcepts,
        out double averageBiasMagnitude)
    {
        averageBiasMagnitude = 0.0;
        var contextConcepts = BuildContextConcepts(inputTokens, generatedTokens, checkpointContextConcepts);
        if (contextConcepts.Count == 0)
            return null;

        var biases = new Dictionary<int, double>();
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
            foreach (var concept in contextConcepts)
            {
                total += _memory.GetContradiction(candidate, concept);
                count++;
            }

            if (count == 0)
                continue;

            var avg = total / count;
            var bias = Math.Clamp((0.5 - avg) * (0.9 * Math.Max(0.0, strength)), -0.9, 0.9);
            if (Math.Abs(bias) > 1e-6)
                biases[token] = bias;
        }

        if (biases.Count == 0)
            return null;

        averageBiasMagnitude = biases.Values.Average(v => Math.Abs(v));
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
