using GenesisNova.Axioms;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Train;

public sealed class GenesisTrainer
{
    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly PlatonicSpaceMemory _platonicSpace;
    private readonly SpaceManager _spaceManager;
    private readonly GenesisInferenceEngine _inferencePolicy;
    private readonly GenesisCompositeObjective _objective;
    private readonly FoldPathDiscovery _foldPathDiscovery;
    private readonly TransformAccumulator _transformAccumulator;
    private PlatonicState _tickState;
    private int _tickPatternPromotions;
    private int _trainStepCount;
    private double _cachedConservationLoss;
    private readonly Dictionary<string, int> _conceptCoverageCounts = new(StringComparer.OrdinalIgnoreCase);
    
    // Quality metrics caching (Performance optimization)
    private double? _cachedQualityLoss;
    private int _lastRelationCountForQuality;
    
    // Phase 1: BridgeConfidence metric (Theory validation)
    private BridgeConfidenceMetric? _bridgeConfidence;
    
    // Phase 2: Concept graph (Symbolic structure)
    private ConceptGraph? _conceptGraph;
    
    // Phase 3: Complement pairs (Conservation law)
    private ComplementPairManager? _complements;
    
    // Phase 4: Hypothesis tracking (Epistemic self-awareness)
    private HypothesisTracker? _hypotheses;
    
    // Phase 5: Transform library (Observable capabilities)
    private TransformLibrary? _transforms;

    public GenesisTrainer(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicSpaceMemory platonicSpace,
        GenesisNovaConfig? config = null,
        GenesisCompositeObjective? objective = null)
    {
        _tokenizer = tokenizer;
        _model = model;
        _platonicSpace = platonicSpace;
        var runtimeConfig = config ?? new GenesisNovaConfig();
        _foldPathDiscovery = new FoldPathDiscovery();
        _transformAccumulator = new TransformAccumulator(Math.Max(4, _platonicSpace.FaceDimension));
        _tickState = new PlatonicState(ImmutableArray<PlatonicElement>.Empty, _transformAccumulator.EmbeddingDimension);
        _spaceManager = new SpaceManager(_platonicSpace, new SpaceManagerSettings(
            Enabled: runtimeConfig.AutoManagePlatonicSpace,
            MaxNodes: Math.Max(256, runtimeConfig.MaxPlatonicNodes),
            MaxRelations: Math.Max(1_024, runtimeConfig.MaxPlatonicRelations)));
        _objective = objective ?? new GenesisCompositeObjective(
            TokenWeight: 1.0,
            RouteWeight: 0.3,
            ConsistencyWeight: 0.1,
            ConservationWeight: 0.1,
            MemoryWeight: 0.05);
        
        // Initialize all platonic reasoning phases
        InitializeGenesissPlatonicPhases();
        _inferencePolicy = new GenesisInferenceEngine(
            tokenizer,
            model,
            platonicSpace,
            foldPathDiscovery: _foldPathDiscovery,
            transformLibrary: _transforms,
            transformAccumulator: _transformAccumulator);
    }
    
    private void InitializeGenesissPlatonicPhases()
    {
        // Phase 1: BridgeConfidence metric - initialize with empty graph
        var emptyGraph = ImmutableDictionary<string, ImmutableHashSet<string>>.Empty;
        _bridgeConfidence = new BridgeConfidenceMetric(emptyGraph, _ => Array.Empty<(string, double)>());
        
        // Phase 2: Concept graph (symbolic structure)
        _conceptGraph = new ConceptGraph();
        
        // Phase 3: Complement pairs (conservation law)
        _complements = new ComplementPairManager();
        
        // Phase 4: Hypothesis tracking (epistemic self-awareness)
        _hypotheses = new HypothesisTracker();
        
        // Phase 5: Transform library (observable capabilities)
        _transforms = new TransformLibrary();
    }

    public int[] EncodeInput(string input)
        => _tokenizer.Encode(input);

    public int[] EncodeTarget(string output)
        => _tokenizer.Encode(output, addEos: true);

    public FoldPathDiscovery FoldPathDiscovery => _foldPathDiscovery;
    public TransformAccumulator TransformAccumulator => _transformAccumulator;
    public TransformLibrary? TransformLibrary => _transforms;
    public int TickPatternPromotions => _tickPatternPromotions;

    public GenesisStepLoss TrainStepPreTokenized(int[] inputTokens, int[] targetTokens)
    {
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var inputText = _tokenizer.Decode(inputTokens);
        var outputText = _tokenizer.Decode(targetTokens);
        var example = new GenesisExample(inputText, outputText);
        var weight = ComputeTrainingWeight(example);
        var baseLoss = _model.TrainExample(
            inputTokens,
            targetTokens,
            _tokenizer.BosTokenId,
            lossScale: weight);
        
        // Break computation graph after training
        _model.CloneParametersToBreakGraph();

        var concepts = ObserveLearningSignals(example);
        
        var consistencyLoss = EstimatePlatonicConsistency(concepts, IsNegativeText(outputText));
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = EstimateMemoryLoad();
        var routeLoss = EstimateUnifiedPathMismatch(example, targetTokens);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }

    public GenesisStepLoss TrainStep(GenesisExample example)
    {
        var inputTokens = _tokenizer.Encode(example.Input);
        var targetTokens = _tokenizer.Encode(example.Output, addEos: true);
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var weight = ComputeTrainingWeight(example);

        var baseLoss = _model.TrainExample(
            inputTokens,
            targetTokens,
            _tokenizer.BosTokenId,
            lossScale: weight);
         
        // Break computation graph after each example to allow sequential training
        _model.CloneParametersToBreakGraph();

        var concepts = ObserveLearningSignals(example);
        
        var consistencyLoss = EstimatePlatonicConsistency(concepts, IsNegativeText(example.Output));
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = EstimateMemoryLoad();
        var routeLoss = EstimateUnifiedPathMismatch(example, targetTokens);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }
 
    public void CloneParametersToBreakGraph()
        => _model.CloneParametersToBreakGraph();

    /// <summary>
    /// Train a batch of examples without cloning between them (cloning happens at epoch boundary).
    /// </summary>
    public GenesisStepLoss TrainBatchPreTokenized(IReadOnlyList<PreTokenizedExample> batch)
    {
       if (batch.Count == 0)
           return new GenesisStepLoss(0, 0, 0, 0, 0, 0);

       _model.EnsureVocabularySize(_tokenizer.VocabularySize);

       var totalTokenLoss = 0.0;
       var totalTokenWeight = 0.0;
       var consistencySum = 0.0;
       var memorySum = 0.0;
       var conservationSum = 0.0;
       var routeSum = 0.0;
       var qualitySum = 0.0;

       // Train each example and clone parameters between them to break computation graph
       for (int i = 0; i < batch.Count; i++)
       {
          var item = batch[i];
          var inputTokens = item.InputTokens;
          var targetTokens = item.TargetTokens;
          var example = item.Original;
          var weight = ComputeTrainingWeight(example);

          var loss = _model.TrainExample(
              inputTokens,
              targetTokens,
              _tokenizer.BosTokenId,
              lossScale: weight);
          totalTokenLoss += loss.TokenLoss * targetTokens.Length * weight;
          totalTokenWeight += targetTokens.Length * weight;
                 
          _trainStepCount++;

          var concepts = ObserveLearningSignals(example);
          consistencySum += EstimatePlatonicConsistency(concepts, IsNegativeText(example.Output));
          conservationSum += EstimateConservationDrift(inputTokens);
          memorySum += EstimateMemoryLoad();
          routeSum += EstimateUnifiedPathMismatch(example, targetTokens);
          qualitySum += ComputeQualityLoss();
              
          // Clone parameters between examples to break the computation graph
          // This prevents "backward through graph a second time" errors
          // REQUIRED: not optional, TorchSharp design limitation
          if (i < batch.Count - 1)
              _model.CloneParametersToBreakGraph();
       }

       var avgTokenLoss = totalTokenLoss / Math.Max(1.0, totalTokenWeight);
       var avgConsistency = consistencySum / Math.Max(1, batch.Count);
       var avgConservation = conservationSum / Math.Max(1, batch.Count);
       var avgMemory = memorySum / Math.Max(1, batch.Count);
       var avgRoute = routeSum / Math.Max(1, batch.Count);
       var avgQuality = qualitySum / Math.Max(1, batch.Count);
       
       // Integrate quality loss into total: 10% weight
       var total = (_objective.ComputeTotal(avgTokenLoss, avgRoute, avgConsistency, avgConservation, avgMemory) * 0.9) +
                   (avgQuality * 0.1);

       return new GenesisStepLoss(
          avgTokenLoss,
          avgRoute,
          avgConsistency,
          avgConservation,
          avgMemory,
          total);
    }

    private double EstimateUnifiedPathMismatch(GenesisExample example, IReadOnlyList<int> targetTokens)
    {
       var expected = targetTokens
           .Where(t => t != _tokenizer.EosTokenId && t != _tokenizer.BosTokenId && t != _tokenizer.PadTokenId)
           .ToArray();
       if (expected.Length == 0)
           return 0.0;

       var preview = _inferencePolicy.Generate(new GenerationRequest(example.Input, Math.Max(1, expected.Length)));
       return ComputeNormalizedTokenDistance(preview.GeneratedTokens, expected);
    }

    private static double ComputeNormalizedTokenDistance(IReadOnlyList<int> predicted, IReadOnlyList<int> expected)
    {
       if (expected.Count == 0)
           return 0.0;

       var overlap = Math.Min(predicted.Count, expected.Count);
       var mismatches = 0;
       for (var i = 0; i < overlap; i++)
       {
           if (predicted[i] != expected[i])
               mismatches++;
       }

       mismatches += Math.Abs(predicted.Count - expected.Count);
       return Clamp01((double)mismatches / Math.Max(1, expected.Count));
    }

    public double ComputeSchedulingPriority(GenesisExample example)
    {
       var weight = ComputeTrainingWeight(example);
       var concepts = ExtractConcepts(example);
       if (concepts.Count == 0)
           return weight;

       var rarityBoost = concepts
           .Select(c => _conceptCoverageCounts.TryGetValue(c, out var count) ? count : 0)
           .Select(count => 1.0 / (1.0 + count))
           .DefaultIfEmpty(1.0)
           .Average();
       return weight + (0.6 * rarityBoost);
    }

    private double EstimateConservationDrift(IReadOnlyList<int> inputTokens)
    {
        if (_trainStepCount % 128 != 0 && _cachedConservationLoss > 0)
            return _cachedConservationLoss;

        var snapshot = _model.Export();
        var emb = snapshot.Embeddings;
        if (emb.Length == 0)
            return 0.0;

        var rows = emb.GetLength(0);
        var cols = emb.GetLength(1);

        var sum = 0.0;
        var count = 0;
        foreach (var token in inputTokens.Distinct())
        {
            if (token < 0 || token >= rows)
                continue;
            for (var h = 0; h < cols; h++)
            {
                sum += Math.Abs(emb[token, h]);
                count++;
            }
        }

        if (count == 0)
        {
            var sampleRows = Math.Min(16, rows);
            for (var i = 0; i < sampleRows; i++)
            {
                var row = (i * 997 + _trainStepCount) % rows;
                for (var h = 0; h < cols; h++)
                {
                    sum += Math.Abs(emb[row, h]);
                    count++;
                }
            }
        }

        _cachedConservationLoss = (sum / Math.Max(1, count)) * 0.001;
        return _cachedConservationLoss;
    }

    private static IReadOnlyList<string> ExtractConcepts(string input, string output)
    {
        var words = $"{input} {output}"
            .ToLowerInvariant()
            .Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(NormalizeConceptToken)
            .Where(static w => w.Length > 0)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(24)
            .ToArray();

        return words.Length == 0 ? ["unknown"] : words;
    }

    private static IReadOnlyList<string> ExtractConcepts(GenesisExample example)
    {
        return ExtractConcepts(example.Input, example.Output);
    }

    private double ComputeTrainingWeight(GenesisExample example)
    {
        var weight = 1.0;
        var concepts = ExtractConcepts(example);
        
        // PHASE 1: BridgeConfidence metric (replaces arbitrary weights with ionization energy)
        if (_bridgeConfidence != null && concepts.Count > 0 && _conceptGraph != null)
        {
            // Update bridge confidence with current graph snapshot
            var graph = _conceptGraph.GetSnapshot();
            _bridgeConfidence = new BridgeConfidenceMetric(graph, _ => Array.Empty<(string, double)>());
            
            // Compute average bridge confidence for all concepts in the example
            var avgBridgeConfidence = _bridgeConfidence.ComputeAverageForConcepts(concepts);
            var ionizationEnergy = _bridgeConfidence.ComputeIonizationEnergyFromConfidence(avgBridgeConfidence);
            
            // Weight based on how much this concept needs practice (high ionization = low confidence = needs training)
            // TUNED: Reduced from 3.0 to 0.1 - phases should be subtle, not dominant
            weight += 0.1 * ionizationEnergy;
        }

        // PHASE 4: Use hypothesis tracking to boost weight on uncertain concepts
        if (_hypotheses != null && concepts.Count > 0)
        {
            var uncertainConcepts = concepts
                .Where(c => _hypotheses.GetHypothesisStatus(c) == HypothesisStatus.Conjecture)
                .Count();
            if (uncertainConcepts > 0)
            {
                // TUNED: Reduced from 0.5 to 0.1
                weight += 0.1 * ((double)uncertainConcepts / concepts.Count);
            }
        }

        // PHASE 3: Apply complement boost for conservation training
        if (_complements != null && IsNegativeText(example.Output))
        {
            // TUNED: Reduced from 1.0 to 0.2
            weight += 0.2;
        }

        // PHASE 5: Boost weight when learning new transforms
        if (_transforms != null && _transforms.Count < 20)
        {
            // TUNED: Reduced from 0.3 to 0.1
            weight += 0.1;
        }

        if (IsNegativeText(example.Output))
            weight += 1.0;

        if (concepts.Count >= 2)
        {
            var target = IsNegativeText(example.Output) ? 0.75 : 0.25;
            var current = EstimateAverageContradiction(concepts);
            var mismatch = Math.Abs(current - target);
            weight += 0.5 * Clamp01(mismatch);
        }

        return Math.Clamp(weight, 0.5, 4.0);
    }

    private static bool IsNegativeText(string text)
        => text.TrimStart().StartsWith("not ", StringComparison.OrdinalIgnoreCase);

    public void ObserveInferenceResult(string input, string output)
    {
        if (string.IsNullOrWhiteSpace(input) && string.IsNullOrWhiteSpace(output))
            return;

        var example = new GenesisExample(input ?? string.Empty, output ?? string.Empty);
        _ = ObserveLearningSignals(example, trackCoverage: false);
    }

    private IReadOnlyList<string> ObserveLearningSignals(GenesisExample example, bool trackCoverage = true)
    {
        var concepts = ExtractConcepts(example);
        ObservePlatonicSpace(example, concepts);
        UpdateTransformDiscovery(example);
        RunTickPatternLoop(example, concepts);

        if (trackCoverage)
        {
            foreach (var concept in concepts)
            {
                _conceptCoverageCounts.TryGetValue(concept, out var count);
                _conceptCoverageCounts[concept] = count + 1;
            }
        }

        if (_conceptGraph != null && concepts.Count > 0)
            _conceptGraph.ObserveExample(concepts, concepts);

        if (_hypotheses != null && concepts.Count > 0)
        {
            foreach (var concept in concepts)
                _hypotheses.ConfirmHypothesis(concept, example.Output);
        }

        if (_complements != null && concepts.Count > 0)
        {
            foreach (var concept in concepts)
                _ = _complements.GetComplement(concept);
        }

        return concepts;
    }

    private void UpdateTransformDiscovery(GenesisExample example)
    {
        if (!TryExtractArithmeticObservation(example, out var arithmetic))
            return;

        if (!double.TryParse(arithmetic.LeftToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var left) ||
            !double.TryParse(arithmetic.RightToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var right) ||
            !double.TryParse(arithmetic.ResultToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var output))
            return;

        _foldPathDiscovery.ObserveTrainingPair(arithmetic.OperationConcept, left, right, output);

        // After every 3 examples, attempt fold/log-linear discovery
        if (_trainStepCount % 3 == 0)
        {
            _foldPathDiscovery.TryRunDiscovery(
                arithmetic.OperationConcept,
                (op, a, b) => op switch
                {
                    "add" => (double?)(a + b),
                    "sub" => (double?)(a - b),
                    "mul" => (double?)(a * b),
                    "div" when b != 0 => (double?)(a / b),
                    _ => null
                });
        }

        var mode = _foldPathDiscovery.GetComposition(arithmetic.OperationConcept);
        var dim = _transformAccumulator.EmbeddingDimension;
        var inputEmbedding = InputEmbeddingComposer.ComposeInput(example.Input, mode, dim);
        var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(example.Output, dim);
        _transformAccumulator.Learn(arithmetic.OperationConcept, inputEmbedding, outputEmbedding);

        RegisterTransformCapability(arithmetic.OperationConcept);
    }

    private void RegisterTransformCapability(string operation)
    {
        if (_transforms is null)
            return;
        if (!_transformAccumulator.TryGetTransform(operation, out var transform))
            return;

        var vector = transform.Vector.Select(v => (float)v).ToArray();
        _transforms.RegisterTransform(operation, tensor(vector));
    }

    private void RunTickPatternLoop(GenesisExample example, IReadOnlyList<string> concepts)
    {
        if (concepts.Count == 0)
            return;

        var dimension = _tickState.EmbeddingDimension;
        _tickState = _tickState with { CurrentTick = _tickState.CurrentTick + 1 };

        var primarySymbol = concepts[0];
        var primaryId = EnsureTickElement(primarySymbol, ElementKind.Function);

        if (TryExtractArithmeticObservation(example, out var arithmetic))
        {
            var mode = _foldPathDiscovery.GetComposition(arithmetic.OperationConcept);
            var inputEmbedding = InputEmbeddingComposer.ComposeInput(example.Input, mode, dimension);
            var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(example.Output, dimension);
            var delta = new double[dimension];
            for (var i = 0; i < dimension; i++)
                delta[i] = outputEmbedding[i] - inputEmbedding[i];

            var (afterLearn, _) = TickExecutor.ExecuteTick(
                new TickAction(
                    Kind: TickKind.LocalLearn,
                    PrimaryElementId: primaryId,
                    Parameter: arithmetic.OperationConcept,
                    AuxiliaryEmbedding: delta),
                _tickState);
            _tickState = afterLearn;

            var leftId = EnsureTickElement(arithmetic.LeftToken, ElementKind.Object);
            var rightId = EnsureTickElement(arithmetic.RightToken, ElementKind.Object);
            var resultId = EnsureTickElement(arithmetic.ResultToken, ElementKind.Object);
            var (afterFold, foldElements) = TickExecutor.ExecuteTick(
                new TickAction(
                    Kind: TickKind.FoldChain,
                    PrimaryElementId: primaryId,
                    SecondaryIds: [leftId, rightId, resultId],
                    Parameter: arithmetic.OperationConcept),
                _tickState);
            _tickState = afterFold;
            PromoteDetectedPatterns(arithmetic.OperationConcept, foldElements);
            return;
        }

        var secondary = concepts.Skip(1).Take(2)
            .Select(c => EnsureTickElement(c, ElementKind.Object))
            .ToArray();
        if (secondary.Length >= 1)
        {
            var (updatedState, generated) = TickExecutor.ExecuteTick(
                new TickAction(
                    Kind: TickKind.FoldChain,
                    PrimaryElementId: primaryId,
                    SecondaryIds: secondary),
                _tickState);
            _tickState = updatedState;
            PromoteDetectedPatterns(primarySymbol, generated);
        }
    }

    private void PromoteDetectedPatterns(string operation, IReadOnlyList<PlatonicElement> candidates)
    {
        if (candidates.Count == 0)
            return;

        var state = _tickState;
        var existingPromotion = state.Elements.Any(e =>
            e.Kind == ElementKind.Function &&
            e.Symbol.Equals($"promoted:{operation}", StringComparison.OrdinalIgnoreCase));
        if (existingPromotion)
            return;

        var candidate = candidates
            .Where(e => e.Kind == ElementKind.Composition)
            .OrderByDescending(e => e.NoveltyScore)
            .FirstOrDefault();
        if (candidate is null || candidate.NoveltyScore < 0.75)
            return;

        var promoted = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Function,
            Embedding: candidate.Embedding.ToArray(),
            Symbol: $"promoted:{operation}",
            GeneratedAtTick: state.CurrentTick,
            NoveltyScore: candidate.NoveltyScore,
            BridgeConfidence: Math.Max(0.5, candidate.BridgeConfidence),
            RelatedTo: ImmutableArray.Create(candidate.Id),
            GenerationPath: $"tick-promote:{operation}");

        _tickState = state with
        {
            Elements = state.Elements.Add(promoted),
            NextId = state.NextId + 1
        };
        _tickPatternPromotions++;
    }

    private int EnsureTickElement(string symbol, ElementKind kind)
    {
        var existing = _tickState.Elements.FirstOrDefault(e =>
            e.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase) && e.Kind == kind);
        if (existing is not null)
            return existing.Id;

        var element = new PlatonicElement(
            Id: _tickState.NextId,
            Kind: kind,
            Embedding: InputEmbeddingComposer.GetInputEmbedding(symbol, _tickState.EmbeddingDimension),
            Symbol: symbol,
            GeneratedAtTick: _tickState.CurrentTick,
            NoveltyScore: 0.6,
            BridgeConfidence: 0.4,
            RelatedTo: ImmutableArray<int>.Empty,
            GenerationPath: "tick-observe");

        _tickState = _tickState with
        {
            Elements = _tickState.Elements.Add(element),
            NextId = _tickState.NextId + 1
        };

        return element.Id;
    }

    private void ObservePlatonicSpace(GenesisExample example, IReadOnlyList<string> concepts)
    {
        if (concepts.Count == 0)
            return;

        if (TryExtractArithmeticObservation(example, out var arithmetic))
            ObserveArithmeticFaces(arithmetic);

        var isNegative = IsNegativeText(example.Output);
        var inputConcepts = ExtractConcepts(example.Input, string.Empty);
        var outputConcepts = ExtractConcepts(string.Empty, example.Output);
        if (inputConcepts.Count == 0 || outputConcepts.Count == 0)
        {
            for (var i = 0; i < concepts.Count; i++)
            {
                for (var j = i + 1; j < concepts.Count; j++)
                {
                    var contradiction = isNegative ? 0.8 : 0.2;
                    _platonicSpace.ObserveContradiction(concepts[i], concepts[j], contradiction);
                }
            }
            _platonicSpace.FineEditFromExample(concepts, concepts, isNegative);
            return;
        }

        foreach (var left in inputConcepts)
        {
            foreach (var right in outputConcepts)
            {
                var contradiction = left.Equals(right, StringComparison.OrdinalIgnoreCase)
                    ? 0.05
                    : (isNegative ? 0.8 : 0.2);
                _platonicSpace.ObserveContradiction(left, right, contradiction);
            }
        }

        for (var i = 0; i < outputConcepts.Count; i++)
        {
            for (var j = i + 1; j < outputConcepts.Count; j++)
                _platonicSpace.ObserveContradiction(outputConcepts[i], outputConcepts[j], 0.1);
        }

        _platonicSpace.FineEditFromExample(inputConcepts, outputConcepts, isNegative);
    }

    private void ObserveArithmeticFaces(ArithmeticObservation arithmetic)
    {
        if (arithmetic.Operation is ArithmeticOperation.Add or ArithmeticOperation.Subtract)
        {
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:poly", 0.05);
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:log", 0.85);
        }
        else if (arithmetic.Operation is ArithmeticOperation.Multiply or ArithmeticOperation.Divide)
        {
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:poly", 0.85);
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:log", 0.05);
        }

        // Tighten arithmetic links when equation is numerically correct.
        var accuracyContradiction = Clamp01(arithmetic.AbsoluteError <= 1e-6 ? 0.05 : 0.6);
        _platonicSpace.ObserveContradiction(arithmetic.LeftToken, arithmetic.ResultToken, accuracyContradiction);
        _platonicSpace.ObserveContradiction(arithmetic.RightToken, arithmetic.ResultToken, accuracyContradiction);
    }

    private static bool TryExtractArithmeticObservation(GenesisExample example, out ArithmeticObservation observation)
    {
        observation = default;
        var input = example.Input ?? string.Empty;
        var output = example.Output ?? string.Empty;

        // Only handle compact symbol-based expressions (same pattern as inference).
        // Natural language forms are not recognized here — they route through the ML layer.
        var compact = System.Text.RegularExpressions.Regex.Match(
            input.ToLowerInvariant(),
            @"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)\s*$");
        if (!compact.Success)
            return false;

        if (!double.TryParse(compact.Groups[1].Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var left) ||
            !double.TryParse(compact.Groups[3].Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var right))
            return false;

        var outputMatch = System.Text.RegularExpressions.Regex.Match(output, @"-?\d+(\.\d+)?");
        if (!outputMatch.Success ||
            !double.TryParse(outputMatch.Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var actual))
            return false;

        ArithmeticOperation operation;
        string operationConcept;
        double expected;

        switch (compact.Groups[2].Value)
        {
            case "+":
                operation = ArithmeticOperation.Add;
                operationConcept = "add";
                expected = left + right;
                break;
            case "-":
                operation = ArithmeticOperation.Subtract;
                operationConcept = "sub";
                expected = left - right;
                break;
            case "*":
            case "x":
                operation = ArithmeticOperation.Multiply;
                operationConcept = "mul";
                expected = left * right;
                break;
            case "/":
                if (Math.Abs(right) <= 1e-12)
                    return false;
                operation = ArithmeticOperation.Divide;
                operationConcept = "div";
                expected = left / right;
                break;
            default:
                return false;
        }

        observation = new ArithmeticObservation(
            Operation: operation,
            OperationConcept: operationConcept,
            LeftToken: compact.Groups[1].Value,
            RightToken: compact.Groups[3].Value,
            ResultToken: outputMatch.Value,
            AbsoluteError: Math.Abs(expected - actual));
        return true;
    }

    private double EstimatePlatonicConsistency(IReadOnlyList<string> concepts, bool negativeExample)
    {
        if (concepts.Count < 2)
            return 0.0;

        var target = negativeExample ? 0.75 : 0.25;
        var avg = EstimateAverageContradiction(concepts);
        return Clamp01(Math.Abs(avg - target));
    }

    private double EstimateAverageContradiction(IReadOnlyList<string> concepts)
    {
        if (concepts.Count < 2)
            return 0.5;

        var total = 0.0;
        var pairs = 0;
        for (var i = 0; i < concepts.Count; i++)
        {
            for (var j = i + 1; j < concepts.Count; j++)
            {
                total += _platonicSpace.GetContradiction(concepts[i], concepts[j]);
                pairs++;
            }
        }
        return pairs == 0 ? 0.5 : total / pairs;
    }

    private double EstimateMemoryLoad()
    {
        var scale = Math.Max(1, _platonicSpace.NodeCount + _platonicSpace.RelationCount);
        return Clamp01(scale / 50_000.0);
    }

    private static string NormalizeConceptToken(string raw)
    {
        var cleaned = new string(raw.Where(ch =>
                char.IsLetterOrDigit(ch) || ch is '+' or '-' or '*' or '/' or '=' or '.')
            .ToArray());
        return cleaned.Trim();
    }

    private static Dictionary<string, double> BuildRelationMap(IEnumerable<PlatonicRelationSnapshot> relations)
    {
        var map = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var relation in relations)
            map[$"{relation.Left}|{relation.Right}"] = Clamp01(relation.SynthesisContradiction);
        return map;
    }

    private static double EstimateConceptTension(IReadOnlyList<string> concepts, IReadOnlyDictionary<string, double> relations)
    {
        if (concepts.Count < 2)
            return 0.0;

        var total = 0.0;
        var pairs = 0;
        for (var i = 0; i < concepts.Count; i++)
        {
            for (var j = i + 1; j < concepts.Count; j++)
            {
                var left = concepts[i].Trim().ToLowerInvariant();
                var right = concepts[j].Trim().ToLowerInvariant();
                var key = string.Compare(left, right, StringComparison.OrdinalIgnoreCase) <= 0
                    ? $"{left}|{right}"
                    : $"{right}|{left}";
                if (relations.TryGetValue(key, out var tension))
                    total += tension;
                else
                    total += 0.5;
                pairs++;
            }
        }

        return pairs == 0 ? 0.0 : total / pairs;
    }

     private double ComputeQualityLoss()
     {
         // Extract all relations from platonic space
         var allRelations = _platonicSpace.GetAllRelations();
         if (allRelations.Count == 0)
             return 0.0;

         // Cache: if relation count unchanged, reuse cached value
         if (_lastRelationCountForQuality == allRelations.Count && _cachedQualityLoss.HasValue)
             return _cachedQualityLoss.Value;

         // Compute average quality penalty across all operations in the space
         var qualitySum = 0.0;
         var operationSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
         
         foreach (var (left, right, obsCount) in allRelations)
         {
             operationSet.Add(left);
         }

         // For each operation, compute quality penalty
         foreach (var operation in operationSet)
         {
             // Get all targets for this operation
             var targetsForOp = allRelations
                 .Where(r => r.Left.Equals(operation, StringComparison.OrdinalIgnoreCase))
                 .ToList();

             if (targetsForOp.Count == 0)
                 continue;

             // Sample a relation to compute quality for
             var targetRel = targetsForOp.FirstOrDefault();
             var qualityPenalty = TransformQualityMetrics.ComputeQualityLossPenalty(
                 operation: operation,
                 target: targetRel.Right,
                 relationObservationCount: targetRel.ObservationCount,
                 allRelations: allRelations,
                 utilityThreshold: 0.3);
             qualitySum += qualityPenalty;
         }

         // Cache result and return
         var result = operationSet.Count > 0 ? qualitySum / operationSet.Count : 0.0;
         _lastRelationCountForQuality = allRelations.Count;
         _cachedQualityLoss = result;
         return result;
     }

     private static double Clamp01(double value)
         => Math.Max(0.0, Math.Min(1.0, value));

     public SpaceManagementResult ManagePlatonicSpace()
         => _spaceManager.Manage();

     public (int Nodes, int Relations) GetPlatonicSpaceSize()
         => (_platonicSpace.NodeCount, _platonicSpace.RelationCount);

     private readonly record struct ArithmeticObservation(
         ArithmeticOperation Operation,
         string OperationConcept,
         string LeftToken,
         string RightToken,
         string ResultToken,
         double AbsoluteError);

     private enum ArithmeticOperation
     {
         Add,
         Subtract,
         Multiply,
         Divide
     }
    
    // Platonic Reasoning Query Methods (Phases 1-5)
    
    /// <summary>
    /// Get what the system knows about a concept via bridge confidence.
    /// Low confidence = needs practice, High confidence = stable.
    /// </summary>
    public string GetBridgeConfidenceReport(IEnumerable<string> concepts)
    {
        if (_bridgeConfidence == null || _conceptGraph == null)
            return "Bridge confidence not available";
        
        var lines = new List<string>();
        foreach (var concept in concepts)
        {
            var confidence = _bridgeConfidence.ComputeForConcept(concept);
            var ionization = _bridgeConfidence.ComputeIonizationEnergyFromConfidence(confidence);
            var status = ionization > 0.7 ? "⚡ Needs practice" : 
                        ionization > 0.4 ? "◐ Exploring" : 
                        "✓ Stable";
            lines.Add($"  {concept}: confidence={confidence:P0}, ionization={ionization:P0} {status}");
        }
        
        return lines.Count == 0 
            ? "No concepts to report"
            : "Bridge Confidence:\n" + string.Join("\n", lines);
    }
    
    /// <summary>
    /// Get symbolic structure: what the system knows structurally (not just empirically).
    /// </summary>
    public string GetConceptGraphReport()
    {
        if (_conceptGraph == null)
            return "Concept graph not available";
        
        var report = _conceptGraph.GetStructuralReport();
        return $"Concept Graph ({_conceptGraph.GetRelationCount()} relations):\n{report}";
    }
    
    /// <summary>
    /// Get what complements exist (conservation law enforcement).
    /// </summary>
    public string GetComplementsReport()
    {
        if (_complements == null)
            return "Complements not available";
        
        var allComplements = _complements.GetAllComplements();
        if (allComplements.Count == 0)
            return "No complements registered yet";
        
        var lines = allComplements
            .Take(20)
            .Select(kvp => $"  {kvp.Key} ↔ {kvp.Value}")
            .ToList();
        
        if (allComplements.Count > 20)
            lines.Add($"  ... and {allComplements.Count - 20} more");
        
        return $"Complement Pairs ({allComplements.Count} total):\n" + string.Join("\n", lines);
    }
    
    /// <summary>
    /// Get epistemic self-awareness: what the system is certain about vs exploring.
    /// </summary>
    public string GetHypothesisReport()
    {
        if (_hypotheses == null)
            return "Hypothesis tracking not available";
        
        var axioms = _hypotheses.QueryWhatICertainlyKnow();
        var report = _hypotheses.GetStatusReport();
        
        return $"Epistemic Status:\n{report}\n\nAxioms (≥10 confirmations):\n{axioms}";
    }
    
    /// <summary>
    /// Get observable transforms: capabilities the system knows it has.
    /// </summary>
    public string GetTransformReport()
    {
        if (_transforms == null)
            return "Transform library not available";
        
        return _transforms.QueryKnownCapabilities();
    }
    
    /// <summary>
    /// Get comprehensive platonic reasoning state.
    /// </summary>
    public string GetPlatonicReasoningState(IEnumerable<string> conceptsToAnalyze)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("═══════════════════════════════════════");
        sb.AppendLine("PLATONIC REASONING STATE");
        sb.AppendLine("═══════════════════════════════════════");
        sb.AppendLine();
        sb.AppendLine(GetBridgeConfidenceReport(conceptsToAnalyze));
        sb.AppendLine();
        sb.AppendLine(GetConceptGraphReport());
        sb.AppendLine();
        sb.AppendLine(GetComplementsReport());
        sb.AppendLine();
        sb.AppendLine(GetHypothesisReport());
        sb.AppendLine();
        sb.AppendLine(GetTransformReport());
        sb.AppendLine();
        sb.AppendLine("═══════════════════════════════════════");
        
        return sb.ToString();
    }
}

public sealed record GenesisStepLoss(
    double TokenLoss,
    double RouteLoss,
    double ConsistencyLoss,
    double ConservationLoss,
    double MemoryLoss,
    double TotalLoss);
