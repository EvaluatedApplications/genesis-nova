using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public class GenesisNeuralModel
{
    private GenesisNovaConfig _config;
    private readonly Device _trainingDevice;  // CPU for stability & large models
    private readonly Device _inferenceDevice; // GPU for speed if available
    private int _hiddenSize;
    private int _vocabSize;

    private TorchSharp.Modules.Parameter? _embT;
    private TorchSharp.Modules.Parameter? _wOutT;
    private TorchSharp.Modules.Parameter? _bOutT;
    private torch.optim.Optimizer? _optimizer;

    private const int NumRoutes = 2;
    private const double RouteLossWeight = 0.3;
    private TorchSharp.Modules.Parameter? _routeWT;
    private TorchSharp.Modules.Parameter? _routeB;

    public GenesisNeuralModel(GenesisNovaConfig config)
    {
        _config = config;
        _hiddenSize = config.HiddenSize;
        
        // Training on CPU: unlimited memory, stable
        _trainingDevice = CPU;
        
        // Inference on GPU if available, else CPU
        _inferenceDevice = (torch.cuda_is_available() && config.Backend != ComputeBackend.Cpu) ? CUDA : CPU;
    }

    public int HiddenSize => _hiddenSize;
    public int VocabularySize => _vocabSize;

    public void UpdateConfig(GenesisNovaConfig newConfig)
    {
        _config = newConfig;
    }

    public void EnsureVocabularySize(int vocabSize)
    {
        EnsureVocabularySizeGpu(vocabSize);
    }

    public void EnsureHiddenSize(int hiddenSize)
    {
        if (hiddenSize <= HiddenSize)
            return;

        var oldHidden = HiddenSize;
        var vocab = VocabularySize;
        EnsureHiddenSizeGpu(hiddenSize, oldHidden, vocab);
    }

    public int PredictNextToken(
        IReadOnlyList<int> inputTokens,
        int previousToken,
        int stepIndex = 0,
        int? disallowToken = null,
        IReadOnlyCollection<int>? penalizedTokens = null,
        double repetitionPenalty = 0.0,
        IReadOnlyDictionary<int, double>? tokenBiases = null)
    {
        return PredictNextTokenGpu(inputTokens, previousToken, stepIndex, disallowToken, penalizedTokens, repetitionPenalty, tokenBiases);
    }

    public TrainingLoss TrainExample(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId,
        double lossScale = 1.0)
    {
        return TrainExampleGpu(inputTokens, targetTokens, bosTokenId, lossScale);
    }

    /// <summary>Forward pass only - accumulates loss, no backward yet.</summary>
    public (torch.Tensor AccumulatedLoss, List<torch.Tensor> ForwardTensors) ForwardPass(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId)
    {
        EnsureModelInitialized();

        var forwardTensors = new List<torch.Tensor>();
        torch.Tensor? accumulatedLoss = null;
        var prev = bosTokenId;

        for (var t = 0; t < targetTokens.Count; t++)
        {
            var inputVec = MeanEmbeddingTensor(inputTokens);
            forwardTensors.Add(inputVec);
            
            var prevEmb = GetEmbeddingTensor(prev);
            forwardTensors.Add(prevEmb);
            
            var decoderScale = PositionScale(t);
            
            var scaled = prevEmb * decoderScale;
            forwardTensors.Add(scaled);
            
            var combined = inputVec + scaled;
            forwardTensors.Add(combined);
            
            var hidden = combined.tanh();
            forwardTensors.Add(hidden);
            
            var logits = hidden.matmul(_wOutT!) + _bOutT!;
            forwardTensors.Add(logits);
            
            var targetToken = tensor(new long[] { targetTokens[t] }, dtype: ScalarType.Int64, device: _trainingDevice);
            forwardTensors.Add(targetToken);
            
            var stepLoss = nn.functional.cross_entropy(logits.unsqueeze(0), targetToken);
            forwardTensors.Add(stepLoss);
            
            if (accumulatedLoss is null)
                accumulatedLoss = stepLoss;
            else
            {
                var newAccum = accumulatedLoss + stepLoss;
                accumulatedLoss = newAccum;
            }
            
            prev = targetTokens[t];
        }

        return (accumulatedLoss!, forwardTensors);
    }

    /// <summary>Backward pass on accumulated loss.</summary>
    public void BackwardAndStep(torch.Tensor accumulatedLoss)
    {
        _optimizer!.zero_grad();
        accumulatedLoss.backward();
        _optimizer.step();
    }

    public ModelSnapshot Export()
    {
        EnsureModelInitialized();
        return new ModelSnapshot(
            TensorToMatrix(_embT!),
            TensorToMatrix(_wOutT!),
            TensorToVector(_bOutT!),
            _routeWT is not null ? TensorToMatrix(_routeWT) : null,
            _routeB is not null ? TensorToVector(_routeB) : null);
    }

    public void Import(ModelSnapshot snapshot)
    {
        _hiddenSize = snapshot.Embeddings.GetLength(1);
        _vocabSize = snapshot.Embeddings.GetLength(0);

        DisposeParameters();
        _embT = MatrixToParameter(snapshot.Embeddings);
        _wOutT = MatrixToParameter(snapshot.OutputWeights);
        _bOutT = VectorToParameter(snapshot.OutputBias);
        if (snapshot.RouteWeights is not null && snapshot.RouteBias is not null)
        {
            _routeWT = MatrixToParameter(snapshot.RouteWeights);
            _routeB = VectorToParameter(snapshot.RouteBias);
        }
        RecreateOptimizer();
    }

    private int PredictNextTokenGpu(
        IReadOnlyList<int> inputTokens,
        int previousToken,
        int stepIndex,
        int? disallowToken,
        IReadOnlyCollection<int>? penalizedTokens,
        double repetitionPenalty,
        IReadOnlyDictionary<int, double>? tokenBiases)
    {
        EnsureModelInitialized();
        using var noGrad = no_grad();
        using var inputVec = MeanEmbeddingTensor(inputTokens);
        using var prevEmb = GetEmbeddingTensor(previousToken);
        var decoderScale = PositionScale(stepIndex);
        using var hidden = (inputVec + (prevEmb * decoderScale)).tanh();
        using var logits = hidden.matmul(_wOutT!) + _bOutT!;

        var scores = logits.cpu().data<float>().ToArray();
        if (disallowToken.HasValue && disallowToken.Value >= 0 && disallowToken.Value < scores.Length)
            scores[disallowToken.Value] = float.NegativeInfinity;

        if (repetitionPenalty > 0.0 && penalizedTokens is not null)
        {
            foreach (var token in penalizedTokens)
            {
                if (token >= 0 && token < scores.Length)
                    scores[token] -= (float)repetitionPenalty;
            }
        }

        if (tokenBiases is not null)
        {
            foreach (var (token, bias) in tokenBiases)
            {
                if (token >= 0 && token < scores.Length)
                    scores[token] += (float)bias;
            }
        }

        return ArgMax(scores);
    }

    private TrainingLoss TrainExampleGpu(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId,
        double lossScale)
    {
        EnsureModelInitialized();

        double totalLoss = 0.0;
        var prev = bosTokenId;
        
        var exStart = System.Diagnostics.Stopwatch.StartNew();
        
        // Forward pass: accumulate losses WITHOUT disposing tensors
        // This builds the full computation graph from BOS to EOS
        torch.Tensor? accumulatedLoss = null;
        var forwardTensors = new List<torch.Tensor>();
        var stepLosses = new List<torch.Tensor>();

        try
        {
            for (var t = 0; t < targetTokens.Count; t++)
            {
                // Get embeddings
                var inputVec = MeanEmbeddingTensor(inputTokens);
                forwardTensors.Add(inputVec);
                
                var prevEmb = GetEmbeddingTensor(prev);
                forwardTensors.Add(prevEmb);
                
                var decoderScale = PositionScale(t);
                
                // Forward computation: input + pos-scaled embedding
                var scaled = prevEmb * decoderScale;
                forwardTensors.Add(scaled);
                
                var combined = inputVec + scaled;
                forwardTensors.Add(combined);
                
                var hidden = combined.tanh();
                forwardTensors.Add(hidden);
                
                // Logits
                var logits = hidden.matmul(_wOutT!) + _bOutT!;
                forwardTensors.Add(logits);
                
                // Loss for this token
                var targetToken = tensor(new long[] { targetTokens[t] }, dtype: ScalarType.Int64, device: _trainingDevice);
                forwardTensors.Add(targetToken);
                var stepLoss = nn.functional.cross_entropy(logits.unsqueeze(0), targetToken);
                forwardTensors.Add(stepLoss);
                stepLosses.Add(stepLoss);
                
                totalLoss += stepLoss.ToDouble();
                
                prev = targetTokens[t];
            }

            if (stepLosses.Count > 1)
            {
                var stackedLoss = stack(stepLosses.ToArray());
                forwardTensors.Add(stackedLoss);
                accumulatedLoss = stackedLoss.sum();
                forwardTensors.Add(accumulatedLoss);
            }
            else if (stepLosses.Count == 1)
            {
                accumulatedLoss = stepLosses[0];
            }

            exStart.Stop();
            var backStart = System.Diagnostics.Stopwatch.StartNew();
            
            // L2 regularization to prevent weight bloat
            if (accumulatedLoss is not null && _config.L2RegularizationCoefficient > 0)
            {
                var l2Penalty = ComputeL2Penalty();
                forwardTensors.Add(l2Penalty);
                accumulatedLoss = accumulatedLoss + l2Penalty;
            }
            
            // Backward pass: single step for entire sequence
            if (accumulatedLoss is not null)
            {
                _optimizer!.zero_grad();
                if (Math.Abs(lossScale - 1.0) > 1e-9)
                {
                    using var scaledLoss = accumulatedLoss * lossScale;
                    scaledLoss.backward();
                }
                else
                {
                    accumulatedLoss.backward();
                }
                _optimizer.step();
            }
            
            backStart.Stop();
            
            // Log if slow
            if (exStart.ElapsedMilliseconds + backStart.ElapsedMilliseconds > 20)
                System.Diagnostics.Debug.WriteLine($"[GPU-TIME] seqlen={targetTokens.Count} fwd={exStart.ElapsedMilliseconds}ms bwd={backStart.ElapsedMilliseconds}ms total={exStart.ElapsedMilliseconds + backStart.ElapsedMilliseconds}ms");
        }
        finally
        {
            // Dispose all forward tensors
            foreach (var t in forwardTensors)
            {
                try { t?.Dispose(); } catch { }
            }
            try { accumulatedLoss?.Dispose(); } catch { }
        }

        return new TrainingLoss(totalLoss / Math.Max(1, targetTokens.Count), RouteLoss: 0.0);
    }
    
    /// <summary>
    /// Compute L2 regularization penalty on all trainable weights.
    /// Prevents weight bloat by penalizing large weight magnitudes.
    /// </summary>
    private torch.Tensor ComputeL2Penalty()
    {
        var penalty = torch.tensor(0.0, device: _trainingDevice);
        
        // L2 penalty on embedding weights
        if (_embT is not null)
            penalty = penalty + (_embT * _embT).sum();
        
        // L2 penalty on output weights
        if (_wOutT is not null)
            penalty = penalty + (_wOutT * _wOutT).sum();
        
        // L2 penalty on output bias
        if (_bOutT is not null)
            penalty = penalty + (_bOutT * _bOutT).sum();
        
        // L2 penalty on route head weights
        if (_routeWT is not null)
            penalty = penalty + (_routeWT * _routeWT).sum();
        
        // L2 penalty on route head bias
        if (_routeB is not null)
            penalty = penalty + (_routeB * _routeB).sum();
        
        // Apply lambda coefficient and normalize by count
        return penalty * _config.L2RegularizationCoefficient * 0.5;
    }
    
    /// <summary>
    /// Generate a platonic query from input text.
    /// This is a simplified version that:
    /// 1. Encodes the input
    /// 2. Creates a query with that encoding as the operation embedding
    /// 3. Marks all tokens as operands
    /// 4. Returns the input encoding as the predicted result
    /// 
    /// A full implementation would have 3 separate heads with attention mechanisms.
    /// </summary>
    public GenesisNova.Inference.PlatonicQuery GenerateQuery(
        IReadOnlyList<int> inputTokens,
        int bosTokenId = 0)
    {
        using var noGrad = TorchSharp.torch.no_grad();
        EnsureModelInitialized();
        
        // Encode input
        var inputEmbedding = MeanEmbeddingTensor(inputTokens).detach().cpu().data<float>().ToArray().Select(x => (double)x).ToArray();
        
        // Generate operand mask (all tokens are operands for now)
        var operandMask = Enumerable.Repeat(true, inputTokens.Count).ToArray();
        
        // Operand embeddings: just use input embedding repeated
        var operandEmbeddings = new[] { inputEmbedding };
        
        // Operation embedding: normalized input
        var operationEmbedding = Normalize(inputEmbedding);
        
        return new(
            OperationEmbedding: operationEmbedding,
            OperationConfidence: 0.5,  // Medium confidence for now
            OperandMask: operandMask,
            OperandEmbeddings: operandEmbeddings,
            PredictedResultEmbedding: inputEmbedding);
    }
    
    private static double[] Normalize(double[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(x => x * x));
        if (norm < 1e-10)
            return vector;
        return vector.Select(x => x / norm).ToArray();
    }
    
    /// <summary>Clone parameters to break computation graph chain after epoch.</summary>
    public void CloneParametersToBreakGraph()
    {
        using (no_grad())
        {
            var embClone = _embT!.clone().detach().to(_trainingDevice);
            var wClone = _wOutT!.clone().detach().to(_trainingDevice);
            var bClone = _bOutT!.clone().detach().to(_trainingDevice);

            ReplaceModelParameters(
                new TorchSharp.Modules.Parameter(embClone, true),
                new TorchSharp.Modules.Parameter(wClone, true),
                new TorchSharp.Modules.Parameter(bClone, true));

            if (_routeWT is not null)
            {
                var rwClone = _routeWT.clone().detach().to(_trainingDevice);
                var rbClone = _routeB!.clone().detach().to(_trainingDevice);
                var oldRW = _routeWT;
                var oldRB = _routeB;
                _routeWT = new TorchSharp.Modules.Parameter(rwClone, true);
                _routeB = new TorchSharp.Modules.Parameter(rbClone, true);
                try { oldRW.Dispose(); } catch { }
                try { oldRB.Dispose(); } catch { }
            }

            RecreateOptimizer();
        }
    }

    private void EnsureVocabularySizeGpu(int vocabSize)
    {
        if (vocabSize <= _vocabSize)
            return;

        var oldVocab = _vocabSize;
        var hidden = HiddenSize;

        var newEmb = ((rand(new long[] { vocabSize, hidden }, device: _trainingDevice) * 2.0) - 1.0) * 0.05;
        if (_embT is not null && oldVocab > 0)
            newEmb.slice(0, 0, oldVocab, 1).copy_(_embT);
        var newEmbP = new TorchSharp.Modules.Parameter(newEmb, true);

        var newWOut = ((rand(new long[] { hidden, vocabSize }, device: _trainingDevice) * 2.0) - 1.0) * 0.05;
        if (_wOutT is not null && oldVocab > 0)
            newWOut.slice(1, 0, oldVocab, 1).copy_(_wOutT);
        var newWOutP = new TorchSharp.Modules.Parameter(newWOut, true);

        var newBOut = zeros(new long[] { vocabSize }, dtype: ScalarType.Float32, device: _trainingDevice);
        if (_bOutT is not null && oldVocab > 0)
            newBOut.slice(0, 0, oldVocab, 1).copy_(_bOutT);
        var newBOutP = new TorchSharp.Modules.Parameter(newBOut, true);

        ReplaceModelParameters(newEmbP, newWOutP, newBOutP);

        _vocabSize = vocabSize;
        RecreateOptimizer();
    }

    private Tensor MeanEmbeddingTensor(IReadOnlyList<int> inputTokens)
    {
        EnsureModelInitialized();
        if (inputTokens.Count == 0 || _vocabSize == 0)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice);

        var idx = inputTokens.Select(i => (long)i).ToArray();
        using var idxTensor = tensor(idx, dtype: ScalarType.Int64, device: _trainingDevice);
        using var embRows = _embT!.index_select(0, idxTensor);
        
        // Sequential processing with lighter decay: exponential smoothing (alpha=0.3)
        // This preserves more information from all tokens, not just early ones
        Tensor result = zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice);
        const float alpha = 0.3f;  // Lighter exponential decay
        
        for (var i = 0; i < inputTokens.Count; i++)
        {
            using var emb = embRows.index_select(0, tensor(new long[] { i }, dtype: ScalarType.Int64, device: _trainingDevice)).squeeze(0);
            using var prev = result;
            using var decayed = prev * (1.0f - alpha);
            using var combined = decayed + (emb * alpha);
            result = combined.tanh();
        }
        
        return result;
    }

    private Tensor GetEmbeddingTensor(int token)
    {
        EnsureModelInitialized();
        if (token < 0 || token >= _vocabSize)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice);
        using var idx = tensor(new long[] { token }, dtype: ScalarType.Int64, device: _trainingDevice);
        return _embT!.index_select(0, idx).squeeze(0);
    }

    private void EnsureModelInitialized()
    {
        if (_embT is not null)
            return;

        EnsureVocabularySizeGpu(Math.Max(3, _vocabSize));
    }

    private void EnsureRouteHeadInitialized()
    {
        if (_routeWT is not null)
            return;

        _routeWT = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { _hiddenSize, NumRoutes }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _routeB = new TorchSharp.Modules.Parameter(
            zeros(new long[] { NumRoutes }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        RecreateOptimizer();
    }

    public (int RouteId, double Confidence) PredictRoute(IReadOnlyList<int> inputTokens)
    {
        EnsureModelInitialized();
        EnsureRouteHeadInitialized();
        using var noGrad = no_grad();
        using var meanEmb = MeanEmbeddingTensor(inputTokens);
        using var hidden = meanEmb.tanh();
        using var logits = hidden.matmul(_routeWT!) + _routeB!;
        using var probs = nn.functional.softmax(logits, 0);
        var scores = probs.cpu().data<float>().ToArray();
        var routeId = scores[1] > scores[0] ? 1 : 0;
        var confidence = (double)(routeId == 1 ? scores[1] : scores[0]);
        return (routeId, confidence);
    }

    private void EnsureHiddenSizeGpu(int hiddenSize, int oldHidden, int vocab)
    {
        var newEmb = ((rand(new long[] { vocab, hiddenSize }, device: _trainingDevice) * 2.0) - 1.0) * 0.05;
        if (_embT is not null && oldHidden > 0)
            newEmb.slice(1, 0, oldHidden, 1).copy_(_embT);
        var newEmbP = new TorchSharp.Modules.Parameter(newEmb, true);

        var newWOut = ((rand(new long[] { hiddenSize, vocab }, device: _trainingDevice) * 2.0) - 1.0) * 0.05;
        if (_wOutT is not null && oldHidden > 0)
            newWOut.slice(0, 0, oldHidden, 1).copy_(_wOutT);
        var newWOutP = new TorchSharp.Modules.Parameter(newWOut, true);

        var newBOutP = _bOutT is null
            ? new TorchSharp.Modules.Parameter(zeros(new long[] { vocab }, dtype: ScalarType.Float32, device: _trainingDevice), true)
            : new TorchSharp.Modules.Parameter(_bOutT.clone().detach().to(_trainingDevice), true);

        ReplaceModelParameters(newEmbP, newWOutP, newBOutP);

        _hiddenSize = hiddenSize;

        // Route head shape is [oldHidden, NumRoutes] — incompatible after resize; reinitialize.
        if (_routeWT is not null)
        {
            var oldRW = _routeWT;
            var oldRB = _routeB;
            _routeWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { hiddenSize, NumRoutes }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _routeB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { NumRoutes }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            try { oldRW.Dispose(); } catch { }
            try { oldRB?.Dispose(); } catch { }
        }

        RecreateOptimizer();
    }

    private void RecreateOptimizer()
    {
        _optimizer?.Dispose();
        var parameters = new List<TorchSharp.Modules.Parameter> { _embT!, _wOutT!, _bOutT! };
        if (_routeWT is not null) parameters.Add(_routeWT);
        if (_routeB is not null) parameters.Add(_routeB!);
        _optimizer = torch.optim.SGD(parameters, _config.LearningRate);
    }

    private void ReplaceModelParameters(
        TorchSharp.Modules.Parameter emb,
        TorchSharp.Modules.Parameter wOut,
        TorchSharp.Modules.Parameter bOut)
    {
        var oldEmb = _embT;
        var oldWOut = _wOutT;
        var oldBOut = _bOutT;

        _embT = emb;
        _wOutT = wOut;
        _bOutT = bOut;

        try { oldEmb?.Dispose(); } catch { }
        try { oldWOut?.Dispose(); } catch { }
        try { oldBOut?.Dispose(); } catch { }
    }

    private void DisposeParameters()
    {
        _optimizer?.Dispose();
        _optimizer = null;
        try { _embT?.Dispose(); } catch { }
        try { _wOutT?.Dispose(); } catch { }
        try { _bOutT?.Dispose(); } catch { }
        _embT = null;
        _wOutT = null;
        _bOutT = null;
        try { _routeWT?.Dispose(); } catch { }
        try { _routeB?.Dispose(); } catch { }
        _routeWT = null;
        _routeB = null;
    }

    private static int ArgMax(IReadOnlyList<float> values)
    {
        var idx = 0;
        var best = values[0];
        for (var i = 1; i < values.Count; i++)
        {
            if (values[i] <= best)
                continue;
            best = values[i];
            idx = i;
        }
        return idx;
    }

    private static double PositionScale(int index)
    {
        if (index <= 0)
            return 1.0;
        return 1.0 / (1.0 + (0.25 * index));
    }

    private double[,] TensorToMatrix(Tensor tensor)
    {
        using var cpu = tensor.detach().cpu();
        var rows = (int)cpu.shape[0];
        var cols = (int)cpu.shape[1];
        var flat = cpu.data<float>().ToArray();
        var matrix = new double[rows, cols];
        var k = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                matrix[i, j] = flat[k++];
        return matrix;
    }

    private double[] TensorToVector(Tensor tensor)
    {
        using var cpu = tensor.detach().cpu();
        var flat = cpu.data<float>().ToArray();
        return flat.Select(x => (double)x).ToArray();
    }

    private TorchSharp.Modules.Parameter MatrixToParameter(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var values = new float[rows * cols];
        var k = 0;
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                values[k++] = (float)matrix[i, j];
        return new TorchSharp.Modules.Parameter(
            tensor(values, dtype: ScalarType.Float32, device: _trainingDevice).reshape(rows, cols),
            true);
    }

    private TorchSharp.Modules.Parameter VectorToParameter(double[] vector)
    {
        var values = vector.Select(x => (float)x).ToArray();
        return new TorchSharp.Modules.Parameter(tensor(values, dtype: ScalarType.Float32, device: _trainingDevice), true);
    }
}

public sealed record TrainingLoss(double TokenLoss, double RouteLoss = 0.0)
{
    public double Total => TokenLoss + RouteLoss;
}

public sealed record ModelSnapshot(
    double[,] Embeddings,
    double[,] OutputWeights,
    double[] OutputBias,
    double[,]? RouteWeights = null,
    double[]? RouteBias = null);
