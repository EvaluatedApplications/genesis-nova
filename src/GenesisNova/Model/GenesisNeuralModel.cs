using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public sealed class GenesisNeuralModel
{
    private readonly GenesisNovaConfig _config;
    private readonly Device _device;
    private int _hiddenSize;
    private int _vocabSize;

    private TorchSharp.Modules.Parameter? _embT;
    private TorchSharp.Modules.Parameter? _wOutT;
    private TorchSharp.Modules.Parameter? _bOutT;
    private torch.optim.Optimizer? _optimizer;

    public GenesisNeuralModel(GenesisNovaConfig config)
    {
        _config = config;
        _hiddenSize = config.HiddenSize;
        _device = CUDA;
    }

    public int HiddenSize => _hiddenSize;
    public int VocabularySize => _vocabSize;

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
        double repetitionPenalty = 0.0)
    {
        return PredictNextTokenGpu(inputTokens, previousToken, stepIndex, disallowToken, penalizedTokens, repetitionPenalty);
    }

    public TrainingLoss TrainExample(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId,
        int? routeLabel)
    {
        return TrainExampleGpu(inputTokens, targetTokens, bosTokenId);
    }

    /// <summary>Forward pass only - accumulates loss, no backward yet.</summary>
    public (torch.Tensor AccumulatedLoss, List<torch.Tensor> ForwardTensors) ForwardPass(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId)
    {
        EnsureGpuInitialized();

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
            
            var targetToken = tensor(new long[] { targetTokens[t] }, dtype: ScalarType.Int64, device: _device);
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
        EnsureGpuInitialized();
        return new ModelSnapshot(
            TensorToMatrix(_embT!),
            TensorToMatrix(_wOutT!),
            TensorToVector(_bOutT!));
    }

    public void Import(ModelSnapshot snapshot)
    {
        _hiddenSize = snapshot.Embeddings.GetLength(1);
        _vocabSize = snapshot.Embeddings.GetLength(0);

        _embT = MatrixToParameter(snapshot.Embeddings);
        _wOutT = MatrixToParameter(snapshot.OutputWeights);
        _bOutT = VectorToParameter(snapshot.OutputBias);
        RecreateOptimizer();
    }

    private int PredictNextTokenGpu(
        IReadOnlyList<int> inputTokens,
        int previousToken,
        int stepIndex,
        int? disallowToken,
        IReadOnlyCollection<int>? penalizedTokens,
        double repetitionPenalty)
    {
        EnsureGpuInitialized();
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

        return ArgMax(scores);
    }

    private TrainingLoss TrainExampleGpu(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId)
    {
        EnsureGpuInitialized();

        double totalLoss = 0.0;
        var prev = bosTokenId;
        
        var exStart = System.Diagnostics.Stopwatch.StartNew();
        
        // Forward pass: accumulate losses WITHOUT disposing tensors
        // This builds the full computation graph from BOS to EOS
        torch.Tensor? accumulatedLoss = null;
        var forwardTensors = new List<torch.Tensor>();

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
                var targetToken = tensor(new long[] { targetTokens[t] }, dtype: ScalarType.Int64, device: _device);
                forwardTensors.Add(targetToken);
                
                var stepLoss = nn.functional.cross_entropy(logits.unsqueeze(0), targetToken);
                forwardTensors.Add(stepLoss);
                
                totalLoss += stepLoss.ToDouble();
                
                // Accumulate into computation graph
                if (accumulatedLoss is null)
                {
                    accumulatedLoss = stepLoss;
                }
                else
                {
                    var newAccum = accumulatedLoss + stepLoss;
                    accumulatedLoss = newAccum;
                }
                
                prev = targetTokens[t];
            }

            exStart.Stop();
            var backStart = System.Diagnostics.Stopwatch.StartNew();
            
            // Backward pass: single step for entire sequence
            if (accumulatedLoss is not null)
            {
                _optimizer!.zero_grad();
                accumulatedLoss.backward();
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

        return new TrainingLoss(totalLoss / Math.Max(1, targetTokens.Count));
    }
    
    /// <summary>Clone parameters to break computation graph chain after epoch.</summary>
    public void CloneParametersToBreakGraph()
    {
        using (no_grad())
        {
            var embClone = _embT!.clone().detach();
            var wClone = _wOutT!.clone().detach();
            var bClone = _bOutT!.clone().detach();
            
            _embT = new TorchSharp.Modules.Parameter(embClone, true);
            _wOutT = new TorchSharp.Modules.Parameter(wClone, true);
            _bOutT = new TorchSharp.Modules.Parameter(bClone, true);
            
            RecreateOptimizer();
        }
    }

    private void EnsureVocabularySizeGpu(int vocabSize)
    {
        if (vocabSize <= _vocabSize)
            return;

        var oldVocab = _vocabSize;
        var hidden = HiddenSize;

        var newEmb = ((rand(new long[] { vocabSize, hidden }, device: _device) * 2.0) - 1.0) * 0.05;
        if (_embT is not null && oldVocab > 0)
            newEmb.slice(0, 0, oldVocab, 1).copy_(_embT);
        _embT = new TorchSharp.Modules.Parameter(newEmb, true);

        var newWOut = ((rand(new long[] { hidden, vocabSize }, device: _device) * 2.0) - 1.0) * 0.05;
        if (_wOutT is not null && oldVocab > 0)
            newWOut.slice(1, 0, oldVocab, 1).copy_(_wOutT);
        _wOutT = new TorchSharp.Modules.Parameter(newWOut, true);

        var newBOut = zeros(new long[] { vocabSize }, dtype: ScalarType.Float32, device: _device);
        if (_bOutT is not null && oldVocab > 0)
            newBOut.slice(0, 0, oldVocab, 1).copy_(_bOutT);
        _bOutT = new TorchSharp.Modules.Parameter(newBOut, true);

        _vocabSize = vocabSize;
        RecreateOptimizer();
    }

    private Tensor MeanEmbeddingTensor(IReadOnlyList<int> inputTokens)
    {
        EnsureGpuInitialized();
        if (inputTokens.Count == 0 || _vocabSize == 0)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _device);

        var idx = inputTokens.Select(i => (long)i).ToArray();
        using var idxTensor = tensor(idx, dtype: ScalarType.Int64, device: _device);
        using var embRows = _embT!.index_select(0, idxTensor);
        
        // Sequential processing with lighter decay: exponential smoothing (alpha=0.3)
        // This preserves more information from all tokens, not just early ones
        Tensor result = zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _device);
        const float alpha = 0.3f;  // Lighter exponential decay
        
        for (var i = 0; i < inputTokens.Count; i++)
        {
            using var emb = embRows.index_select(0, tensor(new long[] { i }, dtype: ScalarType.Int64, device: _device)).squeeze(0);
            using var prev = result;
            using var decayed = prev * (1.0f - alpha);
            using var combined = decayed + (emb * alpha);
            result = combined.tanh();
        }
        
        return result;
    }

    private Tensor GetEmbeddingTensor(int token)
    {
        EnsureGpuInitialized();
        if (token < 0 || token >= _vocabSize)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _device);
        using var idx = tensor(new long[] { token }, dtype: ScalarType.Int64, device: _device);
        return _embT!.index_select(0, idx).squeeze(0);
    }

    private void EnsureGpuInitialized()
    {
        if (!cuda.is_available())
            throw new NotSupportedException("GPU backend requested but CUDA is not available.");

        if (_embT is not null)
            return;

        EnsureVocabularySizeGpu(Math.Max(3, _vocabSize));
    }

    private void EnsureHiddenSizeGpu(int hiddenSize, int oldHidden, int vocab)
    {
        var newEmb = ((rand(new long[] { vocab, hiddenSize }, device: _device) * 2.0) - 1.0) * 0.05;
        if (_embT is not null && oldHidden > 0)
            newEmb.slice(1, 0, oldHidden, 1).copy_(_embT);
        _embT = new TorchSharp.Modules.Parameter(newEmb, true);

        var newWOut = ((rand(new long[] { hiddenSize, vocab }, device: _device) * 2.0) - 1.0) * 0.05;
        if (_wOutT is not null && oldHidden > 0)
            newWOut.slice(0, 0, oldHidden, 1).copy_(_wOutT);
        _wOutT = new TorchSharp.Modules.Parameter(newWOut, true);

        _hiddenSize = hiddenSize;
        RecreateOptimizer();
    }

    private void RecreateOptimizer()
    {
        _optimizer?.Dispose();
        _optimizer = torch.optim.SGD(
            [_embT!, _wOutT!, _bOutT!],
            _config.LearningRate);
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
            tensor(values, dtype: ScalarType.Float32, device: _device).reshape(rows, cols),
            true);
    }

    private TorchSharp.Modules.Parameter VectorToParameter(double[] vector)
    {
        var values = vector.Select(x => (float)x).ToArray();
        return new TorchSharp.Modules.Parameter(tensor(values, dtype: ScalarType.Float32, device: _device), true);
    }
}

public sealed record TrainingLoss(double TokenLoss)
{
    public double Total => TokenLoss;
}

public sealed record ModelSnapshot(
    double[,] Embeddings,
    double[,] OutputWeights,
    double[] OutputBias);
