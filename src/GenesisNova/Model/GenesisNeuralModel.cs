using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public class GenesisNeuralModel
{
    private GenesisNovaConfig _config;
    private readonly Device _trainingDevice;  // CUDA when available, CPU fallback
    private readonly Device _inferenceDevice; // GPU for speed if available
    private int _hiddenSize;
    private int _vocabSize;

    private TorchSharp.Modules.Parameter? _embT;
    private TorchSharp.Modules.Parameter? _wOutT;
    private TorchSharp.Modules.Parameter? _bOutT;
    private torch.optim.Optimizer? _optimizer;

    // SHARED GRU ENCODER (the core of the controller).
    // A single learned GRU recurrence is used for BOTH encoding the input tokens and decoding the
    // target tokens, with SHARED weights. Input size == hidden size == embedding dim, so every gate
    // weight is a square [hidden, hidden] block. We store the three gate blocks stacked as [3h, h]
    // (reset, update, new — matching PyTorch's GRU layout) and apply them via x.matmul(W.t()):
    //   r = sigmoid( x·W_ir^T + b_ir + h·W_hr^T + b_hr )
    //   z = sigmoid( x·W_iz^T + b_iz + h·W_hz^T + b_hz )
    //   n = tanh( x·W_in^T + b_in + r * (h·W_hn^T + b_hn) )
    //   h' = (1 - z) * n + z * h
    // Hand-rolled with raw Parameters (rather than nn.GRUCell) so the weights drop straight into the
    // existing manual optimizer (RecreateOptimizer), Export/Import (TensorToMatrix/Vector helpers),
    // resize (EnsureHiddenSizeGpu) and dispose (DisposeParameters) discipline with no foreign module
    // lifecycle. Lazily initialized like the route head (EnsureGruInitialized); persisted as OPTIONAL
    // snapshot fields with the same graceful-degradation (HasUsableGru) so old checkpoints reinit a
    // fresh (untrained) GRU rather than throwing.
    private TorchSharp.Modules.Parameter? _gruWih; // [3h, h] input-to-hidden gate weights
    private TorchSharp.Modules.Parameter? _gruWhh; // [3h, h] hidden-to-hidden gate weights
    private TorchSharp.Modules.Parameter? _gruBih; // [3h]    input-to-hidden gate biases
    private TorchSharp.Modules.Parameter? _gruBhh; // [3h]    hidden-to-hidden gate biases

    // Route decision space (widened 2 -> 3 for the introspective controller):
    //   0 = neural-only, 1 = platonic-direct, 2 = platonic-assisted reasoning.
    // Old checkpoints carry a [hidden, 2] route head; those are treated as incompatible on
    // Import (see HasUsableRouteHead) and reinitialized, so untrained 3-way heads degrade to
    // neural-only rather than throwing.
    public const int RouteCount = 3;
    private const int NumRoutes = RouteCount;
    private const double RouteLossWeight = 0.25;
    private readonly object _routeClassBalanceLock = new();
    private readonly long[] _routeClassCounts = { 1L, 1L, 1L };
    private TorchSharp.Modules.Parameter? _routeWT;
    private TorchSharp.Modules.Parameter? _routeB;

    // Learned edit-head: predicts HOW STRONGLY the platonic space should be edited for a given
    // input context. Shape mirrors the route head but emits a single scalar: _editWT is [hidden, 1]
    // and _editB is [1]; PredictEditMagnitude pools the input embeddings, applies this linear layer
    // and a sigmoid to yield a bounded magnitude in [0,1]. Because the platonic space is a plain
    // double[] store (not a torch tensor) we cannot backprop through a space edit, so the head is
    // trained REINFORCE-style by ReinforceEditHead rather than by the autograd token/route losses.
    // Lazily initialized like the route head; persisted in Export/Import with the same
    // graceful-degradation (HasUsableEditHead) so old checkpoints simply reinitialize it.
    private const double EditHeadLearningRate = 0.01;     // small, bounded manual REINFORCE step
    private const double EditHeadRewardClamp = 1.0;       // bound the reward magnitude for stability
    private TorchSharp.Modules.Parameter? _editWT;
    private TorchSharp.Modules.Parameter? _editB;

    public GenesisNeuralModel(GenesisNovaConfig config)
    {
        _config = config;
        _hiddenSize = config.HiddenSize;

        // Training uses CUDA when the backend allows it; otherwise fall back to CPU.
        _trainingDevice = SelectTrainingDevice(config);

        // Inference on GPU: prefer A3000 (discrete), fallback to any CUDA device
        _inferenceDevice = SelectInferenceDevice(config);
    }

    private static Device SelectTrainingDevice(GenesisNovaConfig config)
    {
        if (config.Backend == ComputeBackend.Cpu || !torch.cuda_is_available())
        {
            Console.WriteLine("[GPU] CUDA unavailable for training; using CPU fallback");
            return CPU;
        }

        var deviceCount = torch.cuda.device_count();
        if (deviceCount == 0)
        {
            Console.WriteLine("[GPU] No CUDA devices detected for training; using CPU fallback");
            return CPU;
        }

        Console.WriteLine($"[GPU] CUDA devices available for training: {deviceCount}");
        Console.WriteLine("[GPU] Using device 0 for training");
        return new Device(DeviceType.CUDA, 0);
    }

    private static Device SelectInferenceDevice(GenesisNovaConfig config)
    {
        if (config.Backend == ComputeBackend.Cpu || !torch.cuda_is_available())
            return CPU;

        var deviceCount = torch.cuda.device_count();
        if (deviceCount == 0)
            return CPU;

        // Prefer device 0 (likely discrete A3000 if it's the primary GPU)
        Console.WriteLine($"[GPU] CUDA devices available: {deviceCount}");
        Console.WriteLine($"[GPU] Using device 0 for inference (A3000 preferred)");
        
        return new Device(DeviceType.CUDA, 0);
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
        double lossScale = 1.0,
        int? routeLabel = null)
    {
        return TrainExampleGpu(inputTokens, targetTokens, bosTokenId, lossScale, routeLabel);
    }

    public ModelSnapshot Export()
    {
        EnsureModelInitialized();
        return new ModelSnapshot(
            TensorToMatrix(_embT!),
            TensorToMatrix(_wOutT!),
            TensorToVector(_bOutT!),
            _routeWT is not null ? TensorToMatrix(_routeWT) : null,
            _routeB is not null ? TensorToVector(_routeB) : null,
            _editWT is not null ? TensorToMatrix(_editWT) : null,
            _editB is not null ? TensorToVector(_editB) : null,
            // SHARED GRU gate weights/biases (null until lazily initialized).
            _gruWih is not null ? TensorToMatrix(_gruWih) : null,
            _gruWhh is not null ? TensorToMatrix(_gruWhh) : null,
            _gruBih is not null ? TensorToVector(_gruBih) : null,
            _gruBhh is not null ? TensorToVector(_gruBhh) : null);
    }

    public void Import(ModelSnapshot snapshot)
    {
        _hiddenSize = snapshot.Embeddings.GetLength(1);
        _vocabSize = snapshot.Embeddings.GetLength(0);

        DisposeParameters();
        _embT = MatrixToParameter(snapshot.Embeddings);
        _wOutT = MatrixToParameter(snapshot.OutputWeights);
        _bOutT = VectorToParameter(snapshot.OutputBias);
        if (HasUsableRouteHead(snapshot.RouteWeights, snapshot.RouteBias, _hiddenSize))
        {
            _routeWT = MatrixToParameter(snapshot.RouteWeights!);
            _routeB = VectorToParameter(snapshot.RouteBias!);
        }
        if (HasUsableEditHead(snapshot.EditWeights, snapshot.EditBias, _hiddenSize))
        {
            _editWT = MatrixToParameter(snapshot.EditWeights!);
            _editB = VectorToParameter(snapshot.EditBias!);
        }
        // SHARED GRU: load only when all four tensors are present AND shape-compatible with the
        // current hidden size (HasUsableGru). Otherwise leave the GRU null — DisposeParameters above
        // already nulled it — so the next forward lazily reinitializes a fresh untrained GRU and old
        // checkpoints degrade gracefully (just like the route/edit heads) instead of throwing.
        if (HasUsableGru(snapshot.GruWih, snapshot.GruWhh, snapshot.GruBih, snapshot.GruBhh, _hiddenSize))
        {
            _gruWih = MatrixToParameter(snapshot.GruWih!);
            _gruWhh = MatrixToParameter(snapshot.GruWhh!);
            _gruBih = VectorToParameter(snapshot.GruBih!);
            _gruBhh = VectorToParameter(snapshot.GruBhh!);
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
        EnsureGruInitialized();
        using var noGrad = no_grad();

        // Single-step decode through the SHARED GRU (mirrors the stateless contract of the old
        // single-step forward: input context + previous token -> next-token logits, now learned).
        // 1) Encode the input tokens to hInput on the inference device.
        // 2) One decoder GRU step from hInput feeding the previous token's embedding.
        // stepIndex is retained for the public contract but no longer drives a positional decay — the
        // GRU recurrence subsumes positional weighting. All scratch tensors are disposed in finally.
        var scratch = new List<Tensor>();
        float[] scores;
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);
            var prevEmb = GetEmbeddingTensor(previousToken, _inferenceDevice);
            scratch.Add(prevEmb);
            var hDec = GruStep(prevEmb, hInput, scratch, _inferenceDevice);

            // Move output weights to inference device if needed.
            var wOut = _wOutT!.to(_inferenceDevice); scratch.Add(wOut);
            var bOut = _bOutT!.to(_inferenceDevice); scratch.Add(bOut);
            var logits = hDec.matmul(wOut) + bOut; scratch.Add(logits);
            using var logitsCpu = logits.cpu();
            scores = logitsCpu.data<float>().ToArray();
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
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
        double lossScale,
        int? routeLabel)
    {
        EnsureModelInitialized();
        EnsureGruInitialized();
        if (routeLabel is >= 0 and < NumRoutes)
            EnsureRouteHeadInitialized();

        double totalLoss = 0.0;
        var prev = bosTokenId;

        var exStart = System.Diagnostics.Stopwatch.StartNew();

        // Forward pass: accumulate losses WITHOUT disposing tensors
        // This builds the full computation graph from BOS to EOS
        torch.Tensor? accumulatedLoss = null;
        var forwardTensors = new List<torch.Tensor>();
        var stepLosses = new List<torch.Tensor>();
        double routeLossValue = 0.0;

        try
        {
            // SHARED ENCODER: run the GRU over the INPUT tokens to produce hInput, the single learned
            // representation that both the token decoder (as its initial hidden) and the route head
            // read. All encoder intermediates are tracked in forwardTensors for post-backward disposal.
            var hInput = EncodeInput(inputTokens, forwardTensors, _trainingDevice);

            // DECODER: start the recurrence from hInput and step the SAME GRU once per target token,
            // feeding the previous (teacher-forced) token's embedding. Per-step hidden h_t feeds the
            // token head (_wOutT/_bOutT). This replaces the old tanh(inputVec + scaled-prevEmb) pool.
            var hDec = hInput;
            for (var t = 0; t < targetTokens.Count; t++)
            {
                var prevEmb = GetEmbeddingTensor(prev);
                forwardTensors.Add(prevEmb);

                // GRU step: h_t = GRU(embedding(prev), h_{t-1}). Intermediates tracked in forwardTensors.
                hDec = GruStep(prevEmb, hDec, forwardTensors, _trainingDevice);

                // Logits from the per-step decoder hidden.
                var logits = hDec.matmul(_wOutT!) + _bOutT!;
                forwardTensors.Add(logits);

                // Loss for this token
                var targetToken = tensor(new long[] { targetTokens[t] }, dtype: ScalarType.Int64, device: _trainingDevice);
                var logitsBatch = logits.unsqueeze(0);
                forwardTensors.Add(logitsBatch);
                var stepLoss = nn.functional.cross_entropy(logitsBatch, targetToken);
                // Clean up targetToken immediately - it's only used for cross_entropy call
                targetToken.Dispose();

                stepLosses.Add(stepLoss);
                forwardTensors.Add(stepLoss);

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

            if (routeLabel is >= 0 and < NumRoutes)
            {
                // Route head reads the SHARED GRU representation hInput (already a bounded learned
                // vector — no extra hand-pool/tanh). hInput is a live graph node so the route CE loss
                // backprops into the shared encoder, jointly shaping it for routing and decoding.
                var routeLogits = hInput.matmul(_routeWT!) + _routeB!;
                forwardTensors.Add(routeLogits);
                var routeBatchLogits = routeLogits.unsqueeze(0);
                forwardTensors.Add(routeBatchLogits);
                var routeTarget = tensor(new long[] { routeLabel.Value }, dtype: ScalarType.Int64, device: _trainingDevice);
                var routeLoss = nn.functional.cross_entropy(routeBatchLogits, routeTarget);
                routeTarget.Dispose();
                forwardTensors.Add(routeLoss);
                routeLossValue = routeLoss.ToDouble();

                var classBalance = ObserveRouteClassWeight(routeLabel.Value);
                var weightedRouteLoss = routeLoss * (RouteLossWeight * classBalance);
                forwardTensors.Add(weightedRouteLoss);
                if (accumulatedLoss is null)
                {
                    accumulatedLoss = weightedRouteLoss;
                }
                else
                {
                    accumulatedLoss = accumulatedLoss + weightedRouteLoss;
                    forwardTensors.Add(accumulatedLoss);
                }
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

        return new TrainingLoss(totalLoss / Math.Max(1, targetTokens.Count), RouteLoss: routeLossValue);
    }

    private double ObserveRouteClassWeight(int routeLabel)
    {
        lock (_routeClassBalanceLock)
        {
            _routeClassCounts[routeLabel]++;
            var total = _routeClassCounts.Sum();
            var classCount = Math.Max(1L, _routeClassCounts[routeLabel]);
            var weight = total / (double)(NumRoutes * classCount);
            return Math.Clamp(weight, 0.25, 4.0);
        }
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

        // L2 penalty on shared GRU gate weights (biases left unregularized, as for _bOut elsewhere
        // we do penalize, but GRU biases are kept free to let gates saturate). Weight decay here
        // keeps the recurrence from bloating just like the other learned matrices.
        if (_gruWih is not null)
            penalty = penalty + (_gruWih * _gruWih).sum();
        if (_gruWhh is not null)
            penalty = penalty + (_gruWhh * _gruWhh).sum();

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
            // CRITICAL: Dispose optimizer FIRST before replacing parameters.
            // Old optimizer holds references to old parameters, which we're about to dispose.
            // If we don't dispose the optimizer first, it may try to access disposed tensors during next step.
            _optimizer?.Dispose();
            _optimizer = null;
            
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

            if (_editWT is not null)
            {
                var ewClone = _editWT.clone().detach().to(_trainingDevice);
                var ebClone = _editB!.clone().detach().to(_trainingDevice);
                var oldEW = _editWT;
                var oldEB = _editB;
                _editWT = new TorchSharp.Modules.Parameter(ewClone, true);
                _editB = new TorchSharp.Modules.Parameter(ebClone, true);
                try { oldEW.Dispose(); } catch { }
                try { oldEB.Dispose(); } catch { }
            }

            // SHARED GRU gate weights/biases — clone+detach to break the graph chain, mirroring the
            // route/edit heads. These ARE optimizer params, so RecreateOptimizer below picks up the
            // fresh leaves.
            if (_gruWih is not null)
            {
                var wihClone = _gruWih.clone().detach().to(_trainingDevice);
                var whhClone = _gruWhh!.clone().detach().to(_trainingDevice);
                var bihClone = _gruBih!.clone().detach().to(_trainingDevice);
                var bhhClone = _gruBhh!.clone().detach().to(_trainingDevice);
                var oldWih = _gruWih;
                var oldWhh = _gruWhh;
                var oldBih = _gruBih;
                var oldBhh = _gruBhh;
                _gruWih = new TorchSharp.Modules.Parameter(wihClone, true);
                _gruWhh = new TorchSharp.Modules.Parameter(whhClone, true);
                _gruBih = new TorchSharp.Modules.Parameter(bihClone, true);
                _gruBhh = new TorchSharp.Modules.Parameter(bhhClone, true);
                try { oldWih.Dispose(); } catch { }
                try { oldWhh.Dispose(); } catch { }
                try { oldBih.Dispose(); } catch { }
                try { oldBhh.Dispose(); } catch { }
            }

            // NOW create a fresh optimizer with the new parameters
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

    private Tensor MeanEmbeddingTensor(IReadOnlyList<int> inputTokens, Device? device = null)
    {
        device ??= _trainingDevice;
        EnsureModelInitialized();
        if (inputTokens.Count == 0 || _vocabSize == 0)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: device);

        var maxToken = inputTokens.Max();
        if (maxToken >= _vocabSize)
            EnsureVocabularySizeGpu(maxToken + 1);

        var idx = inputTokens
            .Where(i => i >= 0 && i < _vocabSize)
            .Select(i => (long)i)
            .ToArray();

        if (idx.Length == 0)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: device);

        using var idxTensor = tensor(idx, dtype: ScalarType.Int64, device: _trainingDevice);
        using var embRows = _embT!.index_select(0, idxTensor);
         
        // Sequential processing with lighter decay: exponential smoothing (alpha=0.3)
        // This preserves more information from all tokens, not just early ones
        Tensor result = zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice);
        const float alpha = 0.3f;  // Lighter exponential decay
         
        for (var i = 0; i < idx.Length; i++)
        {
            using var emb = embRows.index_select(0, tensor(new long[] { i }, dtype: ScalarType.Int64, device: _trainingDevice)).squeeze(0);
            using var prev = result;
            using var decayed = prev * (1.0f - alpha);
            using var combined = decayed + (emb * alpha);
            result = combined.tanh();
        }
         
        if (device != _trainingDevice)
            result = result.to(device);
         
        return result;
    }

    private Tensor GetEmbeddingTensor(int token, Device? device = null)
    {
        device ??= _trainingDevice;
        EnsureModelInitialized();
        if (token < 0 || token >= _vocabSize)
            return zeros(new long[] { HiddenSize }, dtype: ScalarType.Float32, device: device);
        using var idx = tensor(new long[] { token }, dtype: ScalarType.Int64, device: _trainingDevice);
        var emb = _embT!.index_select(0, idx).squeeze(0);
         
        if (device != _trainingDevice)
            emb = emb.to(device);
         
        return emb;
    }

    private void EnsureModelInitialized()
    {
        if (_embT is not null)
            return;

        EnsureVocabularySizeGpu(Math.Max(3, _vocabSize));
    }

    private void EnsureGruInitialized()
    {
        if (_gruWih is not null)
            return;

        var h = _hiddenSize;
        // Small-uniform init in [-k, k] with k = 1/sqrt(h), the conventional GRU init scale. Stacked
        // [3h, h] for the three gates (reset, update, new); biases zeroed.
        var k = 1.0 / Math.Sqrt(Math.Max(1, h));
        _gruWih = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { 3 * h, h }, device: _trainingDevice) * 2.0) - 1.0) * k, true);
        _gruWhh = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { 3 * h, h }, device: _trainingDevice) * 2.0) - 1.0) * k, true);
        _gruBih = new TorchSharp.Modules.Parameter(
            zeros(new long[] { 3 * h }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        _gruBhh = new TorchSharp.Modules.Parameter(
            zeros(new long[] { 3 * h }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        RecreateOptimizer();
    }

    /// <summary>
    /// One GRU recurrence step on a single (un-batched) hidden/input vector. <paramref name="x"/> and
    /// <paramref name="h"/> are both [hidden]; returns the new hidden [hidden]. Every intermediate
    /// tensor created here is appended to <paramref name="scratch"/> so the caller can dispose them
    /// after backward() — NOTHING is disposed inside this method (the returned tensor and all gate
    /// intermediates remain part of the live autograd graph until backward runs). The GRU weights
    /// (_gruW*/_gruB*) are leaf Parameters and are NOT added to scratch.
    /// Uses W stacked as [3h, h]; gate slices are rows [0,h) reset, [h,2h) update, [2h,3h) new.
    /// </summary>
    private Tensor GruStep(Tensor x, Tensor h, List<Tensor> scratch, Device device)
    {
        var hsz = _hiddenSize;

        // Move weights to the requested device when it differs from the training device (inference).
        // These device-moved copies are tracked so they are disposed with the rest of the scratch.
        Tensor wih = _gruWih!, whh = _gruWhh!, bih = _gruBih!, bhh = _gruBhh!;
        if (device != _trainingDevice)
        {
            wih = _gruWih!.to(device); scratch.Add(wih);
            whh = _gruWhh!.to(device); scratch.Add(whh);
            bih = _gruBih!.to(device); scratch.Add(bih);
            bhh = _gruBhh!.to(device); scratch.Add(bhh);
        }

        // Full pre-activations: x·W_ih^T + b_ih  (gi: [3h])  and  h·W_hh^T + b_hh  (gh: [3h]).
        var wihT = wih.t(); scratch.Add(wihT);
        var xgi = x.matmul(wihT); scratch.Add(xgi);
        var gi = xgi + bih; scratch.Add(gi);

        var whhT = whh.t(); scratch.Add(whhT);
        var hgh = h.matmul(whhT); scratch.Add(hgh);
        var gh = hgh + bhh; scratch.Add(gh);

        // Slice the three gate blocks out of the [3h] pre-activations.
        var iR = gi.narrow(0, 0, hsz); scratch.Add(iR);
        var iZ = gi.narrow(0, hsz, hsz); scratch.Add(iZ);
        var iN = gi.narrow(0, 2 * hsz, hsz); scratch.Add(iN);
        var hR = gh.narrow(0, 0, hsz); scratch.Add(hR);
        var hZ = gh.narrow(0, hsz, hsz); scratch.Add(hZ);
        var hN = gh.narrow(0, 2 * hsz, hsz); scratch.Add(hN);

        // r = sigmoid(iR + hR)
        var rPre = iR + hR; scratch.Add(rPre);
        var r = torch.sigmoid(rPre); scratch.Add(r);

        // z = sigmoid(iZ + hZ)
        var zPre = iZ + hZ; scratch.Add(zPre);
        var z = torch.sigmoid(zPre); scratch.Add(z);

        // n = tanh(iN + r * hN)
        var rhN = r * hN; scratch.Add(rhN);
        var nPre = iN + rhN; scratch.Add(nPre);
        var n = nPre.tanh(); scratch.Add(n);

        // h' = (1 - z) * n + z * h
        var oneMinusZ = 1.0 - z; scratch.Add(oneMinusZ);
        var newPart = oneMinusZ * n; scratch.Add(newPart);
        var keepPart = z * h; scratch.Add(keepPart);
        var hNew = newPart + keepPart; scratch.Add(hNew);

        return hNew;
    }

    /// <summary>
    /// Encode the input tokens with the shared GRU: h = zeros; for each input token,
    /// h = GruStep(embedding(token), h). The final hidden is the SHARED representation (hInput) that
    /// the router and write-gate read. Every intermediate (embeddings + per-step GRU scratch) is
    /// appended to <paramref name="scratch"/> for disposal after backward(); the returned tensor is
    /// the live graph leaf into hInput and is NOT disposed here.
    /// </summary>
    private Tensor EncodeInput(IReadOnlyList<int> inputTokens, List<Tensor> scratch, Device device)
    {
        EnsureGruInitialized();
        Tensor h = zeros(new long[] { _hiddenSize }, dtype: ScalarType.Float32, device: device);
        scratch.Add(h);

        foreach (var tok in inputTokens)
        {
            var emb = GetEmbeddingTensor(tok, device);
            scratch.Add(emb);
            h = GruStep(emb, h, scratch, device);
        }

        return h;
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

    private void EnsureEditHeadInitialized()
    {
        if (_editWT is not null)
            return;

        // Mirror EnsureRouteHeadInitialized, but emit a single scalar ([hidden, 1] + [1]).
        // Deliberately NOT added to _optimizer / RecreateOptimizer: the platonic space is a plain
        // double[] store, so there is no autograd path from a space edit back to these weights. The
        // head is trained only by ReinforceEditHead's manual REINFORCE step.
        _editWT = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { _hiddenSize, 1 }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _editB = new TorchSharp.Modules.Parameter(
            zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: _trainingDevice), true);
    }

    /// <summary>
    /// Predict how strongly the platonic space should be edited for this input context.
    /// Encodes the input tokens through the SHARED GRU to hInput — the same learned representation the
    /// router reads — passes it (detached) through the learned edit-head linear layer, and squashes
    /// with a sigmoid to a bounded magnitude in [0,1]. Deterministic given weights. Returns 0.5
    /// (neutral) when there is no signal: empty input or an uninitialized model/head.
    /// hInput is detached implicitly here (the whole call runs under no_grad), so the edit-head reads a
    /// fixed snapshot of the shared encoder and never backprops into it via this path.
    /// </summary>
    public double PredictEditMagnitude(IReadOnlyList<int> inputTokens)
    {
        if (inputTokens.Count == 0)
            return 0.5;

        EnsureModelInitialized();
        EnsureGruInitialized();
        EnsureEditHeadInitialized();
        if (_editWT is null || _editB is null)
            return 0.5;

        using var noGrad = no_grad();
        // Encode through the SHARED GRU onto the inference device (route weights also move there).
        // All encoder scratch is disposed in finally; only the final [hidden] hInput is consumed.
        var scratch = new List<Tensor>();
        float[] value;
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);
            Tensor editW = _editWT!, editB = _editB!;
            if (_inferenceDevice != _trainingDevice)
            {
                editW = _editWT!.to(_inferenceDevice); scratch.Add(editW);
                editB = _editB!.to(_inferenceDevice); scratch.Add(editB);
            }
            // hInput: [hidden]; editW: [hidden, 1] -> matmul yields [1]; add bias [1]; sigmoid -> [0,1].
            var raw = hInput.matmul(editW) + editB; scratch.Add(raw);
            var mag = torch.sigmoid(raw); scratch.Add(mag);
            using var magCpu = mag.cpu();
            value = magCpu.data<float>().ToArray();
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
        if (value.Length == 0)
            return 0.5;
        return Math.Clamp((double)value[0], 0.0, 1.0);
    }

    /// <summary>
    /// REINFORCE-style policy update for the edit-head. Treats <paramref name="appliedMagnitude"/>
    /// as the action taken in this context and <paramref name="reward"/> as its return.
    ///
    /// The head's sigmoid output is treated as the mean of a (fixed-variance) continuous policy, so
    /// the REINFORCE surrogate to MINIMIZE is  L = reward * (appliedMagnitude - predicted)^2:
    ///   reward &gt; 0 -&gt; minimizing pulls 'predicted' CLOSER to appliedMagnitude for this context;
    ///   reward &lt; 0 -&gt; the negative coefficient turns it into pushing 'predicted' AWAY.
    /// A single manual SGD step (small, bounded learning rate) is applied to the edit-head params
    /// only, leaving the token/route weights and the shared optimizer untouched so the rest of the
    /// model is not destabilized.
    /// </summary>
    public void ReinforceEditHead(IReadOnlyList<int> inputTokens, double appliedMagnitude, double reward)
    {
        if (inputTokens.Count == 0)
            return;

        EnsureModelInitialized();
        EnsureGruInitialized();
        EnsureEditHeadInitialized();
        if (_editWT is null || _editB is null)
            return;

        if (double.IsNaN(reward) || double.IsInfinity(reward) || reward == 0.0)
            return;

        // Bound the action and the reward so a single example cannot blow up the head.
        var action = (float)Math.Clamp(appliedMagnitude, 0.0, 1.0);
        var boundedReward = (float)Math.Clamp(reward, -EditHeadRewardClamp, EditHeadRewardClamp);

        // Encode the context through the SHARED GRU to hInput, fully DETACHED. We run the encode under
        // no_grad and clone the result into a standalone leaf so the subsequent grad-enabled forward
        // backprops into ONLY _editWT/_editB — the REINFORCE reward must never nudge the shared GRU /
        // embedding table (and cannot leave stale grads on them for the next TrainExampleGpu pass).
        Tensor meanEmb;
        var encScratch = new List<Tensor>();
        try
        {
            using (no_grad())
            {
                var hInput = EncodeInput(inputTokens, encScratch, _trainingDevice);
                meanEmb = hInput.detach().clone(); // standalone detached leaf on the training device
            }
        }
        finally
        {
            foreach (var t in encScratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }

        // Build a fresh forward pass WITH grad so we get d/dparams of the surrogate, then take a
        // manual gradient-descent step on the edit-head parameters only (no shared optimizer).
        using var meanEmbHolder = meanEmb;
        using var raw = meanEmb.matmul(_editWT!) + _editB!;
        using var predicted = torch.sigmoid(raw);              // [1], requires grad
        using var target = tensor(new float[] { action }, dtype: ScalarType.Float32, device: _trainingDevice);
        using var diff = predicted - target;
        using var sq = diff * diff;
        // Surrogate loss; minimizing moves 'predicted' toward action when reward>0, away when <0.
        using var surrogate = sq * boundedReward;
        using var loss = surrogate.sum();

        if (_editWT.grad is not null) _editWT.grad.zero_();
        if (_editB.grad is not null) _editB.grad.zero_();
        loss.backward();

        using (no_grad())
        {
            if (_editWT.grad is not null)
            {
                using var stepW = _editWT.grad * EditHeadLearningRate;
                _editWT.sub_(stepW);
            }
            if (_editB.grad is not null)
            {
                using var stepB = _editB.grad * EditHeadLearningRate;
                _editB.sub_(stepB);
            }
            // Clear grads so this manual step never leaks into the shared optimizer's backward().
            if (_editWT.grad is not null) _editWT.grad.zero_();
            if (_editB.grad is not null) _editB.grad.zero_();
        }
    }

    public (int RouteId, double Confidence) PredictRoute(IReadOnlyList<int> inputTokens)
    {
        EnsureModelInitialized();
        EnsureGruInitialized();
        EnsureRouteHeadInitialized();
        using var noGrad = no_grad();

        // Route head reads the SHARED GRU representation hInput (replaces the old meanEmb.tanh() pool).
        var scratch = new List<Tensor>();
        float[] scores;
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);

            // Route weights live on the training device; move to inference device when they differ.
            Tensor routeW = _routeWT!, routeB = _routeB!;
            if (_inferenceDevice != _trainingDevice)
            {
                routeW = _routeWT!.to(_inferenceDevice); scratch.Add(routeW);
                routeB = _routeB!.to(_inferenceDevice); scratch.Add(routeB);
            }
            var logits = hInput.matmul(routeW) + routeB; scratch.Add(logits);
            var probs = nn.functional.softmax(logits, 0); scratch.Add(probs);
            using var probsCpu = probs.cpu();
            scores = probsCpu.data<float>().ToArray();
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
        if (scores.Length == 0)
            return (0, 0.0);

        // N-way argmax (N == route head width; tolerates an old [hidden,2] head that slipped
        // through Import without throwing — it just can't select route 2).
        //
        // Tie-break is deliberately UNBIASED. The OLD code argmaxed with '<=', which on a tie kept
        // the lowest index (route 0 = neural), so an untrained near-uniform head silently collapsed
        // to neural. Below we first find the true maximum probability, then break exact/near ties
        // without favouring index 0 (see the median pick), and surface the winner's TRUE softmax
        // probability as the confidence — near 1/N on an uncertain head — so callers see honest low
        // confidence instead of a forced neural default.
        var best = scores[0];
        for (var i = 1; i < scores.Length; i++)
        {
            if (scores[i] > best)
                best = scores[i];
        }

        // Collect every index within a tiny epsilon of the maximum. With a decisive head this is a
        // single index (the genuine argmax). On an exact/near tie it holds several indices; we then
        // break the tie deterministically but WITHOUT favouring index 0 / neural — we pick the
        // median of the tied indices, so route 0 wins a tie only when it is actually the central
        // (or sole) tied route, never merely because it has the smallest index.
        const float TieEpsilon = 1e-6f;
        var tied = new List<int>();
        for (var i = 0; i < scores.Length; i++)
        {
            if (best - scores[i] <= TieEpsilon)
                tied.Add(i);
        }
        var routeId = tied[tied.Count / 2];

        // Confidence is the SELECTED route's softmax probability (scores[routeId]), not the bare max —
        // on a tie the chosen routeId is the median tied index, so reporting `best` would attach a
        // different route's probability. On a tie/near-uniform head this sits near 1/N (honest low
        // confidence); with a decisive head it equals the true maximum.
        return (routeId, (double)scores[routeId]);
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

        // Edit head shape is [oldHidden, 1] — incompatible after resize; reinitialize.
        if (_editWT is not null)
        {
            var oldEW = _editWT;
            var oldEB = _editB;
            _editWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { hiddenSize, 1 }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _editB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            try { oldEW.Dispose(); } catch { }
            try { oldEB?.Dispose(); } catch { }
        }

        // SHARED GRU gate weights are [3*oldHidden, oldHidden] — incompatible after a hidden resize;
        // reinitialize a fresh untrained GRU at the new hidden size (same init as EnsureGruInitialized).
        // We do NOT attempt to copy old weights: the gate blocks are stacked [3h,h] so a grow would need
        // per-block slice copies in both dims, and a fresh GRU degrades gracefully like the heads above.
        if (_gruWih is not null)
        {
            var oldWih = _gruWih;
            var oldWhh = _gruWhh;
            var oldBih = _gruBih;
            var oldBhh = _gruBhh;

            var k = 1.0 / Math.Sqrt(Math.Max(1, hiddenSize));
            _gruWih = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { 3 * hiddenSize, hiddenSize }, device: _trainingDevice) * 2.0) - 1.0) * k, true);
            _gruWhh = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { 3 * hiddenSize, hiddenSize }, device: _trainingDevice) * 2.0) - 1.0) * k, true);
            _gruBih = new TorchSharp.Modules.Parameter(
                zeros(new long[] { 3 * hiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            _gruBhh = new TorchSharp.Modules.Parameter(
                zeros(new long[] { 3 * hiddenSize }, dtype: ScalarType.Float32, device: _trainingDevice), true);

            try { oldWih.Dispose(); } catch { }
            try { oldWhh?.Dispose(); } catch { }
            try { oldBih?.Dispose(); } catch { }
            try { oldBhh?.Dispose(); } catch { }
        }

        RecreateOptimizer();
    }

    private void RecreateOptimizer()
    {
        _optimizer?.Dispose();
        var parameters = new List<TorchSharp.Modules.Parameter> { _embT!, _wOutT!, _bOutT! };
        if (_routeWT is not null) parameters.Add(_routeWT);
        if (_routeB is not null) parameters.Add(_routeB!);
        // SHARED GRU gate weights/biases are trained by the autograd token+route losses, so they MUST
        // be registered with the shared optimizer. They are lazily created (EnsureGruInitialized) and
        // each create/resize calls RecreateOptimizer, so by the time a backward()/step() runs they are
        // present. The edit-head (_editWT/_editB) is deliberately excluded — it is REINFORCE-trained.
        if (_gruWih is not null) parameters.Add(_gruWih);
        if (_gruWhh is not null) parameters.Add(_gruWhh!);
        if (_gruBih is not null) parameters.Add(_gruBih!);
        if (_gruBhh is not null) parameters.Add(_gruBhh!);
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
        try { _editWT?.Dispose(); } catch { }
        try { _editB?.Dispose(); } catch { }
        _editWT = null;
        _editB = null;
        // SHARED GRU gate weights/biases.
        try { _gruWih?.Dispose(); } catch { }
        try { _gruWhh?.Dispose(); } catch { }
        try { _gruBih?.Dispose(); } catch { }
        try { _gruBhh?.Dispose(); } catch { }
        _gruWih = null;
        _gruWhh = null;
        _gruBih = null;
        _gruBhh = null;
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

    private static bool HasUsableRouteHead(double[,]? routeWeights, double[]? routeBias, int hiddenSize)
    {
        if (routeWeights is null || routeBias is null)
            return false;

        var rows = routeWeights.GetLength(0);
        var cols = routeWeights.GetLength(1);
        // Require an exact NumRoutes-wide head. Old checkpoints store a [hidden, 2] head; those
        // are rejected here and reinitialized as a fresh (untrained) NumRoutes-way head, so the
        // controller degrades to neural-only behaviour instead of loading a mismatched-shape head.
        return rows == hiddenSize && rows > 0 && cols == NumRoutes && routeBias.Length == cols;
    }

    private static bool HasUsableEditHead(double[,]? editWeights, double[]? editBias, int hiddenSize)
    {
        if (editWeights is null || editBias is null)
            return false;

        var rows = editWeights.GetLength(0);
        var cols = editWeights.GetLength(1);
        // Mirror HasUsableRouteHead: require an exact [hidden, 1] head with a matching [1] bias.
        // Checkpoints saved before the edit-head existed carry nulls (handled above); any wrong
        // shape (e.g. a different hidden size) is rejected here and the head is reinitialized as a
        // fresh untrained one, so old checkpoints degrade gracefully instead of throwing.
        return rows == hiddenSize && rows > 0 && cols == 1 && editBias.Length == cols;
    }

    private static bool HasUsableGru(
        double[,]? gruWih, double[,]? gruWhh, double[]? gruBih, double[]? gruBhh, int hiddenSize)
    {
        if (gruWih is null || gruWhh is null || gruBih is null || gruBhh is null)
            return false;
        if (hiddenSize <= 0)
            return false;

        // Both gate-weight matrices are stacked [3h, h]; both gate-bias vectors are [3h]. Reject any
        // checkpoint whose hidden size differs (or that was saved before the GRU existed, handled by
        // the null guard above): the next forward then reinitializes a fresh untrained GRU, mirroring
        // HasUsableRouteHead/HasUsableEditHead graceful degradation.
        var expectRows = 3 * hiddenSize;
        return gruWih.GetLength(0) == expectRows && gruWih.GetLength(1) == hiddenSize
            && gruWhh.GetLength(0) == expectRows && gruWhh.GetLength(1) == hiddenSize
            && gruBih.Length == expectRows
            && gruBhh.Length == expectRows;
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
    double[]? RouteBias = null,
    double[,]? EditWeights = null,
    double[]? EditBias = null,
    // SHARED GRU gate weights/biases — OPTIONAL, appended after the existing fields so every existing
    // `new ModelSnapshot(...)` call site still compiles. Stored as [3h, h] matrices / [3h] vectors.
    // null on checkpoints saved before the GRU existed; HasUsableGru rejects shape-mismatched ones.
    double[,]? GruWih = null,
    double[,]? GruWhh = null,
    double[]? GruBih = null,
    double[]? GruBhh = null);
