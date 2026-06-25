using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
    public int PredictNextToken(
        IReadOnlyList<int> inputTokens,
        int previousToken,
        int stepIndex = 0,
        int? disallowToken = null,
        IReadOnlyCollection<int>? penalizedTokens = null,
        double repetitionPenalty = 0.0,
        IReadOnlyDictionary<int, double>? tokenBiases = null,
        int? stopToken = null,
        Tensor? promptState = null)
    {
        return PredictNextTokenGpu(inputTokens, previousToken, stepIndex, disallowToken, penalizedTokens, repetitionPenalty, tokenBiases, stopToken, promptState);
    }

    /// <summary>
    /// Encode the (invariant) prompt ONCE per generation and return the encoder seed (hInput) the
    /// per-step decoder reuses. ENCODE-ONCE optimization: <see cref="EncodeInput"/> reads only
    /// <paramref name="inputTokens"/>, so for a fixed prompt the seed is identical on every decode step;
    /// the only per-step variation is the previous-token embedding fed to <see cref="GruStep"/>. Because
    /// the model's weights are CONSTANT within a single generation (a pure no-grad forward; no training
    /// runs mid-generation; model ops are serialized by the runtime gate), reusing this seed for every
    /// step is mathematically identical to recomputing it each step — turning the decode loop from
    /// O(N·M) into O(N+M).
    ///
    /// LIFETIME: the returned tensor is detached + cloned so it is INDEPENDENT of the temporary encode
    /// scratch (which is disposed before returning). It is owned by the CALLER and must be disposed at
    /// the end of the generation. It MUST be scoped to ONE generation — never cached on a field / static /
    /// across generations (the trainer may mutate weights between generations, which would stale the seed).
    /// </summary>
    public Tensor EncodePromptState(IReadOnlyList<int> inputTokens)
    {
        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();
        var scratch = new List<Tensor>();
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);
            // Detach + clone so the seed survives disposal of the encode scratch and carries no graph.
            return hInput.detach().clone();
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
    }

    public TrainingLoss TrainExample(
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> targetTokens,
        int bosTokenId,
        double lossScale = 1.0,
        int? routeLabel = null,
        GenesisQueryLabel? queryLabel = null,
        int? planLabel = null,
        int[]? roleLabels = null)
    {
        return TrainExampleGpu(inputTokens, targetTokens, bosTokenId, lossScale, routeLabel, queryLabel, planLabel, roleLabels);
    }

    private int PredictNextTokenGpu(
        IReadOnlyList<int> inputTokens,
        int previousToken,
        int stepIndex,
        int? disallowToken,
        IReadOnlyCollection<int>? penalizedTokens,
        double repetitionPenalty,
        IReadOnlyDictionary<int, double>? tokenBiases,
        int? stopToken = null,
        Tensor? promptState = null)
    {
        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();

        // Single-step decode through the SHARED GRU (mirrors the stateless contract of the old
        // single-step forward: input context + previous token -> next-token logits, now learned).
        // 1) Encode the input tokens to hInput on the inference device — OR reuse a caller-supplied
        //    encoder seed (ENCODE-ONCE: the seed is invariant for a fixed prompt; see EncodePromptState).
        // 2) One decoder GRU step from hInput feeding the previous token's embedding.
        // stepIndex is retained for the public contract but no longer drives a positional decay — the
        // GRU recurrence subsumes positional weighting. All scratch tensors are disposed in finally.
        // A caller-supplied promptState is NOT added to scratch — the CALLER owns its lifetime across the
        // whole generation and disposes it after the decode loop.
        var scratch = new List<Tensor>();
        float[] scores;
        try
        {
            var hInput = promptState ?? EncodeInput(inputTokens, scratch, _inferenceDevice);
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

        // LEARNED-STOP SOVEREIGNTY: platonic token biases are evidence for choosing CONTENT — they
        // must refine WHICH token, never WHETHER to stop. The stop token can never receive a bias
        // (it is not a platonic concept), so once the answer is emitted, biased sibling concepts
        // would always outscore the never-biased stop and decode would cascade through the
        // neighbourhood ("fruit grape banana orange") even though the model has LEARNED to stop
        // (empirically: raw after-answer argmax is EOS 4/4 with biases off). So when the UNBIASED
        // decoder selects the stop token, return it without applying biases.
        if (stopToken.HasValue && tokenBiases is not null)
        {
            var unbiasedBest = ArgMax(scores);
            if (unbiasedBest == stopToken.Value)
                return unbiasedBest;
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
        int? routeLabel,
        GenesisQueryLabel? queryLabel = null,
        int? planLabel = null,
        int[]? roleLabels = null)
    {
        EnsureModelInitialized();
        EnsureGruInitialized();
        if (routeLabel is >= 0 and < NumRoutes)
            EnsureRouteHeadInitialized();
        var superviseQuery = queryLabel is { OperationId: >= 0 and < QueryOpCount } ql
            && ql.OperandMask.Length == inputTokens.Count;
        var supervisePlan = planLabel is >= 0 and < PlanKindCount;
        if (superviseQuery || supervisePlan)
            EnsureQueryHeadsInitialized(); // creates the query + plan heads together
        // PER-TOKEN ROLE supervision — the NN structure recogniser. Labels (one role class per input token, or a
        // negative "ignore") come from the self-supervised assert/recall alignment. Needs the per-token states below.
        var superviseRole = false;
        if (roleLabels is not null && roleLabels.Length == inputTokens.Count)
            for (var i = 0; i < roleLabels.Length; i++) if (roleLabels[i] >= 0 && roleLabels[i] < RoleCount) { superviseRole = true; break; }
        if (superviseRole)
            EnsureRoleHeadInitialized();

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
            // When query supervision is present, also collect the per-token hidden states for the
            // operand-selection head.
            var perTokenStates = (superviseQuery || superviseRole) ? new List<torch.Tensor>() : null;
            var hInput = EncodeInput(inputTokens, forwardTensors, _trainingDevice, perTokenStates);

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
                var routeTrunk = ReasoningTrunk(hInput);
                forwardTensors.Add(routeTrunk);
                var routeLogits = routeTrunk.matmul(_routeWT!) + _routeB!;
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

            if (superviseQuery && perTokenStates is { Count: > 0 })
            {
                var label = queryLabel!.Value;

                // OPERATION head: CE on the final shared-GRU hidden — which face op does this input ask
                // for? Backprops into the shared encoder alongside the token/route losses.
                var opTrunk = ReasoningTrunk(hInput);
                forwardTensors.Add(opTrunk);
                var opLogits = opTrunk.matmul(_queryOpWT!) + _queryOpB!;
                forwardTensors.Add(opLogits);
                var opBatch = opLogits.unsqueeze(0);
                forwardTensors.Add(opBatch);
                var opTarget = tensor(new long[] { label.OperationId }, dtype: ScalarType.Int64, device: _trainingDevice);
                var opLoss = nn.functional.cross_entropy(opBatch, opTarget);
                opTarget.Dispose();
                forwardTensors.Add(opLoss);

                // OPERAND head: per-token BCE — is THIS token an operand? Scores each per-token hidden.
                var operandScores = new torch.Tensor[perTokenStates.Count];
                for (var t = 0; t < perTokenStates.Count; t++)
                {
                    var score = perTokenStates[t].matmul(_queryOperandWT!) + _queryOperandB!;
                    forwardTensors.Add(score);
                    operandScores[t] = score;
                }
                var scoresVec = cat(operandScores, 0);
                forwardTensors.Add(scoresVec);
                var maskTarget = tensor(
                    label.OperandMask.Select(m => m ? 1.0f : 0.0f).ToArray(),
                    dtype: ScalarType.Float32, device: _trainingDevice);
                var operandLoss = nn.functional.binary_cross_entropy_with_logits(scoresVec, maskTarget);
                maskTarget.Dispose();
                forwardTensors.Add(operandLoss);

                // Class-balance ONLY the op CE — the imbalanced multiclass that collapses to the majority op.
                // The per-token operand BCE is independent binary decisions, not a collapsing multiclass, so it
                // stays unweighted. Mirrors the route head's inverse-frequency weighting.
                var opClassBalance = ObserveQueryOpClassWeight(label.OperationId);
                var weightedOpLoss = opLoss * opClassBalance;
                forwardTensors.Add(weightedOpLoss);
                var opPlusOperand = weightedOpLoss + operandLoss;
                forwardTensors.Add(opPlusOperand);
                var queryLoss = opPlusOperand * QueryLossWeight;
                forwardTensors.Add(queryLoss);
                if (accumulatedLoss is null)
                {
                    accumulatedLoss = queryLoss;
                }
                else
                {
                    accumulatedLoss = accumulatedLoss + queryLoss;
                    forwardTensors.Add(accumulatedLoss);
                }
            }

            // PER-TOKEN ROLE head: CE per supervised token — what grammatical role does THIS token play? Scores each
            // per-token hidden over {NONE,SUBJECT,VALUE,QUERY} and backprops into the shared encoder, so the GRU learns
            // a representation that recognises structure (the NN-as-recogniser; tokens with a negative label are skipped).
            if (superviseRole && perTokenStates is { Count: > 0 })
            {
                var supLogits = new List<torch.Tensor>();
                var supTargets = new List<long>();
                for (var t = 0; t < perTokenStates.Count && t < roleLabels!.Length; t++)
                {
                    var r = roleLabels[t];
                    if (r < 0 || r >= RoleCount) continue;
                    var logit = (perTokenStates[t].matmul(_roleWT!) + _roleB!).unsqueeze(0); // [1, RoleCount]
                    forwardTensors.Add(logit);
                    supLogits.Add(logit);
                    supTargets.Add(r);
                }
                if (supLogits.Count > 0)
                {
                    var roleLogits = cat(supLogits.ToArray(), 0); // [n, RoleCount]
                    forwardTensors.Add(roleLogits);
                    var roleTarget = tensor(supTargets.ToArray(), dtype: ScalarType.Int64, device: _trainingDevice);
                    var roleWeight = tensor(RoleClassWeights(supTargets), dtype: ScalarType.Float32, device: _trainingDevice);
                    forwardTensors.Add(roleWeight);
                    var roleLoss = nn.functional.cross_entropy(roleLogits, roleTarget, weight: roleWeight) * RoleLossWeight;
                    roleTarget.Dispose();
                    forwardTensors.Add(roleLoss);
                    if (accumulatedLoss is null)
                    {
                        accumulatedLoss = roleLoss;
                    }
                    else
                    {
                        accumulatedLoss = accumulatedLoss + roleLoss;
                        forwardTensors.Add(accumulatedLoss);
                    }
                }
            }

            // PLAN head: CE on the final shared-GRU hidden — which composition SHAPE does this input ask
            // for (arithmetic / predicate / retrieval)? Backprops into the shared encoder like the op head.
            if (supervisePlan)
            {
                var planTrunk = ReasoningTrunk(hInput);
                forwardTensors.Add(planTrunk);
                var planLogits = planTrunk.matmul(_planWT!) + _planB!;
                forwardTensors.Add(planLogits);
                var planBatch = planLogits.unsqueeze(0);
                forwardTensors.Add(planBatch);
                var planTarget = tensor(new long[] { planLabel!.Value }, dtype: ScalarType.Int64, device: _trainingDevice);
                var planClassBalance = ObservePlanClassWeight(planLabel.Value);
                var planLoss = nn.functional.cross_entropy(planBatch, planTarget) * (PlanLossWeight * planClassBalance);
                planTarget.Dispose();
                forwardTensors.Add(planLoss);
                if (accumulatedLoss is null)
                {
                    accumulatedLoss = planLoss;
                }
                else
                {
                    accumulatedLoss = accumulatedLoss + planLoss;
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
    /// Returns <paramref name="trainingParam"/> moved to the inference device when it differs from the
    /// training device (tracking the moved copy in <paramref name="scratch"/> for disposal); otherwise
    /// returns the parameter unchanged and untracked. Exact replacement for the repeated
    /// <c>if (_inferenceDevice != _trainingDevice) { x = p.to(_inferenceDevice); scratch.Add(x); }</c> blocks.
    /// </summary>
    private Tensor ToInfer(Tensor trainingParam, List<Tensor> scratch)
    {
        if (_inferenceDevice != _trainingDevice)
        {
            var moved = trainingParam.to(_inferenceDevice);
            scratch.Add(moved);
            return moved;
        }
        return trainingParam;
    }

    /// <summary>
    /// Encode the input tokens with the shared GRU: h = zeros; for each input token,
    /// h = GruStep(embedding(token), h). The final hidden is the SHARED representation (hInput) that
    /// the router and write-gate read. Every intermediate (embeddings + per-step GRU scratch) is
    /// appended to <paramref name="scratch"/> for disposal after backward(); the returned tensor is
    /// the live graph leaf into hInput and is NOT disposed here.
    /// </summary>
    private Tensor EncodeInput(
        IReadOnlyList<int> inputTokens,
        List<Tensor> scratch,
        Device device,
        List<Tensor>? perTokenStates = null)
    {
        EnsureGruInitialized();
        // Begin each thought from the void (zeros) — the GRU is stateless per input. (Continuity across thoughts is
        // carried by the mind's meaning-space self in the inference engine, not by a hidden-state self here.)
        Tensor h = zeros(new long[] { _hiddenSize }, dtype: ScalarType.Float32, device: device);
        scratch.Add(h);

        foreach (var tok in inputTokens)
        {
            var emb = GetEmbeddingTensor(tok, device);
            scratch.Add(emb);
            h = GruStep(emb, h, scratch, device);
            // Per-token hidden states feed the query operand head ("is THIS token an operand?").
            // They are GruStep outputs already tracked in scratch, so no extra disposal burden.
            perTokenStates?.Add(h);
        }

        return h;
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
}
