using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
    /// <summary>
    /// The GRU CONSTRUCTS its own platonic query from the raw input: which face operation the input
    /// asks for (softmax over the face-derived op vocabulary, 0 = abstain) and which tokens are the
    /// operands (per-token sigmoid). This replaces the pre-GRU <c>GenerateQuery</c> stub — the learned
    /// successor to the hardcoded arithmetic grammar. Returns op 0 (abstain) with empty flags until
    /// the heads have been supervised at least once, so callers degrade gracefully.
    /// </summary>
    public (int OperationId, double Confidence, bool[] OperandFlags) PredictQuery(IReadOnlyList<int> inputTokens, double[]? perception = null)
    {
        if (_queryOpWT is null || inputTokens.Count == 0)
            return (0, 0.0, Array.Empty<bool>());

        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();

        var scratch = new List<Tensor>();
        try
        {
            var perTokenStates = new List<Tensor>();
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice, perTokenStates);

            Tensor opW = ToInfer(_queryOpWT, scratch), opB = ToInfer(_queryOpB!, scratch),
                   owW = ToInfer(_queryOperandWT!, scratch), owB = ToInfer(_queryOperandB!, scratch);
            Tensor tW = ToInfer(_trunkW!, scratch), tB = ToInfer(_trunkB!, scratch);

            // Op head reads the reasoning trunk; the operand head (below) still reads the raw per-token hidden.
            var opTrunk = ReasoningTrunk(hInput, tW, tB); scratch.Add(opTrunk);
            var opLogits = opTrunk.matmul(opW) + opB; scratch.Add(opLogits);
            // SPACE-AWARE: + perception·_queryOpPerceptionW so the op choice can READ the anchor's region
            // (graceful no-op for numeric arithmetic, which has no relational anchor — perception is all-0).
            if (PerceptionQuery && perception is { Length: > 0 } && _queryOpPerceptionW is not null)
            {
                Tensor qpw = ToInfer(_queryOpPerceptionW, scratch);
                var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _inferenceDevice); scratch.Add(pv);
                var perceptLogits = pv.matmul(qpw); scratch.Add(perceptLogits);
                opLogits = opLogits + perceptLogits; scratch.Add(opLogits);
            }
            var opProbs = nn.functional.softmax(opLogits, 0); scratch.Add(opProbs);
            using var opCpu = opProbs.cpu();
            var opScores = opCpu.data<float>().ToArray();

            var best = 0;
            for (var i = 1; i < opScores.Length; i++)
                if (opScores[i] > opScores[best])
                    best = i;

            var flags = new bool[perTokenStates.Count];
            for (var t = 0; t < perTokenStates.Count; t++)
            {
                var score = perTokenStates[t].matmul(owW) + owB; scratch.Add(score);
                var prob = torch.sigmoid(score); scratch.Add(prob);
                using var probCpu = prob.cpu();
                flags[t] = probCpu.data<float>()[0] > 0.5f;
            }

            return (best, opScores.Length > 0 ? opScores[best] : 0.0, flags);
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
    }

    /// <summary>The NN STRUCTURE RECOGNISER: per-token grammatical role from the raw per-token hidden — argmax over
    /// {0=NONE,1=SUBJECT,2=VALUE,3=QUERY} with its softmax confidence. Empty until the role head is trained, so the
    /// field parser degrades gracefully to its bootstrap. This is the learned, generalising replacement for the
    /// hand-coded centrality/tally role classifier (nova-nn-recognizer-space-structural).</summary>
    public (int Role, double Confidence)[] PredictRoles(IReadOnlyList<int> inputTokens)
    {
        if (_roleWT is null || inputTokens.Count == 0)
            return System.Array.Empty<(int, double)>();

        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();
        var scratch = new List<Tensor>();
        try
        {
            var perTokenStates = new List<Tensor>();
            EncodeInput(inputTokens, scratch, _inferenceDevice, perTokenStates);
            Tensor rW = ToInfer(_roleWT, scratch), rB = ToInfer(_roleB!, scratch);
            var roles = new (int, double)[perTokenStates.Count];
            for (var t = 0; t < perTokenStates.Count; t++)
            {
                var logits = perTokenStates[t].matmul(rW) + rB; scratch.Add(logits);
                var probs = nn.functional.softmax(logits, 0); scratch.Add(probs);
                using var cpu = probs.cpu();
                var s = cpu.data<float>().ToArray();
                var best = 0;
                for (var i = 1; i < s.Length; i++) if (s[i] > s[best]) best = i;
                roles[t] = (best, s.Length > 0 ? s[best] : 0.0);
            }
            return roles;
        }
        finally { foreach (var t in scratch) { try { t?.Dispose(); } catch { } } }
    }

    /// <summary>
    /// The GRU classifies which block-COMPOSITION SHAPE the input asks for — the learned composer's shape
    /// selector: 0=none/abstain, 1=arithmetic, 2=predicate, 3=retrieval. Softmax over the plan-kind head.
    /// Returns (0, 0) until the head has been supervised, so callers degrade gracefully.
    /// </summary>
    public (int PlanKind, double Confidence) PredictPlan(IReadOnlyList<int> inputTokens, double[]? perception = null)
    {
        if (_planWT is null || inputTokens.Count == 0)
            return (0, 0.0);

        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();

        var scratch = new List<Tensor>();
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);
            Tensor pW = ToInfer(_planWT, scratch), pB = ToInfer(_planB!, scratch),
                   tW = ToInfer(_trunkW!, scratch), tB = ToInfer(_trunkB!, scratch);
            var planTrunk = ReasoningTrunk(hInput, tW, tB); scratch.Add(planTrunk);
            var logits = planTrunk.matmul(pW) + pB; scratch.Add(logits);
            // SPACE-AWARE: + perception·_planPerceptionW so the shape choice READS the anchor's region.
            if (PerceptionPlan && perception is { Length: > 0 } && _planPerceptionW is not null)
            {
                Tensor ppw = ToInfer(_planPerceptionW, scratch);
                var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _inferenceDevice); scratch.Add(pv);
                var perceptLogits = pv.matmul(ppw); scratch.Add(perceptLogits);
                logits = logits + perceptLogits; scratch.Add(logits);
            }
            var probs = nn.functional.softmax(logits, 0); scratch.Add(probs);
            using var cpu = probs.cpu();
            var scores = cpu.data<float>().ToArray();
            var best = 0;
            for (var i = 1; i < scores.Length; i++)
                if (scores[i] > scores[best]) best = i;
            return (best, scores.Length > 0 ? scores[best] : 0.0);
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
    }

    /// <summary>
    /// Shared inverse-frequency CLASS BALANCE for a classifier head (route / query-op / plan). Under
    /// <paramref name="lck"/>: DECAYS all counts (EMA window) then counts the supervised <paramref name="label"/>
    /// (bounds-checked), then — unless <paramref name="enabled"/> is false (returns 1.0) — returns total /
    /// (<paramref name="k"/> × count) clamped to [0.25, 4.0] so the currently-dominant class can't collapse the
    /// head. Decay is what makes this track the FOCUSED curriculum's recent flood, not the lifetime average.
    /// </summary>
    private static double ObserveClassWeight(double[] counts, object lck, int k, bool enabled, int label)
    {
        lock (lck)
        {
            for (var i = 0; i < counts.Length; i++)
                counts[i] *= ClassBalanceDecay;
            if (label >= 0 && label < counts.Length)
                counts[label] += 1.0;
            if (!enabled)
                return 1.0;
            var total = counts.Sum();
            var classCount = Math.Max(1e-6, label >= 0 && label < counts.Length ? counts[label] : 1.0);
            var weight = total / (k * classCount);
            return Math.Clamp(weight, 0.25, 4.0);
        }
    }

    private double ObserveRouteClassWeight(int routeLabel)
        => ObserveClassWeight(_routeClassCounts, _routeClassBalanceLock, NumRoutes, RouteClassBalanceEnabled, routeLabel);

    /// <summary>
    /// Snapshot of the cumulative SUPERVISED route-label counts (index = route id). Seeded at
    /// {1,1,1} (Laplace) and incremented only when a training example carried a route label — so
    /// route 0 (never labelled by <c>ResolveRouteLabel</c>) stays at its seed, and the route-1 vs
    /// route-2 ratio exposes the class imbalance the balance weight reacts to. Diagnostic only.
    /// </summary>
    public IReadOnlyList<long> RouteClassCounts
    {
        get
        {
            lock (_routeClassBalanceLock)
                return new[] { (long)_routeClassCounts[0], (long)_routeClassCounts[1], (long)_routeClassCounts[2] };
        }
    }

    /// <summary>Inverse-frequency weight for the OP classifier — exact analog of <see
    /// cref="ObserveRouteClassWeight"/>. Counts the supervised op label, then returns total / (classes × count)
    /// clamped to [0.25, 4.0] so the majority op can't dominate the gradient and collapse the head.</summary>
    private double ObserveQueryOpClassWeight(int opLabel)
        => ObserveClassWeight(_queryOpClassCounts, _queryOpClassBalanceLock, QueryOpCount, QueryOpClassBalanceEnabled, opLabel);

    /// <summary>Cumulative supervised OP-label counts (index = op id; 0=abstain,1=add,2=sub,3=mul,4=div). Seeded
    /// {1,…} (Laplace); the spread exposes the imbalance the op class-balance weight reacts to. Diagnostic.</summary>
    public IReadOnlyList<long> QueryOpClassCounts
    {
        get { lock (_queryOpClassBalanceLock) return _queryOpClassCounts.Select(c => (long)c).ToList(); }
    }

    /// <summary>Inverse-frequency weight for the PLAN/shape head — same mechanism as the op/route heads.</summary>
    private double ObservePlanClassWeight(int planLabel)
        => ObserveClassWeight(_planClassCounts, _planClassBalanceLock, PlanKindCount, PlanClassBalanceEnabled, planLabel);

    /// <summary>Cumulative supervised PLAN-label counts (index = plan kind). Diagnostic.</summary>
    public IReadOnlyList<long> PlanClassCounts
    {
        get { lock (_planClassBalanceLock) return _planClassCounts.Select(c => (long)c).ToList(); }
    }

    private void EnsureRouteHeadInitialized()
    {
        EnsureReasoningTrunk(); // route head reads the trunk; ensure it exists (idempotent)
        if (_routeWT is null)
        {
            _routeWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { ReasoningTrunkDim, NumRoutes }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _routeB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { NumRoutes }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            RecreateOptimizer();
        }
        // Perception-routing weight ensured independently (a loaded checkpoint sets _routeWT but not this
        // runtime-only weight). Trained by ReinforceRouteHead, NOT the shared optimizer.
        _routePerceptionW ??= new TorchSharp.Modules.Parameter(
            ((rand(new long[] { EditPerceptionDim, NumRoutes }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
    }

    private void EnsureQueryHeadsInitialized()
    {
        EnsureReasoningTrunk(); // op + plan heads read the trunk; ensure it exists (idempotent)
        if (_queryOpWT is null)
        {
            // Op + plan heads read the REASONING TRUNK ([ReasoningTrunkDim, K]); the operand head still reads
            // the raw per-token hidden ([hidden, 1]).
            _queryOpWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { ReasoningTrunkDim, QueryOpCount }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _queryOpB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { QueryOpCount }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            _queryOperandWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { _hiddenSize, 1 }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _queryOperandB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            _planWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { ReasoningTrunkDim, PlanKindCount }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _planB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { PlanKindCount }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            RecreateOptimizer();
        }
        // SPACE-AWARE perception weights ensured INDEPENDENTLY (a loaded checkpoint sets the heads but not
        // these runtime-only weights). Trained by TrainQueryOpPerception / TrainPlanPerception, NOT the shared
        // optimizer — same discipline as _routePerceptionW / _editPerceptionW.
        _queryOpPerceptionW ??= new TorchSharp.Modules.Parameter(
            ((rand(new long[] { EditPerceptionDim, QueryOpCount }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _planPerceptionW ??= new TorchSharp.Modules.Parameter(
            ((rand(new long[] { EditPerceptionDim, PlanKindCount }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
    }

    // The per-token ROLE head (the structure recogniser). Independent of the op/plan heads — it reads the raw
    // per-token hidden like the operand head, so it is initialised lazily on the first role-supervised example.
    private void EnsureRoleHeadInitialized()
    {
        if (_roleWT is not null) return;
        _roleWT = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { _hiddenSize, RoleCount }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _roleB = new TorchSharp.Modules.Parameter(
            zeros(new long[] { RoleCount }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        RecreateOptimizer();
    }

    private void EnsureEditHeadInitialized()
    {
        if (_editWT is null)
        {
            // Mirror EnsureRouteHeadInitialized, but emit a single scalar ([hidden, 1] + [1]).
            // Deliberately NOT added to _optimizer / RecreateOptimizer: the platonic space is a plain
            // double[] store, so there is no autograd path from a space edit back to these weights. The
            // head is trained only by ReinforceEditHead's manual REINFORCE step.
            _editWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { _hiddenSize, 1 }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _editB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { 1 }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        }
        // Perception weight is ensured independently (a loaded checkpoint sets _editWT but not this runtime-only
        // weight), so the read-before-write head always has its perception input available when fed one.
        _editPerceptionW ??= new TorchSharp.Modules.Parameter(
            ((rand(new long[] { EditPerceptionDim, 1 }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
    }

    public double PredictEditMagnitude(IReadOnlyList<int> inputTokens) => PredictEditMagnitude(inputTokens, null);

    /// <summary>As <see cref="PredictEditMagnitude(IReadOnlyList{int})"/>, but the magnitude ALSO conditions on a
    /// space-PERCEPTION vector (rank-of-target / distractor-winning / …), so the head can learn a state-dependent
    /// read-before-write policy. Pass null for the token-only path (unchanged default behaviour).</summary>
    public double PredictEditMagnitude(IReadOnlyList<int> inputTokens, double[]? perception)
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
            Tensor editW = ToInfer(_editWT!, scratch), editB = ToInfer(_editB!, scratch);
            // hInput: [hidden]; editW: [hidden, 1] -> matmul yields [1]; add bias [1]; sigmoid -> [0,1].
            var raw = hInput.matmul(editW) + editB; scratch.Add(raw);
            // Perception term: + perceptionVec[P] · _editPerceptionW[P,1] -> [1]. Lets the magnitude depend on
            // the CURRENT space state, not just the tokens (the read-before-write signal).
            if (perception is { Length: > 0 } && _editPerceptionW is not null)
            {
                var pw = _editPerceptionW.to(_inferenceDevice); scratch.Add(pw);
                var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _inferenceDevice); scratch.Add(pv);
                var praw = raw + pv.matmul(pw); scratch.Add(praw);
                raw = praw;
            }
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
        => ReinforceEditHead(inputTokens, null, appliedMagnitude, reward);

    /// <summary>As the token-only REINFORCE, but also updates the PERCEPTION weight so the head LEARNS to map the
    /// space-state readout to its action (read-before-write). Pass null perception for the token-only path.</summary>
    public void ReinforceEditHead(IReadOnlyList<int> inputTokens, double[]? perception, double appliedMagnitude, double reward)
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
        var usePerc = perception is { Length: > 0 } && _editPerceptionW is not null;
        using var pv = usePerc
            ? tensor(PerceptionFloats(perception!), new long[] { EditPerceptionDim }, device: _trainingDevice)
            : null;
        // raw = tokens·editW [+ perception·perceptionW] + editB.
        using var raw = usePerc
            ? meanEmb.matmul(_editWT!) + pv!.matmul(_editPerceptionW!) + _editB!
            : meanEmb.matmul(_editWT!) + _editB!;
        using var predicted = torch.sigmoid(raw);              // [1], requires grad
        using var target = tensor(new float[] { action }, dtype: ScalarType.Float32, device: _trainingDevice);
        using var diff = predicted - target;
        using var sq = diff * diff;
        // Surrogate loss; minimizing moves 'predicted' toward action when reward>0, away when <0.
        using var surrogate = sq * boundedReward;
        using var loss = surrogate.sum();

        // Gradient of the surrogate w.r.t. ONLY the edit-head params (+ the perception weight when fed) via
        // autograd.grad, which RETURNS the gradients instead of populating .grad. This keeps the shared GRU's
        // .grad untouched and avoids the non-leaf .grad warning/no-op.
        var gradTargets = usePerc
            ? new List<Tensor> { _editWT!, _editB!, _editPerceptionW! }
            : new List<Tensor> { _editWT!, _editB! };
        var editGrads = autograd.grad(new List<Tensor> { loss }, gradTargets);
        try
        {
            using (no_grad())
            {
                using var stepW = editGrads[0] * EditHeadLearningRate;
                _editWT!.sub_(stepW);
                using var stepB = editGrads[1] * EditHeadLearningRate;
                _editB!.sub_(stepB);
                if (usePerc)
                {
                    using var stepP = editGrads[2] * EditHeadLearningRate;
                    _editPerceptionW!.sub_(stepP);
                }
            }
        }
        finally
        {
            foreach (var g in editGrads)
            {
                try { g.Dispose(); } catch { /* best effort */ }
            }
        }
    }

    public (int RouteId, double Confidence) PredictRoute(IReadOnlyList<int> inputTokens) => PredictRoute(inputTokens, null);

    /// <summary>As <see cref="PredictRoute(IReadOnlyList{int})"/>, but the route logits ALSO condition on a
    /// TARGET-AGNOSTIC space-perception vector (perceived retrievability), so the GRU can route platonic-vs-neural
    /// from what the space can actually answer. Active only when <see cref="PerceptionRouting"/> + perception given;
    /// otherwise identical to the token-only route.</summary>
    public (int RouteId, double Confidence) PredictRoute(IReadOnlyList<int> inputTokens, double[]? perception)
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
            Tensor routeW = ToInfer(_routeWT!, scratch), routeB = ToInfer(_routeB!, scratch),
                   tW = ToInfer(_trunkW!, scratch), tB = ToInfer(_trunkB!, scratch);
            var routeTrunk = ReasoningTrunk(hInput, tW, tB); scratch.Add(routeTrunk);
            var logits = routeTrunk.matmul(routeW) + routeB; scratch.Add(logits);
            // Perception term: route logits += routePerceptionVec[P] · _routePerceptionW[P, RouteCount].
            if (PerceptionRouting && perception is { Length: > 0 } && _routePerceptionW is not null)
            {
                var rpw = _routePerceptionW.to(_inferenceDevice); scratch.Add(rpw);
                var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _inferenceDevice); scratch.Add(pv);
                var biased = logits + pv.matmul(rpw); scratch.Add(biased);
                logits = biased;
            }
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

    /// <summary>REINFORCE the route PERCEPTION weight toward <paramref name="routeId"/> (reward&gt;0 ⇒ raise that
    /// route's probability for this perceived state). Trains ONLY <c>_routePerceptionW</c> — the main route head
    /// stays on its label/autograd training — so the GRU learns the perceived-retrievability→route mapping.
    /// No-op unless <see cref="PerceptionRouting"/>.</summary>
    public void ReinforceRouteHead(IReadOnlyList<int> inputTokens, double[]? perception, int routeId, double reward)
    {
        if (!PerceptionRouting || inputTokens.Count == 0 || perception is not { Length: > 0 })
            return;
        if (double.IsNaN(reward) || double.IsInfinity(reward) || reward == 0.0)
            return;
        if (routeId < 0 || routeId >= NumRoutes)
            return;

        EnsureModelInitialized();
        EnsureGruInitialized();
        EnsureRouteHeadInitialized();
        if (_routeWT is null || _routeB is null || _routePerceptionW is null)
            return;

        var boundedReward = (float)Math.Clamp(reward, -EditHeadRewardClamp, EditHeadRewardClamp);

        Tensor meanEmb;
        var encScratch = new List<Tensor>();
        try
        {
            using (no_grad())
            {
                var hInput = EncodeInput(inputTokens, encScratch, _trainingDevice);
                meanEmb = hInput.detach().clone();
            }
        }
        finally
        {
            foreach (var t in encScratch) { try { t?.Dispose(); } catch { } }
        }

        using var meanEmbHolder = meanEmb;
        using var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _trainingDevice);
        // logits = trunk(h)·routeW + perception·routePerceptionW + routeB; grad requested ONLY for
        // _routePerceptionW (the trunk + route head are trained by the autograd CE path, not REINFORCE).
        using var hTrunk = ReasoningTrunk(meanEmb);
        using var logits = hTrunk.matmul(_routeWT!) + pv.matmul(_routePerceptionW!) + _routeB!;
        using var logp = nn.functional.log_softmax(logits, 0);
        using var chosen = logp.narrow(0, routeId, 1);
        using var loss = (chosen * (-boundedReward)).sum(); // minimize -reward·logp[routeId]
        var grads = autograd.grad(new List<Tensor> { loss }, new List<Tensor> { _routePerceptionW! });
        try
        {
            using (no_grad())
            {
                using var step = grads[0] * EditHeadLearningRate;
                _routePerceptionW!.sub_(step);
            }
        }
        finally
        {
            foreach (var g in grads) { try { g.Dispose(); } catch { } }
        }
    }

    /// <summary>SPACE-AWARE plan head (SPACE_AWARE_GRU.md §A): supervised CE training of ONLY
    /// <c>_planPerceptionW</c> — the perceived anchor region shifts shape selection toward the correct plan
    /// kind; the main plan head keeps its shared-optimizer training. No-op unless <see cref="PerceptionPlan"/>
    /// and a perception vector + valid label are given.</summary>
    public void TrainPlanPerception(IReadOnlyList<int> inputTokens, double[]? perception, int planLabel)
        => TrainHeadPerception(inputTokens, perception, planLabel, PerceptionPlan, PlanKindCount,
            _planWT, _planB, _planPerceptionW);

    /// <summary>SPACE-AWARE op head: supervised CE training of ONLY <c>_queryOpPerceptionW</c>.</summary>
    public void TrainQueryOpPerception(IReadOnlyList<int> inputTokens, double[]? perception, int opLabel)
        => TrainHeadPerception(inputTokens, perception, opLabel, PerceptionQuery, QueryOpCount,
            _queryOpWT, _queryOpB, _queryOpPerceptionW);

    /// <summary>Shared supervised-CE perception trainer: logits = h·W + perception·perceptionW + b, gradient
    /// requested ONLY for perceptionW, which is updated IN PLACE (sub_), so the parameters can be passed by
    /// value — the main head W/b keep their own training. Same detached-encode + manual-step discipline as
    /// <see cref="ReinforceRouteHead"/>.</summary>
    private void TrainHeadPerception(IReadOnlyList<int> inputTokens, double[]? perception, int label,
        bool enabled, int classCount,
        TorchSharp.Modules.Parameter? headW, TorchSharp.Modules.Parameter? headB,
        TorchSharp.Modules.Parameter? perceptionW)
    {
        if (!enabled || inputTokens.Count == 0 || perception is not { Length: > 0 })
            return;
        if (label < 0 || label >= classCount)
            return;

        EnsureModelInitialized();
        EnsureGruInitialized();
        EnsureQueryHeadsInitialized();
        if (headW is null || headB is null || perceptionW is null)
            return;

        Tensor meanEmb;
        var encScratch = new List<Tensor>();
        try
        {
            using (no_grad())
            {
                var hInput = EncodeInput(inputTokens, encScratch, _trainingDevice);
                meanEmb = hInput.detach().clone();
            }
        }
        finally
        {
            foreach (var t in encScratch) { try { t?.Dispose(); } catch { } }
        }

        using var meanEmbHolder = meanEmb;
        using var pv = tensor(PerceptionFloats(perception), new long[] { EditPerceptionDim }, device: _trainingDevice);
        // op + plan heads read the reasoning trunk (this trainer is only called for those two heads).
        using var hTrunk = ReasoningTrunk(meanEmb);
        using var logits = hTrunk.matmul(headW!) + pv.matmul(perceptionW!) + headB!;
        using var batch = logits.unsqueeze(0);
        using var target = tensor(new long[] { label }, dtype: ScalarType.Int64, device: _trainingDevice);
        using var loss = nn.functional.cross_entropy(batch, target);
        var grads = autograd.grad(new List<Tensor> { loss }, new List<Tensor> { perceptionW! });
        try
        {
            using (no_grad())
            {
                using var step = grads[0] * EditHeadLearningRate;
                perceptionW!.sub_(step);
            }
        }
        finally
        {
            foreach (var g in grads) { try { g.Dispose(); } catch { } }
        }
    }
}
