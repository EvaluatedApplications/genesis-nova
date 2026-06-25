using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
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
            _gruBhh is not null ? TensorToVector(_gruBhh) : null,
            // Query-construction + plan heads (null until lazily initialised; created together so all-or-none).
            _queryOpWT is not null ? TensorToMatrix(_queryOpWT) : null,
            _queryOpB is not null ? TensorToVector(_queryOpB) : null,
            _queryOperandWT is not null ? TensorToMatrix(_queryOperandWT) : null,
            _queryOperandB is not null ? TensorToVector(_queryOperandB) : null,
            _planWT is not null ? TensorToMatrix(_planWT) : null,
            _planB is not null ? TensorToVector(_planB) : null,
            _trunkW is not null ? TensorToMatrix(_trunkW) : null,
            _trunkB is not null ? TensorToVector(_trunkB) : null);
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
        // QUERY-construction + PLAN heads — restore ALL SIX or NONE (they are created together by
        // EnsureQueryHeadsInitialized, which returns early once _queryOpWT exists; a partial restore would leave
        // the rest null and NPE). Shape-mismatched / pre-head checkpoints leave them null → lazily reinitialised.
        if (HasUsableQueryPlanHeads(snapshot, _hiddenSize))
        {
            _queryOpWT = MatrixToParameter(snapshot.QueryOpWeights!);
            _queryOpB = VectorToParameter(snapshot.QueryOpBias!);
            _queryOperandWT = MatrixToParameter(snapshot.QueryOperandWeights!);
            _queryOperandB = VectorToParameter(snapshot.QueryOperandBias!);
            _planWT = MatrixToParameter(snapshot.PlanWeights!);
            _planB = VectorToParameter(snapshot.PlanBias!);
        }
        // SHARED REASONING TRUNK — restore if shape-compatible, else leave null so the next forward
        // lazily reinitialises it (graceful, like the heads). Persisted so routing survives a reload.
        if (HasUsableTrunk(snapshot.TrunkWeights, snapshot.TrunkBias, _hiddenSize))
        {
            _trunkW = MatrixToParameter(snapshot.TrunkWeights!);
            _trunkB = VectorToParameter(snapshot.TrunkBias!);
        }
        RecreateOptimizer();
    }

    private static bool HasUsableQueryPlanHeads(ModelSnapshot s, int hiddenSize)
    {
        if (hiddenSize <= 0) return false;
        if (s.QueryOpWeights is null || s.QueryOpBias is null || s.QueryOperandWeights is null
            || s.QueryOperandBias is null || s.PlanWeights is null || s.PlanBias is null)
            return false;
        // Op + plan heads read the REASONING TRUNK, so their input dim is ReasoningTrunkDim (NOT hiddenSize).
        // The operand head still reads the raw per-token hidden, so it stays [hiddenSize, 1].
        return s.QueryOpWeights.GetLength(0) == ReasoningTrunkDim && s.QueryOpWeights.GetLength(1) == QueryOpCount
            && s.QueryOpBias.Length == QueryOpCount
            && s.QueryOperandWeights.GetLength(0) == hiddenSize && s.QueryOperandWeights.GetLength(1) == 1
            && s.QueryOperandBias.Length == 1
            && s.PlanWeights.GetLength(0) == ReasoningTrunkDim && s.PlanWeights.GetLength(1) == PlanKindCount
            && s.PlanBias.Length == PlanKindCount;
    }

    private static bool HasUsableTrunk(double[,]? trunkWeights, double[]? trunkBias, int hiddenSize)
    {
        if (trunkWeights is null || trunkBias is null || hiddenSize <= 0)
            return false;
        return trunkWeights.GetLength(0) == hiddenSize && trunkWeights.GetLength(1) == ReasoningTrunkDim
            && trunkBias.Length == ReasoningTrunkDim;
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

            // Platonic-query construction heads — clone+detach like the route head (autograd params).
            if (_queryOpWT is not null)
            {
                var qoClone = _queryOpWT.clone().detach().to(_trainingDevice);
                var qobClone = _queryOpB!.clone().detach().to(_trainingDevice);
                var qwClone = _queryOperandWT!.clone().detach().to(_trainingDevice);
                var qwbClone = _queryOperandB!.clone().detach().to(_trainingDevice);
                var oldQo = _queryOpWT;
                var oldQob = _queryOpB;
                var oldQw = _queryOperandWT;
                var oldQwb = _queryOperandB;
                _queryOpWT = new TorchSharp.Modules.Parameter(qoClone, true);
                _queryOpB = new TorchSharp.Modules.Parameter(qobClone, true);
                _queryOperandWT = new TorchSharp.Modules.Parameter(qwClone, true);
                _queryOperandB = new TorchSharp.Modules.Parameter(qwbClone, true);
                try { oldQo.Dispose(); } catch { }
                try { oldQob!.Dispose(); } catch { }
                try { oldQw!.Dispose(); } catch { }
                try { oldQwb!.Dispose(); } catch { }
            }

            // PLAN head — clone+detach like the query heads (autograd param).
            if (_planWT is not null)
            {
                var pwClone = _planWT.clone().detach().to(_trainingDevice);
                var pbClone = _planB!.clone().detach().to(_trainingDevice);
                var oldPw = _planWT;
                var oldPb = _planB;
                _planWT = new TorchSharp.Modules.Parameter(pwClone, true);
                _planB = new TorchSharp.Modules.Parameter(pbClone, true);
                try { oldPw.Dispose(); } catch { }
                try { oldPb!.Dispose(); } catch { }
            }

            // PER-TOKEN ROLE head — clone+detach like the query heads (autograd param).
            if (_roleWT is not null)
            {
                var rwClone = _roleWT.clone().detach().to(_trainingDevice);
                var rbClone = _roleB!.clone().detach().to(_trainingDevice);
                var oldRw = _roleWT;
                var oldRb = _roleB;
                _roleWT = new TorchSharp.Modules.Parameter(rwClone, true);
                _roleB = new TorchSharp.Modules.Parameter(rbClone, true);
                try { oldRw.Dispose(); } catch { }
                try { oldRb!.Dispose(); } catch { }
            }

            // SHARED REASONING TRUNK — clone+detach like the heads (autograd param read by route/op/plan).
            if (_trunkW is not null)
            {
                var twClone = _trunkW.clone().detach().to(_trainingDevice);
                var tbClone = _trunkB!.clone().detach().to(_trainingDevice);
                var oldTw = _trunkW;
                var oldTb = _trunkB;
                _trunkW = new TorchSharp.Modules.Parameter(twClone, true);
                _trunkB = new TorchSharp.Modules.Parameter(tbClone, true);
                try { oldTw.Dispose(); } catch { }
                try { oldTb!.Dispose(); } catch { }
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

        // SHARED REASONING TRUNK is [oldHidden, ReasoningTrunkDim] — its first dim is the hidden size, so it is
        // incompatible after a hidden resize; reinitialize at the new hidden size (the heads below read it).
        if (_trunkW is not null)
        {
            var oldTw = _trunkW;
            var oldTb = _trunkB;
            _trunkW = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { hiddenSize, ReasoningTrunkDim }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
            _trunkB = new TorchSharp.Modules.Parameter(
                zeros(new long[] { ReasoningTrunkDim }, dtype: ScalarType.Float32, device: _trainingDevice), true);
            try { oldTw.Dispose(); } catch { }
            try { oldTb?.Dispose(); } catch { }
        }

        // Route head reads the trunk so its shape is [ReasoningTrunkDim, NumRoutes] (hidden-independent), but the
        // trunk it was trained against just reset, so reinitialize it fresh too.
        if (_routeWT is not null)
        {
            var oldRW = _routeWT;
            var oldRB = _routeB;
            _routeWT = new TorchSharp.Modules.Parameter(
                ((rand(new long[] { ReasoningTrunkDim, NumRoutes }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
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

        // Platonic-query heads are [oldHidden, QueryOpCount]/[oldHidden, 1] — incompatible after a
        // hidden resize (the runtime auto-scales hidden to VRAM on start). Dispose and null them so
        // they lazily reinitialize at the new hidden size on the next supervised query label —
        // leaving the old-shape parameters registered corrupts the optimizer/forward.
        if (_queryOpWT is not null)
        {
            var oldQo = _queryOpWT;
            var oldQob = _queryOpB;
            var oldQw = _queryOperandWT;
            var oldQwb = _queryOperandB;
            _queryOpWT = null;
            _queryOpB = null;
            _queryOperandWT = null;
            _queryOperandB = null;
            try { oldQo.Dispose(); } catch { }
            try { oldQob?.Dispose(); } catch { }
            try { oldQw?.Dispose(); } catch { }
            try { oldQwb?.Dispose(); } catch { }
        }

        // PLAN head is [oldHidden, PlanKindCount] — incompatible after a hidden resize; null it so it
        // lazily reinitializes at the new hidden size (same graceful degradation as the query heads).
        if (_planWT is not null)
        {
            var oldPw = _planWT;
            var oldPb = _planB;
            _planWT = null;
            _planB = null;
            try { oldPw.Dispose(); } catch { }
            try { oldPb?.Dispose(); } catch { }
        }

        // PER-TOKEN ROLE head is [oldHidden, RoleCount] — hidden-dependent; null it so it lazily reinitializes at
        // the new hidden size (same graceful degradation as the operand/query heads).
        if (_roleWT is not null)
        {
            var oldRw = _roleWT;
            var oldRb = _roleB;
            _roleWT = null;
            _roleB = null;
            try { oldRw.Dispose(); } catch { }
            try { oldRb?.Dispose(); } catch { }
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
        // Platonic-query construction heads — autograd-trained (CE/BCE), unlike the REINFORCE edit head.
        if (_queryOpWT is not null) parameters.Add(_queryOpWT);
        if (_queryOpB is not null) parameters.Add(_queryOpB!);
        if (_queryOperandWT is not null) parameters.Add(_queryOperandWT!);
        if (_queryOperandB is not null) parameters.Add(_queryOperandB!);
        if (_planWT is not null) parameters.Add(_planWT);
        if (_planB is not null) parameters.Add(_planB!);
        // PER-TOKEN ROLE head — autograd-trained (per-token CE), like the operand head.
        if (_roleWT is not null) parameters.Add(_roleWT);
        if (_roleB is not null) parameters.Add(_roleB!);
        // SHARED REASONING TRUNK — autograd-trained with the heads it feeds (route/op/plan).
        if (_trunkW is not null) parameters.Add(_trunkW);
        if (_trunkB is not null) parameters.Add(_trunkB!);
        _optimizer = torch.optim.SGD(parameters, _currentLearningRate);
    }

    /// <summary>
    /// Total trainable parameter count across all (lazily-initialised) tensors — for footprint reporting
    /// and like-for-like model-size comparisons. Counts every parameter on the GPU, including the
    /// REINFORCE-trained edit head. (The platonic space itself is CPU-side double[] state, not counted here.)
    /// </summary>
    public long ParameterCount()
    {
        long n = 0;
        foreach (var p in new[]
        {
            _embT, _wOutT, _bOutT, _routeWT, _routeB,
            _gruWih, _gruWhh, _gruBih, _gruBhh,
            _queryOpWT, _queryOpB, _queryOperandWT, _queryOperandB, _planWT, _planB, _editWT, _editB,
            _trunkW, _trunkB
        })
            if (p is not null) n += p.numel();
        return n;
    }

    /// <summary>
    /// Current SGD step size. Defaults to <c>config.LearningRate</c> but is ADJUSTABLE so a training
    /// regime can ANNEAL it — the single most important lever against the "reaches ~90% then
    /// oscillates" failure: a fixed step overshoots the optimum once most examples are right, so the
    /// last hard ones flip-flop forever. The optimizer is rebuilt every step (CloneParametersToBreakGraph
    /// → RecreateOptimizer), so a new rate takes effect on the very next step.
    /// </summary>
    public double LearningRate
    {
        get => _currentLearningRate;
        set
        {
            var clamped = Math.Clamp(value, 1e-6, 10.0);
            if (Math.Abs(clamped - _currentLearningRate) < 1e-12)
                return;
            _currentLearningRate = clamped;
            if (_optimizer is not null)
                RecreateOptimizer();
        }
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
        _routePerceptionW = null;
        try { _editWT?.Dispose(); } catch { }
        try { _editB?.Dispose(); } catch { }
        _editWT = null;
        _editB = null;
        _editPerceptionW = null;
        // Platonic-query construction heads. MUST be torn down with the rest: leaving them alive
        // across Import (auto-resume restart) registers STALE parameters from the previous session's
        // graph epoch into the fresh optimizer, corrupting every subsequent training step. They are
        // not persisted, so they lazily reinitialize on the first supervised query label.
        try { _queryOpWT?.Dispose(); } catch { }
        try { _queryOpB?.Dispose(); } catch { }
        try { _queryOperandWT?.Dispose(); } catch { }
        try { _queryOperandB?.Dispose(); } catch { }
        _queryOpWT = null;
        _queryOpB = null;
        _queryOperandWT = null;
        _queryOperandB = null;
        _queryOpPerceptionW = null; // runtime-only perception weight (matches _routePerceptionW handling)
        try { _planWT?.Dispose(); } catch { }
        try { _planB?.Dispose(); } catch { }
        _planWT = null;
        _planB = null;
        _planPerceptionW = null;
        // PER-TOKEN ROLE head — torn down with the rest (same discipline; not persisted yet → lazily reinitialises).
        try { _roleWT?.Dispose(); } catch { }
        try { _roleB?.Dispose(); } catch { }
        _roleWT = null;
        _roleB = null;
        // SHARED REASONING TRUNK — torn down with the heads it feeds (same discipline).
        try { _trunkW?.Dispose(); } catch { }
        try { _trunkB?.Dispose(); } catch { }
        _trunkW = null;
        _trunkB = null;
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
        // The route head reads the REASONING TRUNK, so its input dim is ReasoningTrunkDim (NOT hiddenSize).
        return rows == ReasoningTrunkDim && rows > 0 && cols == NumRoutes && routeBias.Length == cols;
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
