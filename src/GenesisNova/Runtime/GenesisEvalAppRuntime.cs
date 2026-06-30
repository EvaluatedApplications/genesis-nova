using EvalApp.Consumer;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Licensing;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed partial class GenesisEvalAppRuntime : ILearningRuntime
{
    private readonly GenesisNovaConfig _runtimeConfig;
    private readonly GenesisRuntimeState _state;
    private readonly ICompiledPipeline<GenesisTrainTaskData> _trainPipeline;
    private readonly ICompiledPipeline<GenesisTrainOneTaskData> _trainOnePipeline;
    private readonly ICompiledPipeline<GenesisPredictTaskData> _predictPipeline;
    private readonly ICompiledPipeline<GenesisSaveTaskData> _savePipeline;
    private readonly ICompiledPipeline<GenesisLoadTaskData> _loadPipeline;
    private readonly ICompiledPipeline<GenesisAutonomousTrainTaskData> _autonomousRoundPipeline;
    
    private readonly BestLossTracker _lossTracker = new();
    private readonly AutonomousHistoryStore _historyStore = new();
    private readonly GenesisCheckpointPersister _persister;
    private readonly SemaphoreSlim _modelOpsGate = new(1, 1);
    private string? _loadedCheckpointPath;
    private DateTime _loadedCheckpointWriteUtc = DateTime.MinValue;
    // SESSION conversational mode (the talk route). Held on the runtime — NOT in the checkpoint — because a reload
    // rebuilds the inference engine (TalkEnabled defaults off), so it must be RE-APPLIED after every Replace, else an
    // autosave mid-gym silently switches the persona off on the next predict. See SetConversationalMode / Replace.
    private bool _conversationalMode;
    // Count of checkpoint RELOADS (state-Replace). A reload mid-training is the "weird on restart" hazard; surfaced so
    // diagnostics/tests can assert the gym's OWN autosave does NOT trigger a self-reload on the next predict.
    private int _reloadCount;
    public int ReloadCount => _reloadCount;

    /// <summary>The underlying runtime state (model/space/trainer). Test seam — lets tests warm the role head the way
    /// the gym does in production, instead of relying on a (removed) hardcoded grammar fallback.</summary>
    internal GenesisRuntimeState State => _state;

    public GenesisEvalAppRuntime(GenesisNovaConfig? config = null)
    {
        _runtimeConfig = config ?? new GenesisNovaConfig();
        _state = new GenesisRuntimeState(_runtimeConfig);

        // Validate license for adaptive tuning
        var licenseMode = GenesisPipelineValidator.ValidateLicense();
        if (licenseMode == LicenseMode.Licensed)
        {
            Console.WriteLine("[License] ✓ Genesis Nova licensed - full adaptive tuning enabled");
        }
        else
        {
            Console.WriteLine("[License] ⓘ Genesis Nova unlicensed - running in sequential mode");
        }

        ICompiledPipeline<GenesisTrainTaskData> train = null!;
        ICompiledPipeline<GenesisTrainOneTaskData> trainOne = null!;
        ICompiledPipeline<GenesisPredictTaskData> predict = null!;
        ICompiledPipeline<GenesisSaveTaskData> save = null!;
        ICompiledPipeline<GenesisLoadTaskData> load = null!;
        ICompiledPipeline<GenesisAutonomousTrainTaskData> autonomousRound = null!;

        var autonomousPlanner = new GenesisAutonomousTrainingPlanner();
        var planAutonomousRoundStep = new PlanAutonomousRoundStep(autonomousPlanner);
        var generateCandidatePoolItemStep = new GenerateCandidatePoolItemStep();
        var buildAutonomousBatchStep = new BuildAutonomousTrainingBatchStep();
        var runAutonomousRoundStep = new RunAutonomousTrainingRoundStep(_state.Orchestrator);

        // Data-driven parallelism for the candidate-pool prep ForEach: derived from
        // measured VRAM headroom + CPU width (see GpuResourceGatePlanner). The pipeline
        // (and therefore the ForEach TunableConfig) is compiled once here, before any
        // per-call request exists, so we bound the build-time ceiling by the request's
        // default MaxGenerationConcurrency. The adaptive tuner then scales within
        // [1, max] at runtime; lower per-request caps are additionally honored by the
        // planner when sizing per-creator pools.
        var poolPrepParallelism = GpuResourceGatePlanner.CandidatePoolPrepGate(
            requestedConcurrency: new GenesisAutonomousTrainingRequest().MaxGenerationConcurrency);

        _persister = new GenesisCheckpointPersister(_state, _runtimeConfig, _historyStore);

        var loadExamplesStep = new LoadExamplesStep();
        var runTrainingStep = new RunTrainingStep(_state);
        var saveCheckpointStep = new SaveCheckpointStep(_lossTracker, _persister);
        var trainOneStep = new TrainOneStep(_state);
        var predictGpuStep = new PredictGpuStep(_state);
        var saveStep = new SaveStep(_persister);
        var loadStep = new LoadStep(_state, _historyStore, _persister, _runtimeConfig);

        Eval.App("GenesisNovaRuntime")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Cpu, new TunableConfig(Min: 1, Max: Environment.ProcessorCount, Default: Math.Max(2, Environment.ProcessorCount / 2)))
            // GPU kernel concurrency stays serialized (single CUDA stream); see GpuResourceGatePlanner.
            .WithResource(ResourceKind.Of("gpu"), GpuResourceGatePlanner.GpuGate())
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithTuning()  // Enable adaptive tuning (requires license)
            .DefineDomain("GenesisNova", _state)
                .DefineTask<GenesisTrainTaskData>("Train")
                    .Gate(ResourceKind.Cpu, null, g => g
                        .AddStep("LoadExamples", loadExamplesStep)
                        .AddStep("RunTraining", runTrainingStep))
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("SaveCheckpoint", saveCheckpointStep))
                    .Run(out train)
                .DefineTask<GenesisTrainOneTaskData>("TrainOne")
                    .AddStep("TrainOneStep", trainOneStep)
                    .Run(out trainOne)
                .DefineTask<GenesisPredictTaskData>("Predict")
                    .Gate(ResourceKind.Of("gpu"), null, g => g.AddStep("PredictGpu", predictGpuStep))
                    .Run(out predict)
                .DefineTask<GenesisSaveTaskData>("Save")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Save", saveStep))
                    .Run(out save)
                .DefineTask<GenesisLoadTaskData>("Load")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Load", loadStep))
                    .Run(out load)
                .DefineTask<GenesisAutonomousTrainTaskData>("AutonomousTrainRound")
                    .AddStep("PlanRound", planAutonomousRoundStep)
                    .Gate(ResourceKind.Cpu, null, g => g
                        // Candidate-pool generation: resource-gated, VRAM/CPU-tunable ForEach.
                        // Replaces the former raw Parallel.ForEach so per-creator generation
                        // participates in the shared CPU gate + adaptive tuner. Parallelism
                        // bounds are derived from measured VRAM headroom (see GpuResourceGatePlanner).
                        // NOTE: the per-item body is NOT re-gated on Cpu. Nesting the SAME resource
                        // gate inside the outer Cpu gate self-deadlocks at 0% CPU — the parent holds
                        // the Cpu budget the children wait for. The outer gate plus the bounded
                        // ForEach parallelism already throttle per-creator generation.
                        .ForEach(
                            GenerateAutonomousCandidatePoolsStep.SelectWorkItems,
                            GenerateAutonomousCandidatePoolsStep.MergePools,
                            "CandidatePools",
                            poolPrepParallelism,
                            sub => sub.AddStep("GeneratePool", generateCandidatePoolItemStep))
                        .AddStep("BuildBatch", buildAutonomousBatchStep)
                        .AddStep("RunTraining", runAutonomousRoundStep))
                    .Run(out autonomousRound)
                .Build(GenesisPipelineValidator.ActiveKey); // EVALAPP license → adaptive CPU/GPU tuning (else sequential)

        TryBootstrapLatestState();
        EnsureConfiguredHiddenSize();

        _trainPipeline = train;
        _trainOnePipeline = trainOne;
        _predictPipeline = predict;
        _savePipeline = save;
        _loadPipeline = load;
        _autonomousRoundPipeline = autonomousRound;
    }

    /// <summary>
    /// Register an operation token (a language verb, e.g. "find") as a ROUTE TRIGGER: it is excluded from
    /// relation coupling/anchoring (so it never overloads a target and collapses retrieval) while the GRU
    /// route head still learns it from the raw input. Re-register from the language definition at startup —
    /// it is not persisted, being a pure function of that definition. See LANGUAGE_CREATOR.md §2.
    /// </summary>
    public void RegisterOperationToken(string token) => _state.Memory.RegisterOperationToken(token);

    /// <summary>DIAGNOSTIC for the function-word research: for each token, return its learned signals — relation
    /// DEGREE and cloud CENTRALITY (cosine of its meaning-cloud to the space's global centroid). The hypothesis: a
    /// function word's cloud sits near the centroid (it co-occurs with everything) while a content word — even a
    /// popular one — points somewhere specific. Lets a test SEE which signal actually separates them.</summary>
    /// <summary>DIAGNOSTIC: the NN structure-recogniser's per-token role tags for an input (model-ops gated).</summary>
    public IReadOnlyList<(string Token, int Role, double Confidence)> ProbeRoles(string input)
        => WithModelGate(() => _state.Inference.DiagnoseRoles(input));

    /// <summary>DIAGNOSTIC: does this concept exist, and what relations does it hold (concept, confidence, obs)? For
    /// debugging recall (what is stored vs what is retrieved). Model-ops gated.</summary>
    public (bool Exists, IReadOnlyList<(string Concept, double Confidence, long Obs)> Relations) ProbeRelations(string concept)
        => WithModelGate(() =>
        {
            if (_state.Memory is not Cognition.Platonic.DialecticalSpace ds || !ds.ContainsConcept(concept))
                return (false, (IReadOnlyList<(string, double, long)>)System.Array.Empty<(string, double, long)>());
            IReadOnlyList<(string, double, long)> rels;
            try { rels = ds.GetNeighbors(concept, Cognition.PlatonicNeighborhoodType.Relational, 16, 0.0).Select(n => (n.Concept, n.Confidence, (long)n.ObservationCount)).ToList(); }
            catch { rels = System.Array.Empty<(string, double, long)>(); }
            return (true, rels);
        });

    public IReadOnlyList<(string Token, int Degree, double Centrality, bool Known)> ProbeTokenSignals(IReadOnlyList<string> tokens)
        => WithModelGate(() =>
        {
            var result = new List<(string, int, double, bool)>();
            if (_state.Memory is not Cognition.Platonic.DialecticalSpace ds) return result;
            // Global centroid = mean of every concept's (unit) cloud.
            double[]? centroid = null; var n = 0;
            foreach (var c in ds.ActiveConcepts)
            {
                var v = ds.SemanticVectorOf(c);
                if (v is null) continue;
                centroid ??= new double[v.Length];
                for (var i = 0; i < v.Length && i < centroid.Length; i++) centroid[i] += v[i];
                n++;
            }
            if (centroid is not null) { var nn = 0.0; for (var i = 0; i < centroid.Length; i++) nn += centroid[i] * centroid[i]; nn = System.Math.Sqrt(nn); if (nn > 1e-9) for (var i = 0; i < centroid.Length; i++) centroid[i] /= nn; }
            foreach (var t in tokens)
            {
                var tok = t.ToLowerInvariant();
                var known = ds.ContainsConcept(tok);
                var deg = ds.GetRelationDegree(tok);
                var v = ds.SemanticVectorOf(tok);
                var cen = 0.0;
                if (v is not null && centroid is not null) { for (var i = 0; i < v.Length && i < centroid.Length; i++) cen += v[i] * centroid[i]; }
                result.Add((tok, deg, cen, known));
            }
            return result;
        });

    /// <summary>Turn the CONVERSATIONAL talk route (<c>TryFieldRespond</c>) on/off on the live engine. The gym
    /// enables it while the PersonalityCurriculum is in the mix so the persona is GRADED in-character — it follows
    /// the learned cue→reply CHUNK relation instead of relaxation (which drifts to a cue word, ~8%), so the
    /// FocusedCurriculum sees real progress and reinforces the talk edges rather than thrashing. Scoped to
    /// chat-training sessions; off otherwise so non-chat deployments are byte-identical. Model-ops gated.</summary>
    public void SetConversationalMode(bool on)
        => WithModelGate(() => { _conversationalMode = on; _state.Inference.TalkEnabled = on; return 0; });

    /// <summary>SEED a conversational persona's reply CHUNKS: relate each cue to its WHOLE reply (one composite
    /// concept), so <c>TryFieldRespond</c> retrieves a reply as a CHUNK. The gym does NOT decode-train a persona —
    /// decoding a reply token-by-token only builds stray cue→WORD edges that crowd the chunk out of the top-N
    /// neighbours (measured), and in the conscious field the GRU decoder is bypassed anyway. Seeding once is enough:
    /// the chunk relations are stable space edges (persona cues aren't skill cues, so skill training doesn't touch
    /// them). <paramref name="reps"/> repeats strengthen the edge. Model-ops gated.</summary>
    public void SeedConversationalChunks(IReadOnlyList<(string Cue, string Reply)> pairs, int reps = 8)
        => WithModelGate(() =>
        {
            foreach (var (cue, reply) in pairs)
            {
                if (string.IsNullOrWhiteSpace(cue) || string.IsNullOrWhiteSpace(reply)) continue;
                for (var i = 0; i < Math.Max(1, reps); i++)
                    _state.Memory.FineEditFromExample(new[] { cue }, new[] { reply }, isNegativeExample: false);
            }
            return 0;
        });

    /// <summary>CREDIT ASSIGNMENT on edges: reward the relation edges an answer USED when it was graded CORRECT,
    /// penalise them when WRONG (strengthen / weaken / — via the utility-based pruner — detach). Lets the gym's
    /// graded outcome flow back to the space so a framing-word hub that yields wrong answers decays. Gated by the
    /// model-ops gate (safe vs. REPL/training).</summary>
    public void ReinforceEvidence(IReadOnlyList<GenesisNova.Cognition.PlatonicEvidence> evidence, bool success)
        => WithModelGate(() => { _state.Memory.ReinforceEvidence(evidence, success); return 0; });

    /// <summary>Rung 1 task-outcome disruption (PLATONIC_BACKPROP.md): repel a VALUE-WRONG answer from the query
    /// anchor in the space. Caller must have already established the answer is value-incorrect. Model-ops gated.</summary>
    public void DisruptWrongAnswer(string query, string output)
        => WithModelGate(() => { _state.Inference.DisruptWrongAnswer(query, output); return 0; });

    /// <summary>Rung 2 function gradient (PLATONIC_BACKPROP.md): descend softmax-CE so the query anchor retrieves a
    /// valid task answer (pull target, push confusers). No-op unless FunctionGradientEnabled. Model-ops gated.</summary>
    public void TrainRetrievalToward(string query, IReadOnlyList<string> allowedAnswers)
        => WithModelGate(() => { _state.Inference.TrainRetrievalToward(query, allowedAnswers); return 0; });

    /// <summary>SELF-HEAL a CUE MISROUTE (the missing "learn from a wrong route" signal): a value-wrong probe whose
    /// DecisionPath shows a cue-gated INTENT route fired but whose true-answer STRUCTURE wanted a different route — so
    /// contradict the cue(s) that selected it. General across compare / to-word / to-digit. No-op unless
    /// SelfHealMisroutedCues. Model-ops gated.</summary>
    public void HealMisroutedCue(string query, IReadOnlyList<string> allowedAnswers, string output, string decisionPath)
        => WithModelGate(() => { _state.Inference.HealMisroutedCue(query, allowedAnswers, output, decisionPath); return 0; });

    /// <summary>Live op-head class-balance window [abstain, add, sub, mul, div] (decayed EMA counts). A single
    /// dominant entry signals the head COLLAPSING to one operator — the erosion failure mode. Surfaced here because
    /// these counts were exposed on the model but read by NOTHING; they only mean something during training.</summary>
    public IReadOnlyList<long> OpClassBalance => _state.Model.QueryOpClassCounts;

    public int VocabularySize => WithModelGate(() => _state.Tokenizer.VocabularySize);
    public int HiddenSize => WithModelGate(() => _state.Model.HiddenSize);

    /// <summary>The SGD step size, exposed so a caller can implement a training SCHEDULE (e.g. anneal the LR as
    /// a lesson approaches mastery, the way the autonomous orchestrator does). Reads/writes the live model.</summary>
    public double LearningRate
    {
        get => WithModelGate(() => _state.Model.LearningRate);
        set => WithModelGate(() => { _state.Model.LearningRate = value; return true; });
    }
    public string ConversationBrief => WithModelGate(() => _state.Conversation.BuildContextBrief());
    public bool AutoResumeEnabled => _runtimeConfig.AutoResume;
    public string AutoCheckpointPath => GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);
    public GenesisNovaConfig Config => _runtimeConfig;
    
    public void UpdateConfig(Func<GenesisNovaConfig, GenesisNovaConfig> updater)
    {
        WithModelGate(() =>
        {
            var updated = updater(_runtimeConfig);
            // Update the model's internal config reference
            _state.Model.UpdateConfig(updated);
            return true;
        });
    }
    
    public void EnsureHiddenSize(int targetSize)
    {
        WithModelGate(() =>
        {
            if (targetSize > _state.Model.HiddenSize)
                _state.Model.EnsureHiddenSize(targetSize);
            return true;
        });
    }
    
    /// <summary>
    /// Per-epoch/cycle PLATONIC-SPACE MAINTENANCE — the LIVE eviction pass. Runs the trainer's space maintenance
    /// (relevance-decay discharge + the hard active-concept cap) under the model-ops gate so it never races a
    /// train/predict step. The legacy <see cref="GenesisTrainingOrchestrator"/> calls the trainer directly per epoch;
    /// the modular gym path (which trains via <see cref="TrainAsync"/>, not a held trainer) drives it through here, so
    /// BOTH orchestrators keep the space bounded. No-op when AutoManagePlatonicSpace is off (the SpaceManager gates it).
    /// </summary>
    public SpaceManagementResult MaintainPlatonicSpace()
        => WithModelGate(() => _state.Trainer.ManagePlatonicSpace());

    private T WithModelGate<T>(Func<T> action)
    {
        _modelOpsGate.Wait();
        try
        {
            return action();
        }
        finally
        {
            _modelOpsGate.Release();
        }
    }

    private async Task<T> WithModelGateAsync<T>(Func<Task<T>> action, CancellationToken cancellationToken = default)
    {
        await _modelOpsGate.WaitAsync(cancellationToken);
        try
        {
            return await action();
        }
        finally
        {
            _modelOpsGate.Release();
        }
    }
}
