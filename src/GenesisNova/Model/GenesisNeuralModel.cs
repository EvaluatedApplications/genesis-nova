using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
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
    private const double RouteLossWeight = 0.5; // C5: raised 0.25→0.5 so the platonic route head pulls harder on the
                                                // shared GRU vs the token LM (weight 1.0) — the 4:1 token dominance was
                                                // the structural cause of router erosion (a+b drifting to neural).
    private readonly object _routeClassBalanceLock = new();
    private readonly double[] _routeClassCounts = { 1.0, 1.0, 1.0 };
    private TorchSharp.Modules.Parameter? _routeWT;
    private TorchSharp.Modules.Parameter? _routeB;
    // Optional PERCEPTION weight for ROUTING (SPACE_AWARE_GRU.md §I): maps a TARGET-AGNOSTIC space-state vector
    // ("can the space answer this query?" — nearest-neighbour confidence/degree) into the route logits, so the
    // GRU LEARNS to route platonic-vs-neural from PERCEIVED retrievability instead of tokens alone. Runtime-only
    // (reinitialised, not persisted); gated by PerceptionRouting (default off → the route head is unchanged).
    private TorchSharp.Modules.Parameter? _routePerceptionW; // [EditPerceptionDim, RouteCount]
    public bool PerceptionRouting { get; set; } = true;

    // RELIABILITY-WEIGHTED ROUTING: when on, the route perception's spare channel carries the EARNED UCB
    // reliability of the model's best learned transform, so the route head learns to trust the function/platonic
    // route when transforms are proven and distrust it when they're noisy ("bubble up successes"). The channel
    // rides the existing _routePerceptionW (no new params); default ON. Off → the channel is fed 0 (unchanged).
    public bool TransformReliabilityRouting { get; set; } = true;

    // Platonic-QUERY construction heads: the GRU learns to CONSTRUCT the platonic query itself
    // (which face operation, which input tokens are operands) instead of a hardcoded grammar
    // extracting them. Op vocabulary is face-derived ({poly,log} × {+,−} + abstain), see
    // GenesisQueryLabel. Operation head reads the final shared-GRU hidden ([hidden, QueryOpCount]);
    // the operand head scores EVERY per-token hidden ([hidden, 1], sigmoid → "is this an operand").
    // Lazily initialized on first supervised query label (like the route head); trained by autograd
    // CE/BCE losses folded into the shared backward pass, so query construction co-shapes the
    // encoder. Not yet persisted in ModelSnapshot — heads retrain after Import (graceful, like a
    // fresh route head).
    public const int QueryOpCount = 5; // 0=none/abstain, 1=add, 2=sub, 3=mul, 4=div
    private const double QueryLossWeight = 0.5; // C5: raised 0.25→0.5 (see RouteLossWeight) — op classification pulls harder.
    private TorchSharp.Modules.Parameter? _queryOpWT;      // [hidden, QueryOpCount]
    private TorchSharp.Modules.Parameter? _queryOpB;       // [QueryOpCount]
    // Inverse-frequency CLASS BALANCE for the OP classifier — mirrors the route head (ObserveRouteClassWeight).
    // WITHOUT it a skewed op mix (lots of subtract, mul examples vanishing when its transform retires) collapses
    // the op head to the MAJORITY class — observed as every add/mul computing a−b. Seeded {1,…} (Laplace); only
    // the op CE is weighted (the per-token operand BCE is untouched). Toggle is diagnostic, defaults true.
    private readonly object _queryOpClassBalanceLock = new();
    private readonly double[] _queryOpClassCounts = System.Linq.Enumerable.Repeat(1.0, QueryOpCount).ToArray();
    internal bool QueryOpClassBalanceEnabled { get; set; } = true;
    // SPACE-AWARE op head (SPACE_AWARE_GRU.md §A): the op classifier conditions on a perception vector of the
    // query anchor's region, so it READS the space rather than choosing from tokens alone. Runtime-only
    // (reinitialised, not persisted); trained by TrainQueryOpPerception; default ON. No perception / flag off
    // → the channel is fed 0 (head unchanged), so non-relational (numeric) inputs degrade gracefully.
    private TorchSharp.Modules.Parameter? _queryOpPerceptionW; // [EditPerceptionDim, QueryOpCount]
    public bool PerceptionQuery { get; set; } = true;
    private TorchSharp.Modules.Parameter? _queryOperandWT; // [hidden, 1]
    private TorchSharp.Modules.Parameter? _queryOperandB;  // [1]

    // PER-TOKEN ROLE head — the NN as a general STRUCTURE RECOGNISER (nova-nn-recognizer-space-structural): it tags
    // each input token's grammatical role from the GRU's raw per-token hidden, the same shape as the operand head but
    // multi-class. Trained by SELF-SUPERVISED labels (the assert/recall alignment), it learns to recognise subject /
    // value / query and GENERALISES (nonce copulas, new phrasings) — so the platonic space can stay purely structural
    // (it stores the binding the NN extracts). 0=NONE/filler, 1=SUBJECT (the key), 2=VALUE (the asserted thing),
    // 3=QUERY (a retrieval cue). Reads the raw [hidden] per-token state, so it is hidden-dependent (resize nulls it).
    public const int RoleCount = 4;
    private TorchSharp.Modules.Parameter? _roleWT; // [hidden, RoleCount]
    private TorchSharp.Modules.Parameter? _roleB;  // [RoleCount]

    // PLAN head — the learned composer's SHAPE selector. From the input it classifies which block-
    // composition to assemble + run on the substrate (the op/operand heads supply the arguments; this
    // supplies the shape). Lazily initialised + autograd-trained (CE) exactly like the query op head.
    // Increment 1 wires PlanKind.Predicate (Compare→Branch); more shapes extend PlanKindCount.
    // 0=none, 1=arithmetic, 2=predicate, 3=retrieval, 4=arithmetic→word, 5=fold-sum, 6=fold-product,
    // 7=seq (Concatenate-Composition: Literal scaffold + Compute → "the answer is N"),
    // 8=expression-chain (MULTI-operator expression "2 x 7 + 3" → 17: each operator classified from context
    //   by the op head, evaluated with precedence by chaining compute-elements on the substrate).
    // Each shape the GRU can SELECT; the substrate executes it (R2 compose / homomorphism / relations).
    public const int PlanKindCount = 9;
    private const double PlanLossWeight = 0.5; // C5: raised 0.25→0.5 (see RouteLossWeight) — shape selection pulls harder.
    private TorchSharp.Modules.Parameter? _planWT; // [hidden, PlanKindCount]
    private TorchSharp.Modules.Parameter? _planB;  // [PlanKindCount]
    // Inverse-frequency CLASS BALANCE for the PLAN/shape head — same rationale as the op head above: a skewed
    // shape mix would collapse it to the majority shape. Seeded {1,…} (Laplace); diagnostic toggle defaults true.
    private readonly object _planClassBalanceLock = new();
    private readonly double[] _planClassCounts = System.Linq.Enumerable.Repeat(1.0, PlanKindCount).ToArray();
    internal bool PlanClassBalanceEnabled { get; set; } = true;
    // SPACE-AWARE plan head (SPACE_AWARE_GRU.md §A): shape selection conditions on the anchor's perception
    // vector, so the composer READS the space before picking a shape. Runtime-only; trained by
    // TrainPlanPerception; default ON. No perception / flag off → fed 0 (head unchanged).
    private TorchSharp.Modules.Parameter? _planPerceptionW; // [EditPerceptionDim, PlanKindCount]
    public bool PerceptionPlan { get; set; } = true;

    // SHARED REASONING TRUNK: a single nonlinear projection of the GRU hidden that the three DECISION heads
    // (route, query-op, plan) read INSTEAD of the raw hidden. A linear head can only pull linearly-separable
    // decisions out of the representation; the trunk gives the selector genuine nonlinear capacity right where
    // the hard which-route / which-shape / which-op calls are made. The operand head, edit head, and token
    // decoder are UNCHANGED (they still read the raw per-token / final hidden). Autograd-trained with the heads.
    private const int ReasoningTrunkDim = 512;
    private TorchSharp.Modules.Parameter? _trunkW; // [hidden, ReasoningTrunkDim]
    private TorchSharp.Modules.Parameter? _trunkB; // [ReasoningTrunkDim]

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
    // Optional PERCEPTION weight: maps a small space-perception vector (rank-of-target, distractor-winning, …)
    // into the edit-head logit, so the head can LEARN a state-dependent (read-before-write) edit policy instead
    // of conditioning on the input tokens alone. Runtime-only (reinitialised, NOT persisted); active only when a
    // caller passes a perception vector (the space-aware experiments / opt-in trainer flag). Default path unused.
    public const int EditPerceptionDim = 6;
    private TorchSharp.Modules.Parameter? _editPerceptionW; // [EditPerceptionDim, 1]

    private double _currentLearningRate;

    // EMA decay for the class-balance counts (~1/(1−d) ≈ 333-observation window). The counts must track RECENT
    // class frequency, NOT the lifetime average: a FOCUSED curriculum floods a shared head with ONE class for a
    // whole focus turn (~hundreds of same-op examples), and cumulative counts (≈ balanced lifetime) miss that and
    // let the head COLLAPSE to the focused class (op-classifier went add→sub→multiply as the focus rotated). A
    // decaying window makes the inverse-frequency weight react to the current flood — down-weighting it, up-
    // weighting the starved classes — so the per-batch gradient stays roughly balanced even under heavy focus.
    private const double ClassBalanceDecay = 0.997;

    /// <summary>
    /// DIAGNOSTIC TOGGLE — defaults true (production behaviour). When false, <see
    /// cref="ObserveRouteClassWeight"/> returns 1.0 so the route loss is UNWEIGHTED. Used to test
    /// whether inverse-frequency class balancing is causally responsible for router-confidence
    /// erosion under broad training. Counts are still tracked. Not a production switch.
    /// </summary>
    internal bool RouteClassBalanceEnabled { get; set; } = true;

    public GenesisNeuralModel(GenesisNovaConfig config)
    {
        _config = config;
        _hiddenSize = config.HiddenSize;
        _currentLearningRate = config.LearningRate;

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
    double[]? GruBhh = null,
    // Platonic-query construction heads (op classifier [h,QueryOpCount] + operand scorer [h,1]) and the composer
    // PLAN head [h,PlanKindCount]. OPTIONAL, appended so existing call sites compile; null on pre-head checkpoints
    // (then lazily reinitialised). Persisted so a loaded model keeps these TRAINED heads instead of resetting
    // them each reload (the same drop-on-load bug previously fixed for the GRU/edit heads).
    double[,]? QueryOpWeights = null,
    double[]? QueryOpBias = null,
    double[,]? QueryOperandWeights = null,
    double[]? QueryOperandBias = null,
    double[,]? PlanWeights = null,
    double[]? PlanBias = null,
    // SHARED REASONING TRUNK weights [hidden, ReasoningTrunkDim] + bias [ReasoningTrunkDim]. The route/op/plan
    // heads read this, so it MUST persist or a loaded model routes differently (random trunk × trained head).
    double[,]? TrunkWeights = null,
    double[]? TrunkBias = null);
