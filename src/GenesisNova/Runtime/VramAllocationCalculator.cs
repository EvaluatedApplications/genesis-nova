namespace GenesisNova.Runtime;

/// <summary>
/// Single source of truth for VRAM allocation and model sizing on user hardware.
/// 
/// User machine: RTX A3000 Laptop (4GB) + shared RAM
/// Goal: 3GB+ model with safe training headroom on 6GB total VRAM.
/// 
/// This calculator unifies:
/// - Model hidden size allocation
/// - Batch size recommendations
/// - Training reserve overhead
/// - GPU capacity planning
/// </summary>
public static class VramAllocationCalculator
{
    // === HARDWARE PROFILE ===
    // RTX A3000 Laptop GPU with 4GB VRAM + 31.7GB shared RAM
    // User expects to use up to 6GB VRAM for model+training
    public const int UserTotalAvailableVramMb = 6144;  // 6GB available for Genesis Nova
    public const int UserGpuNativeVramMb = 4096;       // RTX A3000 = 4GB native
    public const int UserSharedRamAvailableMb = 2048;  // Conservative: leave 29.7GB for OS/apps

    // === CAPACITY PLANNING CONSTANTS ===
    // These replace the scattered defaults from MainWindow, GpuCapacityPlanner, GenesisCli
    
    // Model parameter estimation: ~4 bytes per parameter
    // For Genesis Nova: emb(v*h) + out(h*v) + route(h*r) + biases
    // Bytes ~= 4 * (2*vocab + h*r + 2*vocab + h + r)
    private const int BytesPerParameter = 4;
    private const int VocabularySize = 8192;  // Assumed for calculation
    private const int RouteCount = 2;         // Actual from GenesisNeuralModel.NumRoutes
    
    // === TIER-BASED TRAINING CAPS (from ResolveTrainingHiddenCap) ===
    public static int GetTrainingHiddenCap(int vramMb)
    {
        return vramMb switch
        {
            <= 4096 => 2048,
            <= 6144 => 3072,
            <= 8192 => 4096,
            _ => 6144
        };
    }

    // === PRIMARY: UNIFIED ALLOCATION FOR 6GB VRAM ===
    /// <summary>
    /// Calculate optimal model size and training parameters for user's 6GB VRAM.
    /// 
    /// Allocation breakdown:
    /// - Model weights: 3GB (target minimum)
    /// - Training overhead: ~1.5GB (batch buffers, gradients, optimizer state)
    /// - System reserve: 1.5GB (kernel, system ops)
    /// Total: ~6GB
    /// </summary>
    public static VramAllocation GetOptimal6GbAllocation()
    {
        const int totalVramMb = 6144;
        const int modelTargetMb = 3072;    // 3GB for model
        const int trainingOverheadMb = 1536;  // 1.5GB for gradients + optimizer + batch
        const int systemReserveMb = 1536;     // 1.5GB system reserve
        
        // Back-calculate hidden size from 3GB model budget
        // Model params ~= 4 * (2*vocab + h*r)
        // Model bytes ~= 3GB = 3,221,225,472 bytes
        // Solving: 3,221,225,472 = 4 * (2*8192 + h*2) * 4
        // h ≈ 398,662,144 / 8 ≈ 49,832,768... That's wrong.
        // Actually model bytes = 4 * h * (2*vocab + r) + emb overhead
        // Simplified: ~4 bytes per parameter, estimating ~750M params = 3GB
        // So hidden size that fits in 3GB with vocab=8192, routes=2:
        var hiddenSize = EstimateHiddenForTargetBytes(
            targetBytes: (long)modelTargetMb * 1024 * 1024,
            vocab: VocabularySize,
            routeCount: RouteCount,
            overheadFactor: 1.35
        );
        
        // Clamp to training cap for 6GB
        var trainingCap = GetTrainingHiddenCap(totalVramMb);
        hiddenSize = Math.Min(hiddenSize, trainingCap);
        
        // Re-estimate actual model size with this hidden size
        var actualModelBytes = EstimateModelBytes(hiddenSize, VocabularySize, RouteCount, overheadFactor: 1.35);
        var actualModelMb = (int)Math.Ceiling(actualModelBytes / (1024.0 * 1024.0));
        
        // Batch size: use remaining training headroom
        var remainingMb = trainingOverheadMb;
        var batchSize = EstimateBatchSizeFromBudget(
            remainingMb,
            hiddenSize,
            sequenceLength: 128,
            vocabSize: VocabularySize
        );
        
        return new VramAllocation
        {
            TotalVramMb = totalVramMb,
            HiddenSize = hiddenSize,
            EstimatedModelMb = actualModelMb,
            TrainingReserveMb = trainingOverheadMb,
            SystemReserveMb = systemReserveMb,
            BatchSize = batchSize,
            TargetUtilization = 0.75,  // Conservative: leave headroom
            ReserveVramMb = systemReserveMb,
            MaxTrainingTokensPerExample = 128,
            EffectiveVramForTrainingMb = totalVramMb - systemReserveMb
        };
    }

    // === SUPPORTING CALCULATIONS ===
    
    /// <summary>
    /// Estimate hidden size that fits in a target byte budget.
    /// Model bytes = 4 * h * (2*vocab + r) * (1 + overhead)
    /// </summary>
    private static int EstimateHiddenForTargetBytes(
        long targetBytes,
        int vocab,
        int routeCount,
        double overheadFactor = 1.35)
    {
        var costPerHidden = BytesPerParameter * (2.0 * vocab + routeCount) * overheadFactor;
        var hidden = (int)Math.Floor(targetBytes / costPerHidden);
        return Math.Clamp(hidden, 48, 8192);
    }
    
    /// <summary>
    /// Estimate model bytes for a given hidden size.
    /// </summary>
    private static long EstimateModelBytes(
        int hiddenSize,
        int vocab,
        int routeCount,
        double overheadFactor = 1.35)
    {
        var costPerHidden = BytesPerParameter * (2.0 * vocab + routeCount);
        return (long)Math.Ceiling(costPerHidden * hiddenSize * overheadFactor);
    }
    
    /// <summary>
    /// Estimate batch size from available VRAM budget for training buffers.
    /// Batch memory ≈ 2 * hidden * sequenceLength * 4 bytes per example
    /// (forward activation + gradient buffer)
    /// </summary>
    private static int EstimateBatchSizeFromBudget(
        int budgetMb,
        int hiddenSize,
        int sequenceLength,
        int vocabSize)
    {
        var budgetBytes = budgetMb * 1024.0 * 1024.0;
        var bytesPerExample = 2.0 * hiddenSize * sequenceLength * 4;
        var batchSize = (int)Math.Floor(budgetBytes / bytesPerExample);
        return Math.Clamp(batchSize, 1, 64);
    }

    // === INFERENCE-ONLY SIZING (minimal overhead) ===
    public static int GetInferenceOnlyHiddenSize()
    {
        // For inference: minimal overhead, can use more of the 6GB
        // Bytes = 4 * h * (2*vocab + r) with ~1.1 overhead
        const long inferenceTargetBytes = 5816361984;  // 5.5 * 1024 * 1024 * 1024
        return EstimateHiddenForTargetBytes(
            targetBytes: inferenceTargetBytes,
            vocab: VocabularySize,
            routeCount: RouteCount,
            overheadFactor: 1.1
        );
    }

    // === BACKWARD COMPATIBILITY: Bridge to existing GpuCapacityPlanner ===
    /// <summary>
    /// Calculate recommended parameters for existing code paths.
    /// This bridges VramAllocationCalculator to GpuCapacityPlanner.EstimateTrainingBatchSize().
    /// </summary>
    public static (int hidden, int batch, int reserve) GetRecommendedTrainingParams()
    {
        var alloc = GetOptimal6GbAllocation();
        return (alloc.HiddenSize, alloc.BatchSize, alloc.ReserveVramMb);
    }

    /// <summary>
    /// UI default for training tab initial values.
    /// </summary>
    public static class UiDefaults
    {
        public static int HiddenSize => GetOptimal6GbAllocation().HiddenSize;
        public static int BatchSize => GetOptimal6GbAllocation().BatchSize;
        public static int Epochs => 3;
        public static double TargetVramUtilization => 0.75;
        public static int ReserveVramMb => 1536;
    }

    /// <summary>
    /// Autonomous training defaults (replaces scattered logic in GetAutonomousResourceDefaults).
    /// </summary>
    public static class AutonomousDefaults
    {
        // Initial sample/train counts scaled for 6GB GPU
        public static int InitialSampleCount => 24;
        public static int InitialTrainCount => 24;
        public static int InitialEpochs => 3;
        public static int MaxSampleCount => 96;
        public static int MaxTrainCount => 96;
        public static int MaxDifficulty => 50;
        public static int RoundBudget => 256;
    }
}

/// <summary>
/// Result from VRAM allocation calculation.
/// Single authoritative source for all allocation parameters.
/// </summary>
public record VramAllocation
{
    public int TotalVramMb { get; init; }
    public int HiddenSize { get; init; }
    public int EstimatedModelMb { get; init; }
    public int TrainingReserveMb { get; init; }
    public int SystemReserveMb { get; init; }
    public int BatchSize { get; init; }
    public double TargetUtilization { get; init; }
    public int ReserveVramMb { get; init; }
    public int MaxTrainingTokensPerExample { get; init; }
    public int EffectiveVramForTrainingMb { get; init; }

    public override string ToString()
    {
        return $"""
            VRAM Allocation for {TotalVramMb}MB GPU:
              Model (Hidden={HiddenSize}): ~{EstimatedModelMb}MB
              Training Reserve: {TrainingReserveMb}MB (Batch={BatchSize})
              System Reserve: {SystemReserveMb}MB
              ---
              Total Effective: {EffectiveVramForTrainingMb}MB @ {TargetUtilization:P0} util
            """;
    }
}
