namespace GenesisNova.Runtime;

/// <summary>
/// VRAM allocation and model sizing for user hardware.
///
/// User machine: RTX A3000 Laptop (4GB) + shared RAM
/// Goal: a model with safe training headroom on 6GB total VRAM.
///
/// Provides <see cref="GetOptimal6GbAllocation"/> (training reserve / batch / utilization budget consumed by
/// <see cref="GpuResourceGatePlanner"/>) and the <see cref="UserTotalAvailableVramMb"/> ceiling.
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
    // Sizing inputs used by GetOptimal6GbAllocation to back-calculate the model hidden size and batch budget.

    // Model parameter estimation: ~4 bytes per parameter
    // For Genesis Nova: emb(v*h) + out(h*v) + route(h*r) + biases
    // Bytes ~= 4 * (2*vocab + h*r + 2*vocab + h + r)
    private const int BytesPerParameter = 4;
    private const int VocabularySize = 8192;  // Assumed for calculation
    private const int RouteCount = 3;         // Actual from GenesisNeuralModel.NumRoutes (neural / platonic-direct / platonic-assisted)
    
    // === TIER-BASED TRAINING CAPS (from ResolveTrainingHiddenCap) ===
    public static int GetTrainingHiddenCap(int vramMb)
    {
        return vramMb switch
        {
            <= 4096 => 512,
            <= 6144 => 512,
            <= 8192 => 768,
            _ => 1024
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
        const int modelTargetMb = 512;    // 0.5GB aggressive model target
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
            TrainingReserveMb = trainingOverheadMb,
            SystemReserveMb = systemReserveMb,
            BatchSize = batchSize,
            TargetUtilization = 0.75  // Conservative: leave headroom
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

}

/// <summary>
/// Result from VRAM allocation calculation. Only the fields consumed by
/// <see cref="GpuResourceGatePlanner"/> are retained.
/// </summary>
public record VramAllocation
{
    public int TrainingReserveMb { get; init; }
    public int SystemReserveMb { get; init; }
    public int BatchSize { get; init; }
    public double TargetUtilization { get; init; }
}
