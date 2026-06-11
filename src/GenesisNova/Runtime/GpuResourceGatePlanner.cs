using EvalApp.Consumer;

namespace GenesisNova.Runtime;

/// <summary>
/// Computes data-driven <see cref="TunableConfig"/> bounds for the EvalApp resource
/// gates from measured GPU VRAM (via <see cref="GpuCapacityPlanner"/>) and the
/// authoritative allocation model (<see cref="VramAllocationCalculator"/>).
///
/// === GPU kernel concurrency ===
/// Genesis Nova executes model forward/backward on a single CUDA stream
/// (see GenesisTrainer / GenesisInferenceEngine). True GPU kernel concurrency MUST
/// remain serialized — overlapping kernels on one stream do not run in parallel and
/// concurrent native allocations risk the AccessViolation / CUDA OOM paths handled in
/// GenesisEvalAppRuntime. The GPU ResourceKind therefore stays pinned at 1
/// (<see cref="GpuGate"/> returns Min=Max=Default=1).
///
/// The VRAM planner is instead applied where parallelism is *safe and useful*:
/// the CPU-side candidate-pool prep ForEach (<see cref="CandidatePoolPrepGate"/>).
/// Concurrency there is bounded by both CPU width and the VRAM headroom available
/// for the staged batches each worker prepares, so we never stage more in-flight
/// example pools than the device can subsequently train within safe VRAM bounds.
/// </summary>
public static class GpuResourceGatePlanner
{
    /// <summary>
    /// GPU kernel gate. Serialized to 1 — single CUDA stream (see class remarks).
    /// </summary>
    public static TunableConfig GpuGate() => new(Min: 1, Max: 1, Default: 1);

    /// <summary>
    /// Data-driven concurrency for the CPU-side candidate-pool generation ForEach.
    ///
    /// Inputs:
    ///   cpu        = Environment.ProcessorCount (CPU-bound generation width)
    ///   totalVram  = measured device VRAM (nvidia-smi) or the user allocation budget
    ///   reserveMb  = system reserve from the VRAM allocation model
    ///   perItemMb  = staged-batch VRAM cost per concurrent worker
    ///                (training reserve / batch size from the allocation model)
    ///
    /// Formula:
    ///   usableMb        = max(256, totalVram * targetUtilization - reserveMb)
    ///   vramBoundWorkers = max(1, floor(usableMb / perItemMb))
    ///   max             = clamp(min(cpu, vramBoundWorkers), 1, 2*cpu)
    ///   default         = clamp(ceil(max / 2), 1, max)   // start conservative; tuner scales up
    ///
    /// This keeps the number of concurrently-staged pools below what the GPU can
    /// subsequently consume within the safe VRAM envelope, while letting the adaptive
    /// tuner scale between 1 and the VRAM/CPU ceiling.
    /// </summary>
    public static TunableConfig CandidatePoolPrepGate(int? requestedConcurrency = null)
    {
        var cpu = Math.Max(1, Environment.ProcessorCount);
        var alloc = VramAllocationCalculator.GetOptimal6GbAllocation();

        // Prefer measured device VRAM; fall back to the user allocation budget.
        var totalVram = GpuCapacityPlanner.TryGetNvidiaVramMb(out var measuredTotal, out var measuredFree)
            ? Math.Min(measuredTotal, VramAllocationCalculator.UserTotalAvailableVramMb)
            : VramAllocationCalculator.UserTotalAvailableVramMb;

        // If the device reports free VRAM, never plan above what is actually free.
        var effectiveVram = measuredFree > 0 ? Math.Min(totalVram, measuredFree + alloc.SystemReserveMb) : totalVram;

        var usableMb = Math.Max(256, (int)(effectiveVram * alloc.TargetUtilization) - alloc.SystemReserveMb);

        // Per concurrent worker we budget the staged training reserve for one batch.
        var perItemMb = Math.Max(64, alloc.TrainingReserveMb / Math.Max(1, alloc.BatchSize));
        var vramBoundWorkers = Math.Max(1, usableMb / perItemMb);

        var ceiling = Math.Clamp(Math.Min(cpu, vramBoundWorkers), 1, cpu * 2);

        // Honor an explicit user/request cap (e.g. MaxGenerationConcurrency) as an upper bound.
        if (requestedConcurrency is int req && req > 0)
            ceiling = Math.Clamp(Math.Min(ceiling, req), 1, cpu * 2);

        var max = ceiling;
        var @default = Math.Clamp((int)Math.Ceiling(max / 2.0), 1, max);
        return new TunableConfig(Min: 1, Max: max, Default: @default);
    }
}
