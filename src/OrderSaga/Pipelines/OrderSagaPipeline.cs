using EvalApp.Solid.Starter.Features.OrderSaga.Services;
using EvalApp.Solid.Starter.Features.OrderSaga.Steps;
using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Features.OrderSaga.Pipelines;

/// <summary>
/// OrderSaga Pipeline — Demonstrates fail-stop distributed transaction pattern.
/// 
/// Saga Topology (Simplified):
///   ReserveInventoryStep (Database I/O)
///     → ChargePaymentStep (Network I/O)
///     → ShipStep (Network I/O)
/// 
/// Failure Behavior:
/// - If any step fails, pipeline stops immediately
/// - Data record captures IDs (ReservationId, ChargeAmount, ShipmentId) for reference
/// - Caller can inspect the partial failure data to manually compensate
/// 
/// SOLID Benefits:
///   - SRP: Each step has single responsibility (reserve, charge, ship)
///   - OCP: New saga steps added without changing topology
///   - DIP: Depend on interfaces (IInventoryService, IPaymentService, IShipmentService)
///   - LSP: All steps follow same contract (AsyncStep<T>)
/// </summary>
public static class OrderSagaPipeline
{
    /// <summary>
    /// Build saga pipeline with resource gating and adaptive tuning for external service calls.
    /// </summary>
    public static ICompiledPipeline<OrderSagaData> Build(
        IInventoryService inventoryService,
        IPaymentService paymentService,
        IShipmentService shipmentService,
        decimal orderAmount)
    {
        ICompiledPipeline<OrderSagaData> pipeline = null!;

        Eval.App("OrderSaga")
            .WithContext(NullGlobalContext.Instance)
            .DefineDomain("Fulfillment", NullGlobalContext.Instance)
                .DefineTask<OrderSagaData>("ProcessOrder")
                    // Sequential saga steps demonstrating distributed transaction pattern
                    .AddStep(
                        "ReserveInventory",
                        async (data, ct) =>
                            await new ReserveInventoryStep(inventoryService).ExecuteAsync(data, ct))
                    .AddStep(
                        "ChargePayment",
                        async (data, ct) =>
                            await new ChargePaymentStep(paymentService, orderAmount).ExecuteAsync(data, ct))
                    .AddStep(
                        "Ship",
                        async (data, ct) =>
                            await new ShipStep(shipmentService).ExecuteAsync(data, ct))
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}
