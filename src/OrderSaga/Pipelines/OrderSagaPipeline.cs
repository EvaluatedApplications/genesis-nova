using EvalApp.Solid.Starter.Features.OrderSaga.Services;
using EvalApp.Solid.Starter.Features.OrderSaga.Steps;

namespace EvalApp.Solid.Starter.Features.OrderSaga.Pipelines;

/// <summary>
/// OrderSaga Pipeline — Demonstrates SOLID principles via distributed transaction handling.
/// 
/// Saga Topology:
///   BeginSagaStep
///     → ReserveInventoryStep [compensates with: ReleaseReservationStep]
///     → ChargePaymentStep [compensates with: RefundPaymentStep]
///     → ShipStep [compensates with: CancelShipmentStep]
///     → EndSagaStep
/// 
/// Compensation Semantics:
///   - If ReserveInventoryStep fails, nothing to compensate
///   - If ChargePaymentStep fails, ReserveInventoryStep is compensated (release reservation)
///   - If ShipStep fails, ChargePaymentStep and ReserveInventoryStep are compensated in reverse order
///   - If any compensation fails, order is marked as orphaned/manual-review required
/// 
/// SOLID Benefits:
///   - SRP: Each step (forward + compensation) has single responsibility
///   - OCP: New saga steps added without changing pipeline topology
///   - DIP: Depend on interfaces (IInventoryService, IPaymentService, IShipmentService)
///   - LSP: All steps follow same contract (AsyncStep<T>)
/// </summary>
public static class OrderSagaPipeline
{
    public static ICompiledPipeline<OrderSagaData> Build(
        IInventoryService inventoryService,
        IPaymentService paymentService,
        IShipmentService shipmentService,
        decimal orderAmount)
    {
        ICompiledPipeline<OrderSagaData> pipeline = null!;

        Eval.App("OrderSaga")
            .DefineDomain("Fulfillment")
                .DefineTask<OrderSagaData>("ProcessOrder")
                    .AddStep("BeginSaga", new BeginSagaStep())
                    .AddStep("ReserveInventory", new ReserveInventoryStep(inventoryService))
                    .AddStep("ChargePayment", new ChargePaymentStep(paymentService, orderAmount))
                    .AddStep("Ship", new ShipStep(shipmentService))
                    .AddStep("EndSaga", new EndSagaStep())
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}
