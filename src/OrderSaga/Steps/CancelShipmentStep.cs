using EvalApp.Solid.Starter.Features.OrderSaga.Services;

namespace EvalApp.Solid.Starter.Features.OrderSaga.Steps;

/// <summary>
/// COMPENSATION step: Cancels shipment if saga fails.
/// Triggered by saga framework if saga fails after shipment creation.
/// Uses ShipmentId stored by ShipStep.
/// SRP: Single responsibility = cancel (undo) shipment.
/// </summary>
public class CancelShipmentStep : AsyncStep<OrderSagaData>
{
    private readonly IShipmentService _shipmentService;

    public CancelShipmentStep(IShipmentService shipmentService)
    {
        _shipmentService = shipmentService;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(data.ShipmentId))
            throw new InvalidOperationException("ShipmentId missing; cannot cancel");

        ct.ThrowIfCancellationRequested();
        var cancelled = await _shipmentService.CancelShipmentAsync(data.ShipmentId, ct);

        if (!cancelled)
            throw new InvalidOperationException($"Failed to cancel shipment {data.ShipmentId}");

        return data with
        {
            State = OrderState.Cancelled,
            ShipmentId = null
        };
    }
}
