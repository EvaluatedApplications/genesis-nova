using EvalApp.Solid.Starter.Orders.Services;

namespace EvalApp.Solid.Starter.Orders.Steps;

/// <summary>
/// Creates shipment via IShipmentService.
/// AsyncStep: makes external service call (shipping API).
/// SRP: Single responsibility = create shipment.
/// Stores ShipmentId for compensation (cancellation).
/// </summary>
public class ShipStep : AsyncStep<OrderSagaData>
{
    private readonly IShipmentService _shipmentService;

    public ShipStep(IShipmentService shipmentService)
    {
        _shipmentService = shipmentService;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var shipmentId = await _shipmentService.CreateShipmentAsync(data.OrderId, data.Items, ct);

        if (shipmentId == null)
            throw new InvalidOperationException($"Failed to create shipment for order {data.OrderId}");

        return data with
        {
            State = OrderState.Shipped,
            ShipmentId = shipmentId
        };
    }
}

