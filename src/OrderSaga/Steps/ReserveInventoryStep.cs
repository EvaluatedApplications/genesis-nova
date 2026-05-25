using EvalApp.Solid.Starter.Features.OrderSaga.Services;

namespace EvalApp.Solid.Starter.Features.OrderSaga.Steps;

/// <summary>
/// Reserves inventory for order items via IInventoryService.
/// AsyncStep: makes external service call (database).
/// SRP: Single responsibility = reserve inventory.
/// </summary>
public class ReserveInventoryStep : AsyncStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;

    public ReserveInventoryStep(IInventoryService inventoryService)
    {
        _inventoryService = inventoryService;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var reservationId = await _inventoryService.ReserveAsync(data.Items, ct);

        if (reservationId == null)
            throw new InvalidOperationException($"Cannot reserve: insufficient inventory for order {data.OrderId}");

        return data with
        {
            State = OrderState.Reserved,
            ReservationId = reservationId
        };
    }
}
