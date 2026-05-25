using EvalApp.Solid.Starter.Features.OrderSaga.Services;

namespace EvalApp.Solid.Starter.Features.OrderSaga.Steps;

/// <summary>
/// COMPENSATION step: Releases reserved inventory if saga fails.
/// Triggered by saga framework if any subsequent step fails.
/// Uses ReservationId stored by ReserveInventoryStep.
/// SRP: Single responsibility = release (undo) inventory.
/// </summary>
public class ReleaseReservationStep : AsyncStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;

    public ReleaseReservationStep(IInventoryService inventoryService)
    {
        _inventoryService = inventoryService;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(data.ReservationId))
            throw new InvalidOperationException("ReservationId missing; cannot release");

        ct.ThrowIfCancellationRequested();
        var released = await _inventoryService.ReleaseAsync(data.ReservationId, ct);

        if (!released)
            throw new InvalidOperationException($"Failed to release reservation {data.ReservationId}");

        return data with
        {
            State = OrderState.Cancelled,
            ReservationId = null
        };
    }
}
