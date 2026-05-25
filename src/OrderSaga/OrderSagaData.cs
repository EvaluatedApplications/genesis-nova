namespace EvalApp.Solid.Starter.Features.OrderSaga;

/// <summary>
/// Saga data flowing through order processing pipeline.
/// Captures state progression: Pending → Reserved → Charged → Shipped.
/// Compensation stores (ReservationId, ChargeAmount, ShipmentId) for rollback.
/// </summary>
public record OrderSagaData(
    string OrderId,
    List<LineItem> Items,
    string CustomerId,
    OrderState State = OrderState.Pending,
    string? ReservationId = null,
    decimal? ChargeAmount = null,
    string? ShipmentId = null,
    string? FailureReason = null);

public enum OrderState
{
    Pending,
    Reserved,
    Charged,
    Shipped,
    Cancelled
}

public record LineItem(string Sku, int Qty);
