namespace EvalApp.Solid.Starter.Orders.Services;

/// <summary>
/// Inventory service interface for reserving and releasing stock.
/// DIP: Steps depend on this abstraction, not concrete implementation.
/// </summary>
public interface IInventoryService
{
    /// <summary>
    /// Reserve items. Returns ReservationId on success, null on failure (out of stock).
    /// </summary>
    Task<string?> ReserveAsync(List<LineItem> items, CancellationToken ct);

    /// <summary>
    /// Release a reservation (compensation for failed order).
    /// </summary>
    Task<bool> ReleaseAsync(string reservationId, CancellationToken ct);
}

/// <summary>
/// Payment service interface for charging and refunding.
/// </summary>
public interface IPaymentService
{
    /// <summary>
    /// Charge customer for order. Returns charge amount on success, null on failure.
    /// </summary>
    Task<decimal?> ChargeAsync(string customerId, decimal amount, CancellationToken ct);

    /// <summary>
    /// Refund a charge (compensation for failed order).
    /// </summary>
    Task<bool> RefundAsync(decimal chargeAmount, CancellationToken ct);
}

/// <summary>
/// Shipment service interface for creating and cancelling shipments.
/// </summary>
public interface IShipmentService
{
    /// <summary>
    /// Create shipment. Returns ShipmentId on success, null on failure.
    /// </summary>
    Task<string?> CreateShipmentAsync(string orderId, List<LineItem> items, CancellationToken ct);

    /// <summary>
    /// Cancel a shipment (compensation for failed order).
    /// </summary>
    Task<bool> CancelShipmentAsync(string shipmentId, CancellationToken ct);
}

