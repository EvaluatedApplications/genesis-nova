using EvalApp.Solid.Starter.Orders;
using EvalApp.Solid.Starter.Orders.Services;

namespace EvalApp.Solid.Starter.Tests.Orders.Shared;

public class MockInventoryService : IInventoryService
{
    private readonly bool _shouldFail;
    private readonly Dictionary<string, List<LineItem>> _reservations = new();

    public MockInventoryService(bool shouldFail = false)
    {
        _shouldFail = shouldFail;
    }

    public Task<string?> ReserveAsync(List<LineItem> items, CancellationToken ct)
    {
        if (_shouldFail)
            return Task.FromResult<string?>(null);

        var reservationId = $"RES-{Guid.NewGuid():N}";
        _reservations[reservationId] = items;
        return Task.FromResult<string?>(reservationId);
    }

    public Task<bool> ReleaseAsync(string reservationId, CancellationToken ct)
    {
        if (!_reservations.ContainsKey(reservationId))
            return Task.FromResult(false);

        _reservations.Remove(reservationId);
        return Task.FromResult(true);
    }

    public Dictionary<string, List<LineItem>> GetActiveReservations() => _reservations;
}

public class MockPaymentService : IPaymentService
{
    private readonly bool _shouldFail;
    private readonly decimal _chargeAmount;
    private readonly Dictionary<string, decimal> _charges = new();

    public MockPaymentService(decimal chargeAmount = 100m, bool shouldFail = false)
    {
        _chargeAmount = chargeAmount;
        _shouldFail = shouldFail;
    }

    public Task<decimal?> ChargeAsync(string customerId, decimal amount, CancellationToken ct)
    {
        if (_shouldFail)
            return Task.FromResult<decimal?>(null);

        var chargeId = $"CHG-{Guid.NewGuid():N}";
        _charges[chargeId] = amount;
        return Task.FromResult<decimal?>(amount);
    }

    public Task<bool> RefundAsync(decimal chargeAmount, CancellationToken ct)
    {
        var chargeKey = _charges.FirstOrDefault(x => x.Value == chargeAmount).Key;
        if (chargeKey == null)
            return Task.FromResult(false);

        _charges.Remove(chargeKey);
        return Task.FromResult(true);
    }

    public Dictionary<string, decimal> GetActiveCharges() => _charges;
}

public class MockShipmentService : IShipmentService
{
    private readonly bool _shouldFail;
    private readonly Dictionary<string, string> _shipments = new();

    public MockShipmentService(bool shouldFail = false)
    {
        _shouldFail = shouldFail;
    }

    public Task<string?> CreateShipmentAsync(string orderId, List<LineItem> items, CancellationToken ct)
    {
        if (_shouldFail)
            return Task.FromResult<string?>(null);

        var shipmentId = $"SHIP-{Guid.NewGuid():N}";
        _shipments[shipmentId] = orderId;
        return Task.FromResult<string?>(shipmentId);
    }

    public Task<bool> CancelShipmentAsync(string shipmentId, CancellationToken ct)
    {
        if (!_shipments.ContainsKey(shipmentId))
            return Task.FromResult(false);

        _shipments.Remove(shipmentId);
        return Task.FromResult(true);
    }

    public Dictionary<string, string> GetActiveShipments() => _shipments;
}



