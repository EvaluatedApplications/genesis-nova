using EvalApp.Solid.Starter.Features.OrderSaga;

namespace EvalApp.Solid.Starter.Tests.Features.OrderSaga.Shared;

public static class OrderSagaTestData
{
    public static LineItem CreateLineItem(string sku = "SKU-1", int qty = 1)
        => new LineItem(sku, qty);

    public static OrderSagaData CreateOrder(
        string orderId = "ORD-001",
        List<LineItem>? items = null,
        string customerId = "CUST-001",
        OrderState state = OrderState.Pending,
        string? reservationId = null,
        decimal? chargeAmount = null,
        string? shipmentId = null)
    {
        items ??= new List<LineItem> { CreateLineItem() };
        return new OrderSagaData(
            OrderId: orderId,
            Items: items,
            CustomerId: customerId,
            State: state,
            ReservationId: reservationId,
            ChargeAmount: chargeAmount,
            ShipmentId: shipmentId);
    }

    public static OrderSagaData CreateOrderWithMultipleItems(
        int itemCount = 3,
        string orderId = "ORD-002")
    {
        var items = Enumerable.Range(1, itemCount)
            .Select(i => CreateLineItem($"SKU-{i}", i))
            .ToList();
        return CreateOrder(orderId, items);
    }
}
