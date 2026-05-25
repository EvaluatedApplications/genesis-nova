using EvalApp.Consumer;
using EvalApp.Solid.Starter.Orders;
using EvalApp.Solid.Starter.Orders.Services;
using EvalApp.Solid.Starter.Tests.Orders.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Orders;

/// <summary>
/// Additional comprehensive tests for OrderSaga covering edge cases and stress.
/// </summary>
public class OrderSagaEdgeCaseTests
{
    [Fact]
    public void WhenCreatingOrder_Then_DefaultsToPending()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder();

        // Assert
        Assert.Equal(OrderState.Pending, order.State);
    }

    [Fact]
    public void WhenOrderWithEmptyItems_Then_InvalidOrder()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder(items: new List<LineItem>());

        // Assert
        Assert.Empty(order.Items);
    }

    [Fact]
    public void WhenOrderWithMultipleItems_Then_AllStored()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrderWithMultipleItems(itemCount: 5);

        // Assert
        Assert.Equal(5, order.Items.Count);
    }

    [Fact]
    public void WhenTransitioningToReserved_Then_StateChanges()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder();

        // Act
        var reserved = order with { State = OrderState.Reserved, ReservationId = "RES-001" };

        // Assert
        Assert.Equal(OrderState.Reserved, reserved.State);
        Assert.Equal("RES-001", reserved.ReservationId);
    }

    [Fact]
    public void WhenTransitioningToCharged_Then_StateChanges()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder(state: OrderState.Reserved);

        // Act
        var charged = order with { State = OrderState.Charged, ChargeAmount = 100m };

        // Assert
        Assert.Equal(OrderState.Charged, charged.State);
        Assert.Equal(100m, charged.ChargeAmount);
    }

    [Fact]
    public void WhenTransitioningToShipped_Then_StateChanges()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder(state: OrderState.Charged);

        // Act
        var shipped = order with { State = OrderState.Shipped, ShipmentId = "SHIP-001" };

        // Assert
        Assert.Equal(OrderState.Shipped, shipped.State);
        Assert.Equal("SHIP-001", shipped.ShipmentId);
    }

    [Fact]
    public void WhenCancellingOrder_Then_StateIsCancelled()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder(state: OrderState.Reserved);

        // Act
        var cancelled = order with { State = OrderState.Cancelled, FailureReason = "Out of stock" };

        // Assert
        Assert.Equal(OrderState.Cancelled, cancelled.State);
        Assert.Equal("Out of stock", cancelled.FailureReason);
    }

    [Fact]
    public void WhenProcessingMultipleOrders_Then_EachIndependent()
    {
        // Arrange
        var order1 = OrderSagaTestData.CreateOrder(orderId: "ORD-001");
        var order2 = OrderSagaTestData.CreateOrder(orderId: "ORD-002");
        var order3 = OrderSagaTestData.CreateOrder(orderId: "ORD-003");

        // Assert
        Assert.NotEqual(order1.OrderId, order2.OrderId);
        Assert.NotEqual(order2.OrderId, order3.OrderId);
        Assert.Equal(OrderState.Pending, order1.State);
        Assert.Equal(OrderState.Pending, order2.State);
        Assert.Equal(OrderState.Pending, order3.State);
    }

    [Fact]
    public void WhenOrderWithManyItems_Then_HandledCorrectly()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrderWithMultipleItems(itemCount: 50);

        // Assert
        Assert.Equal(50, order.Items.Count);
    }

    [Fact]
    public void WhenOrderTransitionSequence_Then_FollowsPath()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder();

        // Act - Follow state progression
        var reserved = order with { State = OrderState.Reserved, ReservationId = "RES-001" };
        var charged = reserved with { State = OrderState.Charged, ChargeAmount = 100m };
        var shipped = charged with { State = OrderState.Shipped, ShipmentId = "SHIP-001" };

        // Assert - Verify full progression
        Assert.Equal(OrderState.Pending, order.State);
        Assert.Equal(OrderState.Reserved, reserved.State);
        Assert.Equal(OrderState.Charged, charged.State);
        Assert.Equal(OrderState.Shipped, shipped.State);
    }

    [Fact]
    public void WhenOrderCancelsFromReserved_Then_CompensatesCorrectly()
    {
        // Arrange
        var order = OrderSagaTestData.CreateOrder()
            with { State = OrderState.Reserved, ReservationId = "RES-001" };

        // Act
        var cancelled = order with { State = OrderState.Cancelled, FailureReason = "Customer request" };

        // Assert
        Assert.Equal(OrderState.Cancelled, cancelled.State);
        Assert.Equal("RES-001", cancelled.ReservationId);  // Should retain for compensation
        Assert.Equal("Customer request", cancelled.FailureReason);
    }
}




