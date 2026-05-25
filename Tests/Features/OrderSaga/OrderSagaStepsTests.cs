using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.OrderSaga;
using EvalApp.Solid.Starter.Features.OrderSaga.Pipelines;
using EvalApp.Solid.Starter.Features.OrderSaga.Steps;
using EvalApp.Solid.Starter.Tests.Features.OrderSaga.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Features.OrderSaga;

public class BeginSagaStepTests
{
    [Fact]
    public void WhenValidOrder_Then_StateSetToPending()
    {
        // Arrange
        var data = OrderSagaTestData.CreateOrder();
        var step = new BeginSagaStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(OrderState.Pending, result.State);
    }

    [Fact]
    public void WhenEmptyItems_Then_ThrowsInvalidOperationException()
    {
        // Arrange
        var data = OrderSagaTestData.CreateOrder(items: new List<LineItem>());
        var step = new BeginSagaStep();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => step.Execute(data));
    }

    [Fact]
    public void WhenCustomerIdMissing_Then_ThrowsInvalidOperationException()
    {
        // Arrange
        var data = OrderSagaTestData.CreateOrder(customerId: "");
        var step = new BeginSagaStep();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => step.Execute(data));
    }
}

public class ReserveInventoryStepTests
{
    [Fact]
    public async Task WhenInventoryAvailable_Then_ReservationIdAssignedAndStateReserved()
    {
        // Arrange
        var mockInventory = new MockInventoryService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new ReserveInventoryStep(mockInventory);
        using var cts = new CancellationTokenSource();

        // Act
        var result = await step.ExecuteAsync(data, cts.Token);

        // Assert
        Assert.NotNull(result.ReservationId);
        Assert.Equal(OrderState.Reserved, result.State);
        Assert.NotEmpty(mockInventory.GetActiveReservations());
    }

    [Fact]
    public async Task WhenInventoryScarce_Then_ThrowsException()
    {
        // Arrange
        var mockInventory = new MockInventoryService(shouldFail: true);
        var data = OrderSagaTestData.CreateOrder();
        var step = new ReserveInventoryStep(mockInventory);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class ReleaseReservationStepTests
{
    [Fact]
    public async Task WhenReservationIdExists_Then_ReleasesAndStateCancelled()
    {
        // Arrange
        var mockInventory = new MockInventoryService();
        var reserveStep = new ReserveInventoryStep(mockInventory);
        var data = OrderSagaTestData.CreateOrder();
        using var cts = new CancellationTokenSource();
        
        var reserved = await reserveStep.ExecuteAsync(data, cts.Token);
        
        var releaseStep = new ReleaseReservationStep(mockInventory);

        // Act
        var result = await releaseStep.ExecuteAsync(reserved, cts.Token);

        // Assert
        Assert.Equal(OrderState.Cancelled, result.State);
        Assert.Null(result.ReservationId);
        Assert.Empty(mockInventory.GetActiveReservations());
    }

    [Fact]
    public async Task WhenReservationIdMissing_Then_ThrowsException()
    {
        // Arrange
        var mockInventory = new MockInventoryService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new ReleaseReservationStep(mockInventory);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class ChargePaymentStepTests
{
    [Fact]
    public async Task WhenPaymentSucceeds_Then_ChargeAmountSetAndStateCharged()
    {
        // Arrange
        var mockPayment = new MockPaymentService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new ChargePaymentStep(mockPayment, 100m);
        using var cts = new CancellationTokenSource();

        // Act
        var result = await step.ExecuteAsync(data, cts.Token);

        // Assert
        Assert.Equal(100m, result.ChargeAmount);
        Assert.Equal(OrderState.Charged, result.State);
        Assert.NotEmpty(mockPayment.GetActiveCharges());
    }

    [Fact]
    public async Task WhenPaymentFails_Then_ThrowsException()
    {
        // Arrange
        var mockPayment = new MockPaymentService(shouldFail: true);
        var data = OrderSagaTestData.CreateOrder();
        var step = new ChargePaymentStep(mockPayment, 100m);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class RefundPaymentStepTests
{
    [Fact]
    public async Task WhenRefundSucceeds_Then_ChargeAmountNulledAndStateCancelled()
    {
        // Arrange
        var mockPayment = new MockPaymentService();
        var chargeStep = new ChargePaymentStep(mockPayment, 100m);
        var data = OrderSagaTestData.CreateOrder();
        using var cts = new CancellationTokenSource();
        
        var charged = await chargeStep.ExecuteAsync(data, cts.Token);

        var refundStep = new RefundPaymentStep(mockPayment);

        // Act
        var result = await refundStep.ExecuteAsync(charged, cts.Token);

        // Assert
        Assert.Null(result.ChargeAmount);
        Assert.Equal(OrderState.Cancelled, result.State);
        Assert.Empty(mockPayment.GetActiveCharges());
    }

    [Fact]
    public async Task WhenChargeAmountMissing_Then_ThrowsException()
    {
        // Arrange
        var mockPayment = new MockPaymentService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new RefundPaymentStep(mockPayment);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class ShipStepTests
{
    [Fact]
    public async Task WhenShipmentSucceeds_Then_ShipmentIdSetAndStateShipped()
    {
        // Arrange
        var mockShipment = new MockShipmentService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new ShipStep(mockShipment);
        using var cts = new CancellationTokenSource();

        // Act
        var result = await step.ExecuteAsync(data, cts.Token);

        // Assert
        Assert.NotNull(result.ShipmentId);
        Assert.Equal(OrderState.Shipped, result.State);
        Assert.NotEmpty(mockShipment.GetActiveShipments());
    }

    [Fact]
    public async Task WhenShipmentFails_Then_ThrowsException()
    {
        // Arrange
        var mockShipment = new MockShipmentService(shouldFail: true);
        var data = OrderSagaTestData.CreateOrder();
        var step = new ShipStep(mockShipment);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class CancelShipmentStepTests
{
    [Fact]
    public async Task WhenCancellationSucceeds_Then_ShipmentIdNulledAndStateCancelled()
    {
        // Arrange
        var mockShipment = new MockShipmentService();
        var shipStep = new ShipStep(mockShipment);
        var data = OrderSagaTestData.CreateOrder();
        using var cts = new CancellationTokenSource();
        
        var shipped = await shipStep.ExecuteAsync(data, cts.Token);

        var cancelStep = new CancelShipmentStep(mockShipment);

        // Act
        var result = await cancelStep.ExecuteAsync(shipped, cts.Token);

        // Assert
        Assert.Null(result.ShipmentId);
        Assert.Equal(OrderState.Cancelled, result.State);
        Assert.Empty(mockShipment.GetActiveShipments());
    }

    [Fact]
    public async Task WhenShipmentIdMissing_Then_ThrowsException()
    {
        // Arrange
        var mockShipment = new MockShipmentService();
        var data = OrderSagaTestData.CreateOrder();
        var step = new CancelShipmentStep(mockShipment);
        using var cts = new CancellationTokenSource();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await step.ExecuteAsync(data, cts.Token));
    }
}

public class EndSagaStepTests
{
    [Fact]
    public void WhenSagaCompletes_Then_StatePreservedAndOrderIdPreserved()
    {
        // Arrange - data in Shipped state (after all saga steps)
        var data = OrderSagaTestData.CreateOrder(state: OrderState.Shipped);
        var step = new EndSagaStep();

        // Act
        var result = step.Execute(data);

        // Assert - final state preserved as Shipped
        Assert.Equal(OrderState.Shipped, result.State);
        Assert.Equal(data.OrderId, result.OrderId);
    }

    [Fact]
    public void WhenSagaCancelledEarly_Then_StateCancelledPreserved()
    {
        // Arrange - data in Cancelled state (if saga failed)
        var data = OrderSagaTestData.CreateOrder(state: OrderState.Cancelled);
        var step = new EndSagaStep();

        // Act
        var result = step.Execute(data);

        // Assert - final state preserved as Cancelled
        Assert.Equal(OrderState.Cancelled, result.State);
        Assert.Equal(data.OrderId, result.OrderId);
    }
}

public class OrderSagaPipelineTests
{
    [Fact]
    public async Task WhenHappyPath_Then_OrderShipped()
    {
        // Arrange
        var inventory = new MockInventoryService();
        var payment = new MockPaymentService();
        var shipment = new MockShipmentService();

        var pipeline = OrderSagaPipeline.Build(inventory, payment, shipment, 100m);
        var order = OrderSagaTestData.CreateOrder();

        // Act
        var result = await pipeline.RunAsync(order);
        var finalData = ((PipelineResult<OrderSagaData>.Success)result).Data;

        // Assert - happy path ends in Shipped state
        Assert.Equal(OrderState.Shipped, finalData.State);
        Assert.NotNull(finalData.ReservationId);
        Assert.Equal(100m, finalData.ChargeAmount);
        Assert.NotNull(finalData.ShipmentId);
    }
}
