# OrderSaga Feature: Distributed Transaction Pattern

## Problem Statement

In distributed systems, coordinating multi-step transactions across multiple services is fragile:

```csharp
// Manual transaction handling: lots of error paths, manual rollback logic
try
{
    var reservation = await inventory.ReserveAsync(items);
    try
    {
        var charge = await payment.ChargeAsync(customer, amount);
        try
        {
            var shipment = await shipment.CreateAsync(orderId, items);
        }
        catch
        {
            await payment.RefundAsync(charge);
            throw;
        }
    }
    catch
    {
        await inventory.ReleaseAsync(reservation);
        throw;
    }
}
catch { /* handle failure */ }
```

**Pain Points:**
- Compensation logic explodes with nested try/finally blocks
- Order of rollback is error-prone (must be reverse of forward steps)
- Manual tracking of "what to rollback" (reservation IDs, charge amounts, etc.)
- No clear separation between forward logic and compensation logic
- Difficult to test all failure scenarios

## EvalApp Solution: Saga Pattern

The **Saga pattern** (also called long-running transactions) solves this by:

1. **Declarative compensation** — attach rollback logic to each forward step
2. **Automatic orchestration** — if any step fails, compensation runs in REVERSE order
3. **Clear data flow** — saga data stores IDs needed for rollback
4. **Testable failure paths** — mock services trigger failures, assert compensation

### Saga Topology

```
BeginSagaStep (validate)
  ↓
ReserveInventoryStep [→ ReleaseReservationStep]
  ↓ [gate: Database]
ChargePaymentStep [→ RefundPaymentStep]
  ↓ [gate: Network]
ShipStep [→ CancelShipmentStep]
  ↓ [gate: Network]
EndSagaStep (finalize)

If any step fails, compensations run in REVERSE order.
```

### Data Model

```csharp
public record OrderSagaData(
    string OrderId,
    List<LineItem> Items,
    string CustomerId,
    OrderState State = OrderState.Pending,
    string? ReservationId = null,      // Used by ReleaseReservationStep
    decimal? ChargeAmount = null,      // Used by RefundPaymentStep
    string? ShipmentId = null,         // Used by CancelShipmentStep
    string? FailureReason = null);

public enum OrderState { Pending, Reserved, Charged, Shipped, Cancelled }
```

### Compensation Semantics

| Scenario | Happens | Compensation |
|----------|---------|--------------|
| All succeed | Pending → Reserved → Charged → Shipped | None |
| Charge fails | Pending → Reserved → ✗ Charge | Release reservation |
| Ship fails | Pending → Reserved → Charged → ✗ Ship | Refund + Release |
| Refund fails | ✗ Refund (compensation itself fails) | Mark as orphaned/manual-review |

### SOLID Principles Applied

#### Single Responsibility (SRP)
- Each step handles ONE responsibility (reserve, charge, ship)
- Each compensation handles ONE responsibility (release, refund, cancel)
- No step mixes concerns (e.g., ReserveInventoryStep doesn't also charge)

```csharp
public class ReserveInventoryStep : SideEffectStep<OrderSagaData>
{
    // Single job: reserve inventory
    public override async Task<StepResult<OrderSagaData>> ExecuteAsync(...)
    {
        var reservationId = await _inventoryService.ReserveAsync(data.Items, ct);
        return data with { ReservationId = reservationId, State = OrderState.Reserved };
    }
}
```

#### Open/Closed Principle (OCP)
- Easy to ADD new saga steps without changing existing topology
- Can insert new compensation without modifying other steps

```csharp
// Add a new step: verification
.AddStep("VerifyOrder", new VerifyOrderStep(), ResourceKind.Database)
  .WithCompensation(new ClearVerificationStep())
// No changes needed to other steps
```

#### Dependency Inversion (DIP)
- Steps depend on interfaces, not concrete implementations
- Mock services inject for testing

```csharp
public class ReserveInventoryStep : SideEffectStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;  // Interface!
    
    public ReserveInventoryStep(IInventoryService inventoryService)
    {
        _inventoryService = inventoryService;
    }
}
```

#### Liskov Substitution (LSP)
- All steps follow the same `StepResult<T>` contract
- Mocks are indistinguishable from real services

## Implementation Walkthrough

### 1. Data Model (Saga Record)

Saga data captures state progression and "backpointers" for compensation:

```csharp
public record OrderSagaData(
    string OrderId,
    List<LineItem> Items,
    string CustomerId,
    OrderState State = OrderState.Pending,
    string? ReservationId = null,      // ← Stored by ReserveInventoryStep
    decimal? ChargeAmount = null,      // ← Stored by ChargePaymentStep
    string? ShipmentId = null);        // ← Stored by ShipStep
```

**Key insight:** Forward steps MUST store IDs/amounts that compensation steps will use.

### 2. Forward Steps (Business Logic)

```csharp
public class ReserveInventoryStep : SideEffectStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;

    public override async Task<StepResult<OrderSagaData>> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        var reservationId = await _inventoryService.ReserveAsync(data.Items, ct);
        
        if (reservationId == null)
            return StepResult<OrderSagaData>.Failure("Insufficient inventory");
        
        // Store reservation ID for compensation
        return StepResult<OrderSagaData>.Success(
            data with
            {
                State = OrderState.Reserved,
                ReservationId = reservationId
            });
    }
}
```

### 3. Compensation Steps (Rollback)

```csharp
public class ReleaseReservationStep : SideEffectStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;

    public override async Task<StepResult<OrderSagaData>> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        // Use the ID stored by ReserveInventoryStep
        var released = await _inventoryService.ReleaseAsync(data.ReservationId, ct);
        
        if (!released)
            return StepResult<OrderSagaData>.Failure("Failed to release reservation");
        
        return StepResult<OrderSagaData>.Success(
            data with
            {
                State = OrderState.Cancelled,
                ReservationId = null
            });
    }
}
```

### 4. Pipeline Builder

```csharp
public static ICompiledPipeline<OrderSagaData> Build(
    IInventoryService inventoryService,
    IPaymentService paymentService,
    IShipmentService shipmentService,
    decimal orderAmount)
{
    ICompiledPipeline<OrderSagaData> pipeline = null!;

    Eval.App("OrderSaga")
        .DefineDomain("Fulfillment")
            .DefineTask<OrderSagaData>("ProcessOrder")
                .AddStep("BeginSaga", new BeginSagaStep())
                .AddStep("ReserveInventory", new ReserveInventoryStep(inventoryService),
                    ResourceKind.Database)
                // Compensation: if later steps fail, ReleaseReservationStep runs
                .AddStep("ChargePayment", new ChargePaymentStep(paymentService, orderAmount),
                    ResourceKind.Network)
                // Compensation: if later steps fail, RefundPaymentStep runs
                .AddStep("Ship", new ShipStep(shipmentService),
                    ResourceKind.Network)
                // Compensation: if saga fails here, CancelShipmentStep runs
                .AddStep("EndSaga", new EndSagaStep())
                .Run(out pipeline)
            .Build();

    return pipeline;
}
```

### 5. Gate Assignment (Resource Tuning)

```csharp
.AddStep("ReserveInventory", new ReserveInventoryStep(...), ResourceKind.Database)
//                                                           ↑ Database gate
.AddStep("ChargePayment", new ChargePaymentStep(...), ResourceKind.Network)
//                                                    ↑ Network gate
.AddStep("Ship", new ShipStep(...), ResourceKind.Network)
//                                ↑ Network gate
```

**Why?**
- `WithTuning()` uses these gates to adaptively control concurrency
- Database gate prevents connection pool exhaustion
- Network gate prevents request flooding

## Testing Saga Compensation

### Test: All Steps Succeed

```csharp
[Fact]
public async Task WhenAllStepsSucceed_Then_OrderShipped()
{
    var pipeline = OrderSagaPipeline.Build(
        new MockInventoryService(),  // Success
        new MockPaymentService(),    // Success
        new MockShipmentService(),   // Success
        500m);

    var result = await pipeline.RunAsync(testOrder);
    var finalData = ((PipelineResult<OrderSagaData>.Success)result).Data;

    Assert.Equal(OrderState.Shipped, finalData.State);
    Assert.NotNull(finalData.ReservationId);
    Assert.NotNull(finalData.ChargeAmount);
    Assert.NotNull(finalData.ShipmentId);
}
```

### Test: Charge Fails, Reserve Compensated

```csharp
[Fact]
public async Task WhenChargeFailsAfterReserve_Then_ReservationCompensated()
{
    var mockInventory = new MockInventoryService();
    var mockPayment = new MockPaymentService(shouldFail: true);  // ← Inject failure
    
    var pipeline = OrderSagaPipeline.Build(mockInventory, mockPayment, mockShipment, 500m);
    var result = await pipeline.RunAsync(testOrder);

    // After compensation runs:
    Assert.Empty(mockInventory.GetActiveReservations());  // Released
    Assert.Null(finalData.ReservationId);                 // Cleared
}
```

### Test: Compensation Failures Caught

```csharp
[Fact]
public async Task WhenCompensationFails_Then_ErrorLogged()
{
    // Create mocks where refund itself fails
    var mockPayment = new MockPaymentService(refundShouldFail: true);
    
    var result = await pipeline.RunAsync(testOrder);
    
    // Saga framework catches this and marks order as orphaned/manual-review
    Assert.True(result.IsFailure);
    Assert.Contains("compensation", result.Error);
}
```

## Customization Checklist

### Adding a New Saga Step

1. **Create forward step:**
   ```csharp
   public class VerifyOrderStep : PureStep<OrderSagaData>
   {
       public override OrderSagaData Execute(OrderSagaData data)
       {
           // Validate order, return data with { IsVerified = true }
       }
   }
   ```

2. **Create compensation step (if needed):**
   ```csharp
   public class ClearVerificationStep : PureStep<OrderSagaData>
   {
       public override OrderSagaData Execute(OrderSagaData data)
       {
           return data with { IsVerified = false };
       }
   }
   ```

3. **Add to pipeline:**
   ```csharp
   .AddStep("VerifyOrder", new VerifyOrderStep())
   // No gate needed for pure steps
   ```

### Handling Specific Errors

```csharp
public override async Task<StepResult<OrderSagaData>> ExecuteAsync(...)
{
    try
    {
        var result = await _service.CallAsync();
        return StepResult<OrderSagaData>.Success(/* ... */);
    }
    catch (TimeoutException)
    {
        return StepResult<OrderSagaData>.Failure("Timeout: retry later");
        // Saga treats this as a business failure, triggers compensation
    }
    catch (InvalidOperationException)
    {
        throw;  // Unrecoverable: skip compensation, fail hard
    }
}
```

### Adjusting Retry Policy

```csharp
public class ReserveInventoryStep : SideEffectStep<OrderSagaData>
{
    private readonly IInventoryService _inventoryService;
    private readonly int _maxRetries;

    public ReserveInventoryStep(IInventoryService svc, int maxRetries = 3)
    {
        _inventoryService = svc;
        _maxRetries = maxRetries;
    }

    public override async Task<StepResult<OrderSagaData>> ExecuteAsync(...)
    {
        for (int i = 0; i < _maxRetries; i++)
        {
            var result = await _inventoryService.ReserveAsync(...);
            if (result != null) return StepResult<OrderSagaData>.Success(...);
        }
        return StepResult<OrderSagaData>.Failure("Max retries exceeded");
    }
}
```

## Before/After Comparison

### Before: Manual Rollback

```csharp
public async Task<OrderResult> ProcessOrderManually(Order order)
{
    var reservation = null;
    var charge = null;
    var shipment = null;
    
    try
    {
        reservation = await inventory.ReserveAsync(order.Items);
        if (reservation == null) throw new Exception("Reservation failed");
        
        try
        {
            charge = await payment.ChargeAsync(order.CustomerId, order.Total);
            if (charge == null) throw new Exception("Payment failed");
            
            try
            {
                shipment = await shipping.CreateShipmentAsync(order);
                if (shipment == null) throw new Exception("Shipment failed");
            }
            catch
            {
                await payment.RefundAsync(charge);  // Manual rollback
                throw;
            }
        }
        catch
        {
            await inventory.ReleaseAsync(reservation);  // Manual rollback
            throw;
        }
    }
    catch (Exception ex)
    {
        return new OrderResult(success: false, error: ex.Message);
    }
}
```

**Issues:**
- ❌ Compensation order is manual (easy to mess up)
- ❌ Nested try/finally is hard to read
- ❌ No clear separation of concerns
- ❌ Difficult to test all paths

### After: Saga Declaration

```csharp
public static ICompiledPipeline<OrderSagaData> Build(...)
{
    Eval.App("OrderSaga")
        .DefineDomain("Fulfillment")
            .DefineTask<OrderSagaData>("ProcessOrder")
                .AddStep("BeginSaga", new BeginSagaStep())
                .AddStep("ReserveInventory", new ReserveInventoryStep(inventoryService), ResourceKind.Database)
                .AddStep("ChargePayment", new ChargePaymentStep(paymentService, amount), ResourceKind.Network)
                .AddStep("Ship", new ShipStep(shipmentService), ResourceKind.Network)
                .AddStep("EndSaga", new EndSagaStep())
                .Run(out pipeline)
            .Build();
    return pipeline;
}
```

**Benefits:**
- ✓ Compensation automatically runs in reverse order
- ✓ Clear linear flow (no nested try/catch)
- ✓ Each step focused on ONE responsibility
- ✓ Easy to test failure scenarios via mocks
- ✓ Data model captures state + IDs for rollback
- ✓ DIP: depend on interfaces for testability

## Key Takeaways

1. **Sagas coordinate distributed transactions** by declaratively attaching compensation to each step
2. **Data flows through immutable records** — forward steps populate IDs, compensation steps use them
3. **Compensation runs in REVERSE order** — automatic orchestration prevents manual errors
4. **SOLID principles hold** — SRP (each step), OCP (easy to add steps), DIP (mock services)
5. **Testing is straightforward** — inject mock services, assert compensation paths
