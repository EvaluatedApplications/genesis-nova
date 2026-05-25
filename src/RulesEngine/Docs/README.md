# RulesEngine Feature

## Problem Statement

**The Anti-Pattern:**
```csharp
// ❌ Typical business logic: if/else explosion
public decimal CalculatePrice(ShopperProfile shopper, List<Item> items, string promo)
{
    var netPrice = items.Sum(i => i.BasePrice);
    bool eligible = shopper.IsVip 
        || shopper.PurchaseHistoryCount > 5 
        || shopper.TotalSpend > 1000m;
    
    decimal discount = 0;
    if (eligible)
    {
        if (items.Any(i => i.Category == ItemCategory.Clearance))
            discount = 0.20m;
        if (promo == "SUMMER20")
            discount = Math.Max(discount, 0.15m);
        if (shopper.IsVip)
            discount = Math.Min(discount + 0.05m, 0.50m);
    }
    
    return netPrice * (1m - discount);
}
```

**Pain Points:**
- Logic becomes a monolithic ball of mud as rules grow.
- Testing requires mocking the entire method.
- Adding new rules means touching the same method (violates OCP).
- Hard to reason about order of operations.
- Difficult to instrument/log intermediate values.

## Solution: EvalApp Pipeline

**The Pipeline Approach:**
```
CalculateNetPrice
    ↓
EvaluateEligibility
    ↓
ApplyPromotionRules
    ↓
CalculateFinalPrice
```

Each step is a **pure function** with a single responsibility:

### Step 1: CalculateNetPriceStep
- **Input**: OrderContext with items
- **Output**: PricingData.NetPrice
- **Responsibility**: Sum items
- **Why**: Isolate calculation concern

### Step 2: EvaluateDiscountEligibilityStep
- **Input**: ShopperProfile
- **Output**: PricingData.IsEligibleForDiscount
- **Responsibility**: Evaluate eligibility rules
- **Why**: Separate eligibility from promotion logic

### Step 3: ApplyPromotionRulesStep
- **Input**: Eligibility flag, promotion code, item categories, VIP status
- **Output**: PricingData.DiscountPercent
- **Responsibility**: Apply promotion/discount logic
- **Why**: Centralize rule engine in one place (still extensible)

### Step 4: CalculateFinalPriceStep
- **Input**: NetPrice + DiscountPercent
- **Output**: PricingData.FinalPrice
- **Responsibility**: Apply discount
- **Why**: Isolate calculation from logic

## SOLID Principles Applied

### Single Responsibility (SRP)
- Each step does one thing: calculate, evaluate, apply, or finalize.
- Tests focus on one concern per step.

### Open/Closed (OCP)
- Want to add a new promotion rule? 
  - **Old way**: Modify the method (closed for modification ✗)
  - **New way**: Extend ApplyPromotionRulesStep logic (open for extension ✓)
  - Pipeline topology never changes.

### Liskov Substitution (LSP)
- All steps are PureStep<PricingData>; they're interchangeable.
- Can replace step implementation without changing pipeline.

### Interface Segregation (ISP)
- Steps don't depend on bloated interfaces; they use StepContext.
- Each step gets only what it needs (data record fields).

### Dependency Inversion (DIP)
- Pipeline depends on PureStep<T> abstraction, not concrete steps.
- New steps integrate without pipeline recompilation.

## Customization Guide

### Adding a "Free Shipping" Rule
1. Extend `PricingData` to include `ShippingCost` field.
2. Create `CalculateShippingStep` (pure step).
3. Add it after `CalculateNetPrice`.
4. In `CalculateFinalPrice`, apply shipping.

### Adding a "Bulk Discount" Rule
1. Modify `ApplyPromotionRulesStep`: add bulk quantity check.
2. No pipeline changes needed.

### Conditional Pricing by Category
1. Create new step `ApplyCategoryPricingStep`.
2. Insert after `CalculateNetPrice`.
3. Update `PricingData` with category-specific adjustments.

## Testing Patterns

### Unit Test: Single Step
```csharp
[Fact]
public void WhenVipShopper_Then_IsEligibleForDiscount()
{
    // Arrange
    var vip = new ShopperProfile("C1", 0, 0m, IsVip: true);
    var data = new PricingData(new OrderContext(vip, []));
    var step = new EvaluateDiscountEligibilityStep();

    // Act
    var result = step.Execute(data);

    // Assert
    Assert.True(result.IsEligibleForDiscount);
}
```

### Integration Test: Full Pipeline
```csharp
[Fact]
public async Task WhenValidOrder_Then_PipelineCalculatesCorrectPrice()
{
    // Arrange
    var pipeline = RulesEnginePipeline.Build();
    var order = TestData.CreateValidOrder();
    var data = new PricingData(order);

    // Act
    var result = await pipeline.RunAsync(data, CancellationToken.None);

    // Assert
    Assert.Equal(expected, result.FinalPrice);
}
```

## Key Takeaways

1. **Think Data First**: Design `PricingData` record before writing steps.
2. **One Step = One Responsibility**: Each step transforms data in one way.
3. **Immutability**: Use `data with { ... }` for all mutations.
4. **No I/O Yet**: RulesEngine is pure; gates come in later features.
5. **Topology = Orchestration**: Pipeline builder defines flow; steps implement logic.

## Running the Example

```bash
cd EvalApp.Solid.Starter
dotnet run --feature RulesEngine

# Output:
# Order: $500 (4 items)
# Shopper: VIP, $5000 total spend
# Discount: 20% (clearance items)
# Final Price: $400
```
