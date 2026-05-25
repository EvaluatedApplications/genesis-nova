using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.RulesEngine;
using EvalApp.Solid.Starter.Shared;
using EvalApp.Solid.Starter.Tests.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Features.RulesEngine;

/// <summary>
/// Additional comprehensive tests for RulesEngine covering edge cases.
/// </summary>
public class RulesEngineEdgeCaseTests
{
    [Fact]
    public void WhenOrderWithZeroPrice_Then_NetPriceIsZero()
    {
        // Arrange
        var order = TestData.CreateOrder(items: ImmutableList<Item>.Empty);
        var data = new PricingData(order);
        var step = new CalculateNetPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(0m, result.NetPrice);
    }

    [Fact]
    public void WhenHighPriceOrder_Then_CalculatesCorrectly()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 10000m, ItemCategory.Premium),
            TestData.CreateItem("SKU2", 5000m, ItemCategory.Premium));
        var order = TestData.CreateOrder(items: items);
        var data = new PricingData(order);
        var step = new CalculateNetPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(15000m, result.NetPrice);
    }

    [Fact]
    public async Task WhenNonVipNonQualified_Then_NoDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false);
        var order = TestData.CreateOrder(shopper: shopper);
        var data = new PricingData(order, IsEligibleForDiscount: false);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0m, result.DiscountPercent);
    }

    [Fact]
    public async Task WhenVipWithNoSpecialItems_Then_BaseDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: true);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order, IsEligibleForDiscount: true);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0.05m, result.DiscountPercent);  // VIP base only
    }

    [Fact]
    public async Task WhenMultipleClearanceItems_Then_ClearanceDiscountApplies()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false, totalSpend: 2000m);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Clearance),
            TestData.CreateItem("SKU2", 50m, ItemCategory.Clearance));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0.20m, result.DiscountPercent);  // Clearance discount
    }

    [Fact]
    public async Task WhenVipHighPurchaseHistoryClearance_Then_CapAtMaximum()
    {
        // Arrange
        var shopper = TestData.CreateShopper(
            isVip: true,
            purchaseHistory: 20);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Clearance));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order, IsEligibleForDiscount: true);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0.25m, result.DiscountPercent);
    }

    [Fact]
    public void WhenCalculatingFinalPriceWithLargeDiscount_Then_CalculatesCorrectly()
    {
        // Arrange
        var data = new PricingData(
            Order: TestData.CreateOrder(),
            NetPrice: 1000m,
            DiscountPercent: 0.30m);
        var step = new CalculateFinalPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(756m, result.FinalPrice);  // 1000 - (1000 * 0.30) + tax
    }

    [Fact]
    public void WhenCalculatingFinalPriceWithNoDiscount_Then_PriceUnchanged()
    {
        // Arrange
        var data = new PricingData(
            Order: TestData.CreateOrder(),
            NetPrice: 500m,
            DiscountPercent: 0m);
        var step = new CalculateFinalPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(540m, result.FinalPrice);
    }

    [Fact]
    public void WhenEvaluatingHighSpendThreshold_Then_QualifiesForDiscount()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 600m, ItemCategory.Standard),
            TestData.CreateItem("SKU2", 600m, ItemCategory.Standard));
        var shopper = TestData.CreateShopper(isVip: false, totalSpend: 1500m);
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);

        // Calculate net price first
        var netPriceStep = new CalculateNetPriceStep();
        var withNetPrice = netPriceStep.Execute(data);

        // Then evaluate eligibility
        var eligibilityStep = new EvaluateDiscountEligibilityStep();
        var result = eligibilityStep.Execute(withNetPrice);

        // Assert
        Assert.True(result.IsEligibleForDiscount);  // High spend qualifies
    }

    [Fact]
    public async Task WhenLargeOrderWithMultipleItems_Then_CalculatesCorrectly()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU2", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU3", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU4", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU5", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(items: items);
        var initialData = new PricingData(order);

        // Act
        var pipeline = RulesEnginePipeline.Build();
        var result = await pipeline.RunAsync(initialData);
        var finalData = result.GetData();

        // Assert
        Assert.Equal(500m, finalData.NetPrice);
        Assert.True(finalData.FinalPrice > 0);
    }
}
