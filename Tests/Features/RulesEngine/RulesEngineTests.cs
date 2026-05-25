using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.RulesEngine;
using EvalApp.Solid.Starter.Features.RulesEngine.Pipelines;
using EvalApp.Solid.Starter.Shared;
using EvalApp.Solid.Starter.Tests.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Features.RulesEngine;

public class CalculateNetPriceStepTests
{
    [Fact]
    public void WhenMultipleItems_Then_SumsAllPrices()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU2", 50m, ItemCategory.Premium),
            TestData.CreateItem("SKU3", 25m, ItemCategory.Clearance));
        var order = TestData.CreateOrder(items: items);
        var data = new PricingData(order);
        var step = new CalculateNetPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(175m, result.NetPrice);
    }

    [Fact]
    public void WhenEmptyItems_Then_NetPriceIsZero()
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
}

public class EvaluateDiscountEligibilityStepTests
{
    [Fact]
    public void WhenVipShopper_Then_EligibleForDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: true);
        var order = TestData.CreateOrder(shopper: shopper);
        var data = new PricingData(order);
        var step = new EvaluateDiscountEligibilityStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.True(result.IsEligibleForDiscount);
    }

    [Fact]
    public void WhenHighPurchaseHistory_Then_EligibleForDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(purchaseHistory: 10);
        var order = TestData.CreateOrder(shopper: shopper);
        var data = new PricingData(order);
        var step = new EvaluateDiscountEligibilityStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.True(result.IsEligibleForDiscount);
    }

    [Fact]
    public void WhenHighTotalSpend_Then_EligibleForDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(totalSpend: 2000m);
        var order = TestData.CreateOrder(shopper: shopper);
        var data = new PricingData(order);
        var step = new EvaluateDiscountEligibilityStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.True(result.IsEligibleForDiscount);
    }

    [Fact]
    public void WhenNewShopper_Then_NotEligibleForDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(purchaseHistory: 0, totalSpend: 0m);
        var order = TestData.CreateOrder(shopper: shopper);
        var data = new PricingData(order);
        var step = new EvaluateDiscountEligibilityStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.False(result.IsEligibleForDiscount);
    }
}

public class ApplyPromotionRulesStepTests
{
    [Fact]
    public async Task WhenNotEligible_Then_NoDiscount()
    {
        // Arrange
        var data = new PricingData(TestData.CreateOrder(), IsEligibleForDiscount: false);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0m, result.DiscountPercent);
    }

    [Fact]
    public async Task WhenClearanceItems_Then_20PercentDiscount()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem(category: ItemCategory.Clearance));
        var shopper = TestData.CreateShopper(totalSpend: 2000m);
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0.20m, result.DiscountPercent);
    }

    [Fact]
    public async Task WhenSummer20Promo_Then_15PercentDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(totalSpend: 2000m);
        var order = TestData.CreateOrder(shopper: shopper, promo: "SUMMER20");
        var data = new PricingData(order);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert
        Assert.Equal(0.15m, result.DiscountPercent);
    }

    [Fact]
    public async Task WhenVipAndClearance_Then_HighestDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: true);
        var items = ImmutableList.Create(
            TestData.CreateItem(category: ItemCategory.Clearance));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order, IsEligibleForDiscount: true);

        // Act
        var result = await RulesEngineTestHelpers.RunPipelineAsync(data);

        // Assert: 20% (clearance) + 5% (VIP) = 25%
        Assert.Equal(0.25m, result.DiscountPercent);
    }
}

public class CalculateFinalPriceStepTests
{
    [Fact]
    public void WhenNoDiscount_Then_FinalPriceEqualsNetPrice()
    {
        // Arrange
        var data = new PricingData(TestData.CreateOrder(), NetPrice: 100m, DiscountPercent: 0m);
        var step = new CalculateFinalPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(108m, result.FinalPrice);
    }

    [Fact]
    public void WhenDiscount_Then_FinalPriceIsReduced()
    {
        // Arrange
        var data = new PricingData(TestData.CreateOrder(), NetPrice: 100m, DiscountPercent: 0.20m);
        var step = new CalculateFinalPriceStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(86.4m, result.FinalPrice);
    }
}

public class RulesEnginePipelineTests
{
    [Fact]
    public async Task WhenValidOrder_Then_PipelineCalculatesFinalPrice()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard));
        var shopper = TestData.CreateShopper(isVip: true);
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<PricingData>.Success)result).Data;

        // Assert: VIP gets 5% discount, then 8% tax
        Assert.Equal(102.6m, finalData.FinalPrice);
    }

    [Fact]
    public async Task WhenVipWithClearanceAndPromo_Then_MaximumDiscount()
    {
        // Arrange
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Clearance));
        var shopper = TestData.CreateShopper(isVip: true);
        var order = TestData.CreateOrder(shopper: shopper, items: items, promo: "SUMMER20");
        var data = new PricingData(order);
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<PricingData>.Success)result).Data;

        // Assert: 20% (clearance) + 5% (VIP) = 25%, then 8% tax on discounted total
        Assert.Equal(81m, finalData.FinalPrice);
    }
}

internal static class RulesEngineTestHelpers
{
    public static async Task<PricingData> RunPipelineAsync(PricingData data)
    {
        var pipeline = RulesEnginePipeline.Build();
        var result = await pipeline.RunAsync(data);
        return result.GetData();
    }
}
