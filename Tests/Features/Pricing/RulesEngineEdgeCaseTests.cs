using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Pricing;
using EvalApp.Solid.Starter.Shared;
using EvalApp.Solid.Starter.Tests.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Pricing;

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
}



