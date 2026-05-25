using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Pricing;
using EvalApp.Solid.Starter.Shared;
using EvalApp.Solid.Starter.Tests.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Pricing;

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




