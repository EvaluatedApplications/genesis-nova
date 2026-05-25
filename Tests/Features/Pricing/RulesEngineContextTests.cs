using Xunit;
using System.Collections.Immutable;
using EvalApp.Solid.Starter.Pricing;
using EvalApp.Solid.Starter.Pricing.Context;
using EvalApp.Solid.Starter.Pricing.Pipelines;
using EvalApp.Solid.Starter.Shared;
using EvalApp.Solid.Starter.Tests.Shared;

namespace EvalApp.Solid.Starter.Tests.Pricing;

/// <summary>
/// Tests for RulesEngine with PricingContext (Phase 4 Advanced Patterns).
/// Demonstrates domain-context-driven pricing logic.
/// </summary>
public class RulesEngineContextTests
{
    [Fact]
    public async Task WhenVipCustomerWithDefaultContext_Then_AppliesVipDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: true);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(100m, finalData.NetPrice); // 1 × 100
        // VIP gets 5% additional discount on top of eligibility rules
        Assert.True(finalData.DiscountPercent > 0, "VIP customer should receive discount");
    }

    [Fact]
    public async Task WhenStandardCustomerWithHighSpend_Then_AppliesStandardDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false, totalSpend: 1500m);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard),
            TestData.CreateItem("SKU2", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(200m, finalData.NetPrice); // 2 × 100
        // Standard customer with high spend qualifies for discount
        Assert.True(finalData.IsEligibleForDiscount, "High-spend customer should be eligible");
        Assert.True(finalData.DiscountPercent > 0, "Eligible customer should receive discount");
    }

    [Fact]
    public async Task WhenClearanceItems_Then_AppliesClearanceDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false, totalSpend: 2000m);
        var items = ImmutableList.Create(
            TestData.CreateItem("CLEARANCE-1", 50m, ItemCategory.Clearance));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(50m, finalData.NetPrice);
        // Clearance items get 20% discount
        Assert.True(finalData.DiscountPercent >= 0.20m, "Clearance items should get at least 20% discount");
    }

    [Fact]
    public async Task WhenMultipleItemTypes_Then_AppliesMaximumApplicableDiscount()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: true);
        var items = ImmutableList.Create(
            TestData.CreateItem("PREMIUM-1", 200m, ItemCategory.Premium),
            TestData.CreateItem("CLEARANCE-1", 100m, ItemCategory.Clearance));
        var order = TestData.CreateOrder(shopper: shopper, items: items, promo: "SUMMER20");
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(300m, finalData.NetPrice); // 200 + 100
        // VIP with clearance items should get maximum allowed discount (capped at 50%)
        Assert.True(finalData.DiscountPercent > 0, "Should have discount");
        Assert.True(finalData.DiscountPercent <= 0.50m, "Discount should be capped at 50%");
    }

    [Fact]
    public async Task WhenCalculatingTax_Then_AppliesTaxToDiscountedPrice()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(100m, finalData.NetPrice);
        // Tax should be calculated on the final discounted price
        // Default tax rate from PricingContext.Default is 8%
        var expectedTax = finalData.SubTotal * 0.08m;
        Assert.Equal(expectedTax, finalData.Tax, 2 /* decimal places */);
    }

    [Fact]
    public async Task WhenCalculatingFinalPrice_Then_IncludesDiscountAndTax()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false, totalSpend: 2000m); // Qualifies for discount
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard));
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        // Final price should be: (NetPrice - discount) + tax
        var expectedFinalPrice = finalData.SubTotal + finalData.Tax;
        Assert.Equal(expectedFinalPrice, finalData.FinalPrice, 2 /* decimal places */);
        // Final price should be less than original (due to discount, slightly offset by tax)
        Assert.True(finalData.FinalPrice < finalData.NetPrice, "Discounted price should be less than net price");
    }

    [Fact]
    public async Task WhenNoEligibility_Then_NoDiscountApplied()
    {
        // Arrange
        var shopper = TestData.CreateShopper(isVip: false, purchaseHistory: 0, totalSpend: 0m);
        var items = ImmutableList.Create(
            TestData.CreateItem("SKU1", 100m, ItemCategory.Standard)); // No clearance
        var order = TestData.CreateOrder(shopper: shopper, items: items);
        var data = new PricingData(order);
        
        var pipeline = RulesEnginePipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = result.GetData();

        // Assert
        Assert.NotNull(finalData);
        Assert.Equal(100m, finalData.NetPrice);
        Assert.False(finalData.IsEligibleForDiscount, "Low-spend non-VIP should not be eligible");
        Assert.Equal(0m, finalData.DiscountPercent);
        // Final price with only tax
        var expectedFinalPrice = finalData.NetPrice * 1.08m; // Add 8% tax
        Assert.Equal(expectedFinalPrice, finalData.FinalPrice, 2 /* decimal places */);
    }

    [Fact]
    public void WhenPricingContextDefault_Then_HasExpectedDiscountValues()
    {
        // Arrange & Act
        var context = PricingContext.Default;

        // Assert
        Assert.Equal(0.10m, context.BaseDiscount);
        Assert.Equal(0.25m, context.VipDiscount);
        Assert.Equal(0.05m, context.VipLoyaltyBonus);
        Assert.Equal(0.08m, context.TaxRate);
    }

    [Fact]
    public void WhenPricingContextForVip_Then_HasVipPremiumValues()
    {
        // Arrange & Act
        var context = PricingContext.ForVip;

        // Assert
        Assert.Equal(0.15m, context.BaseDiscount);
        Assert.Equal(0.30m, context.VipDiscount);
        Assert.Equal(0.10m, context.VipLoyaltyBonus);
        Assert.Equal(0.08m, context.TaxRate);
    }
}




