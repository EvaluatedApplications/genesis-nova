using System.Collections.Immutable;
using EvalApp.Solid.Starter.Features.RulesEngine;
using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Tests.Shared;

public static class TestData
{
    public static ShopperProfile CreateShopper(
        string customerId = "C1",
        int purchaseHistory = 0,
        decimal totalSpend = 0m,
        bool isVip = false)
        => new ShopperProfile(customerId, purchaseHistory, totalSpend, isVip, DateTime.UtcNow);

    public static Item CreateItem(
        string sku = "SKU1",
        decimal price = 100m,
        ItemCategory category = ItemCategory.Standard)
        => new Item(sku, $"Item-{sku}", price, category);

    public static OrderContext CreateOrder(
        ShopperProfile? shopper = null,
        ImmutableList<Item>? items = null,
        string promo = "")
    {
        shopper ??= CreateShopper();
        items ??= ImmutableList<Item>.Empty;
        return new OrderContext(shopper, items, promo);
    }

    public static PricingData CreatePricingData(OrderContext? order = null)
    {
        order ??= CreateOrder();
        return new PricingData(order);
    }
}
