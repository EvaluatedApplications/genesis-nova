using EvalApp.Solid.Starter.Features.RulesEngine;
using EvalApp.Solid.Starter.Shared;
using System.Collections.Immutable;

var items = ImmutableList.Create(
    new Item("TSHIRT-1", "Blue T-Shirt", 45m, ItemCategory.Standard),
    new Item("JEANS-1", "Denim Jeans", 120m, ItemCategory.Standard),
    new Item("HAT-CLEAR", "Summer Hat", 15m, ItemCategory.Clearance));

var shopper = new ShopperProfile(
    CustomerId: "SHOPPER-42",
    PurchaseHistoryCount: 12,
    TotalSpend: 5000m,
    IsVip: true);

var order = new OrderContext(
    Shopper: shopper,
    Items: items,
    PromotionCode: "SUMMER20");

var data = new PricingData(order);
var pipeline = RulesEnginePipeline.Build();

var result = await pipeline.RunAsync(data);

var finalData = result switch
{
    PipelineResult<PricingData>.Success s => s.Data,
    PipelineResult<PricingData>.Failure f => f.Data,
    _ => throw new InvalidOperationException("Pipeline did not complete")
};

Console.WriteLine("=== RulesEngine Feature Demo ===");
Console.WriteLine();
Console.WriteLine($"Shopper: {shopper.CustomerId} (VIP: {shopper.IsVip})");
Console.WriteLine($"Purchase History: {shopper.PurchaseHistoryCount} orders");
Console.WriteLine($"Total Spend: ${shopper.TotalSpend:F2}");
Console.WriteLine();
Console.WriteLine("Items:");
foreach (var item in items)
{
    Console.WriteLine($"  - {item.Name} [{item.Category}]: ${item.BasePrice:F2}");
}
Console.WriteLine();
Console.WriteLine($"Net Price: ${finalData.NetPrice:F2}");
Console.WriteLine($"Eligible for Discount: {finalData.IsEligibleForDiscount}");
Console.WriteLine($"Discount: {finalData.DiscountPercent:P0}");
Console.WriteLine($"Final Price: ${finalData.FinalPrice:F2}");
Console.WriteLine();
Console.WriteLine($"Savings: ${finalData.NetPrice - finalData.FinalPrice:F2}");
