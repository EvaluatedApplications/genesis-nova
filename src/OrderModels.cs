namespace EvalApp.Solid.Starter.Shared;

public record ShopperProfile(
    string CustomerId,
    int PurchaseHistoryCount,
    decimal TotalSpend,
    bool IsVip = false,
    DateTime MemberSince = default);

public record Item(
    string Sku,
    string Name,
    decimal BasePrice,
    ItemCategory Category);

public enum ItemCategory
{
    Standard,
    Premium,
    Clearance
}

public record OrderContext(
    ShopperProfile Shopper,
    ImmutableList<Item> Items,
    string PromotionCode = "");

