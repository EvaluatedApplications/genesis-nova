using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Pricing;

/// <summary>
/// Pipeline data flowing through the RulesEngine rules, capturing decision outcomes at each stage.
/// Immutable record for clean data flow through the pricing pipeline.
/// </summary>
public record PricingData(
    OrderContext Order,
    decimal NetPrice = 0m,
    bool IsEligibleForDiscount = false,
    decimal DiscountPercent = 0m,
    decimal PromotionDiscount = 0m,
    decimal Tax = 0m,
    decimal FinalPrice = 0m)
{
    /// <summary>
    /// Convenience property for backwards compatibility.
    /// </summary>
    public decimal SubTotal => NetPrice * (1m - DiscountPercent);
}

