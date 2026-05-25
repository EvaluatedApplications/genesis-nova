using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// Apply discount rules based on promotion code and eligibility.
/// Pure step: one responsibility, no I/O.
/// </summary>
public class ApplyPromotionRulesStep : PureStep<PricingData>
{
    public override PricingData Execute(PricingData data)
    {
        decimal discount = 0m;

        if (!data.IsEligibleForDiscount)
            return data with { DiscountPercent = 0m };

        // Rule: Clearance items get 20% off
        if (data.Order.Items.Any(i => i.Category == ItemCategory.Clearance))
            discount = 0.20m;

        // Rule: SUMMER20 promo = 15% off
        if (data.Order.PromotionCode == "SUMMER20")
            discount = Math.Max(discount, 0.15m);

        // Rule: VIP gets additional 5% on top
        if (data.Order.Shopper.IsVip)
            discount = Math.Min(discount + 0.05m, 0.50m); // Cap at 50%

        return data with { DiscountPercent = discount };
    }
}
