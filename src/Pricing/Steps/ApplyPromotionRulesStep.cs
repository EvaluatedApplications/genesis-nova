using EvalApp.Consumer;
using EvalApp.Solid.Starter.Pricing.Context;
using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Pricing;

/// <summary>
/// Apply discount rules based on promotion code and eligibility.
/// Pure step: one responsibility, no I/O.
/// </summary>
public class ApplyPromotionRulesStep : ContextPureStep<NullGlobalContext, PricingContext, PricingData>
{
    protected override ValueTask<PricingData> TransformAsync(
        PricingData data,
        NullGlobalContext global,
        PricingContext pricing,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        if (!data.IsEligibleForDiscount)
        {
            return ValueTask.FromResult(data with
            {
                DiscountPercent = 0m,
                PromotionDiscount = 0m
            });
        }

        var specialDiscount = data.Order.Shopper.IsVip ? 0m : pricing.BaseDiscount;

        // Rule: Clearance items get 20% off
        if (data.Order.Items.Any(i => i.Category == ItemCategory.Clearance))
            specialDiscount = Math.Max(specialDiscount, 0.20m);

        // Rule: SUMMER20 promo = 15% off
        if (data.Order.PromotionCode == "SUMMER20")
            specialDiscount = Math.Max(specialDiscount, 0.15m);

        var vipBonus = data.Order.Shopper.IsVip ? pricing.VipLoyaltyBonus : 0m;
        var discount = Math.Min(specialDiscount + vipBonus, pricing.VipDiscount);

        return ValueTask.FromResult(data with
        {
            DiscountPercent = discount,
            PromotionDiscount = specialDiscount
        });
    }
}

