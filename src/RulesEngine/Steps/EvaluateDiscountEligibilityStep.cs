namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// Determine eligibility for discount based on shopper profile rules.
/// - VIP customers always eligible.
/// - Non-VIP: eligible if purchase history > 5 OR total spend > $1000.
/// Pure step: no I/O, single responsibility.
/// </summary>
public class EvaluateDiscountEligibilityStep : PureStep<PricingData>
{
    public override PricingData Execute(PricingData data)
    {
        var shopper = data.Order.Shopper;
        var eligible = shopper.IsVip
            || shopper.PurchaseHistoryCount > 5
            || shopper.TotalSpend > 1000m;

        return data with { IsEligibleForDiscount = eligible };
    }
}
