namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// Finalize price by applying discount.
/// Pure step: one responsibility (discount calculation).
/// </summary>
public class CalculateFinalPriceStep : PureStep<PricingData>
{
    public override PricingData Execute(PricingData data)
    {
        var finalPrice = data.NetPrice * (1m - data.DiscountPercent);
        return data with { FinalPrice = finalPrice };
    }
}
