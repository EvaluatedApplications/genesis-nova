namespace EvalApp.Solid.Starter.Pricing;

/// <summary>
/// Finalize price by ensuring all calculations are complete.
/// Pure step: verifies that tax has been applied and final price is calculated.
/// </summary>
public class CalculateFinalPriceStep : PureStep<PricingData>
{
    public override PricingData Execute(PricingData data)
    {
        // If ApplyTaxStep has already run, FinalPrice should be set
        // This step can be used for any final validation or adjustments
        // Currently, the FinalPrice is calculated by ApplyTaxStep as: SubTotal + Tax
        
        // Ensure FinalPrice includes discount and tax
        if (data.FinalPrice == 0m && data.NetPrice > 0m)
        {
            // Fallback calculation if ApplyTaxStep hasn't run
            var discountedPrice = data.NetPrice * (1m - data.DiscountPercent);
            var tax = discountedPrice * 0.08m; // Default tax rate
            var finalPrice = discountedPrice + tax;
            return data with { FinalPrice = finalPrice };
        }

        return data;
    }
}

