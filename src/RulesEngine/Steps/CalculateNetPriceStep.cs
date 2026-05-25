namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// Calculate net price by summing all items.
/// Pure step: one responsibility, no I/O.
/// </summary>
public class CalculateNetPriceStep : PureStep<PricingData>
{
    public override PricingData Execute(PricingData data)
    {
        var netPrice = data.Order.Items
            .Aggregate(0m, (sum, item) => sum + item.BasePrice);

        return data with { NetPrice = netPrice };
    }
}
