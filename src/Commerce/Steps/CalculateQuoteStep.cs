using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;

namespace EvalApp.Solid.Starter.Commerce.Steps;

public sealed class CalculateQuoteStep : ContextPureStep<NullGlobalContext, PricingDomainContext, CommerceWorkflowData>
{
    protected override ValueTask<CommerceWorkflowData> TransformAsync(
        CommerceWorkflowData data,
        NullGlobalContext global,
        PricingDomainContext pricing,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var net = data.Order.Items.Sum(item => item.BasePrice);
        var discountRate = data.Order.Shopper.IsVip ? pricing.VipDiscount : pricing.StandardDiscount;
        var discount = net * discountRate;
        var taxable = net - discount;
        var tax = taxable * pricing.TaxRate;
        var final = taxable + tax;

        return ValueTask.FromResult(data with
        {
            NetTotal = net,
            Discount = discount,
            Tax = tax,
            FinalTotal = final,
            Trace = (data.Trace ?? ImmutableList<string>.Empty).Add($"Pricing:{(data.Order.Shopper.IsVip ? "VIP" : "Standard")}")
        });
    }
}

