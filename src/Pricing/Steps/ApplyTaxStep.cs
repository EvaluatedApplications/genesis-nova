using EvalApp.Consumer;

using EvalApp.Solid.Starter.Pricing.Context;

namespace EvalApp.Solid.Starter.Pricing;

/// <summary>
/// Apply tax calculations based on order and pricing context.
/// Demonstrates context-driven tax rate selection.
/// ContextPureStep: one responsibility, no I/O, with injected domain policy.
/// </summary>
public class ApplyTaxStep : ContextPureStep<NullGlobalContext, PricingContext, PricingData>
{
    protected override ValueTask<PricingData> TransformAsync(
        PricingData data,
        NullGlobalContext global,
        PricingContext pricing,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var tax = data.SubTotal * pricing.TaxRate;
        return ValueTask.FromResult(data with
        {
            Tax = tax,
            FinalPrice = data.SubTotal + tax
        });
    }
}

