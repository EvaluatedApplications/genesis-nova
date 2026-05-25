using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Contexts;

namespace EvalApp.Solid.Starter.Features.Orchestration.Steps;

public sealed class SelectShippingStep : ContextPureStep<NullGlobalContext, FulfillmentDomainContext, CommerceWorkflowData>
{
    protected override ValueTask<CommerceWorkflowData> TransformAsync(
        CommerceWorkflowData data,
        NullGlobalContext global,
        FulfillmentDomainContext fulfillment,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var requiresExpress = data.FinalTotal >= fulfillment.FreeShippingThreshold || data.Order.Shopper.IsVip;
        var shippingMethod = requiresExpress ? "Express" : "Standard";
        var shipping = requiresExpress ? fulfillment.ExpressShipping : fulfillment.StandardShipping;

        return ValueTask.FromResult(data with
        {
            ShippingMethod = shippingMethod,
            Shipping = shipping,
            FinalTotal = data.FinalTotal + shipping,
            Trace = (data.Trace ?? ImmutableList<string>.Empty).Add($"Fulfillment:Shipping:{shippingMethod}")
        });
    }
}
