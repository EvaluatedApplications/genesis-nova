using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;

namespace EvalApp.Solid.Starter.Commerce.Steps;

public sealed class GenerateShippingLabelStep : ContextSideEffectStep<NullGlobalContext, FulfillmentDomainContext, CommerceWorkflowData>
{
    protected override async ValueTask<CommerceWorkflowData> ExecuteAsync(
        CommerceWorkflowData data,
        NullGlobalContext global,
        FulfillmentDomainContext fulfillment,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        await Task.Delay(10, ct);

        return data with
        {
            LabelId = $"LBL-{fulfillment.WarehouseCode}-{Guid.NewGuid():N}",
            Trace = (data.Trace ?? ImmutableList<string>.Empty).Add("Fulfillment:Label")
        };
    }
}

