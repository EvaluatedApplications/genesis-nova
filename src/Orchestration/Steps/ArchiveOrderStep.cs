using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Contexts;

namespace EvalApp.Solid.Starter.Features.Orchestration.Steps;

public sealed class ArchiveOrderStep : ContextSideEffectStep<NullGlobalContext, FulfillmentDomainContext, CommerceWorkflowData>
{
    protected override async ValueTask<CommerceWorkflowData> ExecuteAsync(
        CommerceWorkflowData data,
        NullGlobalContext global,
        FulfillmentDomainContext fulfillment,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        await Task.Delay(5, ct);

        return data with
        {
            ArchiveId = $"ARC-{fulfillment.WarehouseCode}-{Guid.NewGuid():N}",
            Trace = (data.Trace ?? ImmutableList<string>.Empty).Add("Fulfillment:Archive")
        };
    }
}
