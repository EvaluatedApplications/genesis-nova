using System.Collections.Immutable;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Commerce.Steps;

public sealed class PrepareFulfillmentLinesStep : PureStep<CommerceWorkflowData>
{
    public override CommerceWorkflowData Execute(CommerceWorkflowData data)
    {
        var lines = data.Order.Items
            .Select(item => new CommerceLineItem(
                item.Sku,
                item.Name,
                item.BasePrice,
                item.Category is ItemCategory.Premium or ItemCategory.Clearance))
            .ToImmutableList();

        var packableLines = lines.Where(line => !line.IsFragile).ToImmutableList();
        var skippedCount = lines.Count - packableLines.Count;

        return data with
        {
            Lines = packableLines,
            Trace = data.AppendTrace($"Fulfillment:LinesPrepared:{packableLines.Count}:{skippedCount}").Trace
        };
    }
}

