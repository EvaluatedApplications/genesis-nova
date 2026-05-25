using System.Collections.Immutable;
using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Features.Orchestration;

public record CommerceLineItem(
    string Sku,
    string Name,
    decimal BasePrice,
    bool IsFragile,
    bool IsPacked = false,
    string? PackageId = null);

public record CommerceWorkflowData(
    OrderContext Order,
    ImmutableList<CommerceLineItem> Lines,
    decimal NetTotal = 0m,
    decimal Discount = 0m,
    decimal Tax = 0m,
    decimal Shipping = 0m,
    decimal FinalTotal = 0m,
    string ShippingMethod = "Standard",
    string? LabelId = null,
    string? ArchiveId = null,
    ImmutableList<string>? Trace = null)
{
    public CommerceWorkflowData AppendTrace(string entry)
        => this with { Trace = (Trace ?? ImmutableList<string>.Empty).Add(entry) };
}
