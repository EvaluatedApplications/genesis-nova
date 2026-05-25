using EvalApp.Solid.Starter.Commerce;

namespace EvalApp.Solid.Starter.Commerce.Steps;

public sealed class PackLineStep : PureStep<CommerceLineItem>
{
    public override CommerceLineItem Execute(CommerceLineItem data)
    {
        if (data.IsFragile)
            throw new InvalidOperationException($"Fragile line {data.Sku} requires manual handling");

        return data with
        {
            IsPacked = true,
            PackageId = $"PKG-{data.Sku}"
        };
    }
}

