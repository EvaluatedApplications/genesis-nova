using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Commerce.Contexts;

public sealed class FulfillmentDomainContext : DomainContext
{
    public FulfillmentDomainContext(
        string warehouseCode = "WEST",
        decimal standardShipping = 8.50m,
        decimal expressShipping = 18.00m,
        decimal freeShippingThreshold = 250m)
    {
        WarehouseCode = warehouseCode;
        StandardShipping = standardShipping;
        ExpressShipping = expressShipping;
        FreeShippingThreshold = freeShippingThreshold;
    }

    public static FulfillmentDomainContext Default => new();
    public static FulfillmentDomainContext Premium => new(
        warehouseCode: "EAST",
        standardShipping: 6.00m,
        expressShipping: 14.00m,
        freeShippingThreshold: 175m);

    public string WarehouseCode { get; }
    public decimal StandardShipping { get; }
    public decimal ExpressShipping { get; }
    public decimal FreeShippingThreshold { get; }
}

