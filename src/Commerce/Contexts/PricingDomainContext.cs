using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Commerce.Contexts;

public sealed class PricingDomainContext : DomainContext
{
    public PricingDomainContext(
        decimal standardDiscount = 0.10m,
        decimal vipDiscount = 0.25m,
        decimal taxRate = 0.08m)
    {
        StandardDiscount = standardDiscount;
        VipDiscount = vipDiscount;
        TaxRate = taxRate;
    }

    public static PricingDomainContext Default => new();
    public static PricingDomainContext ForVip => new(standardDiscount: 0.15m, vipDiscount: 0.30m, taxRate: 0.08m);

    public decimal StandardDiscount { get; }
    public decimal VipDiscount { get; }
    public decimal TaxRate { get; }
}

