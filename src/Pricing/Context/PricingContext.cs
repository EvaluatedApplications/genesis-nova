using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Pricing.Context;

/// <summary>
/// Pricing context for domain-specific configuration.
/// Demonstrates dependency injection for rules through pipeline domain context.
/// </summary>
public sealed class PricingContext : DomainContext
{
    public PricingContext(
        decimal baseDiscount = 0.10m,
        decimal vipDiscount = 0.25m,
        decimal vipLoyaltyBonus = 0.05m,
        decimal taxRate = 0.08m)
    {
        BaseDiscount = baseDiscount;
        VipDiscount = vipDiscount;
        VipLoyaltyBonus = vipLoyaltyBonus;
        TaxRate = taxRate;
    }

    /// <summary>
    /// Default pricing configuration for standard customers.
    /// </summary>
    public static PricingContext Default => new();

    /// <summary>
    /// VIP pricing configuration with higher discounts.
    /// </summary>
    public static PricingContext ForVip => new(
        baseDiscount: 0.15m,
        vipDiscount: 0.30m,
        vipLoyaltyBonus: 0.10m,
        taxRate: 0.08m);

    public decimal BaseDiscount { get; }
    public decimal VipDiscount { get; }
    public decimal VipLoyaltyBonus { get; }
    public decimal TaxRate { get; }
}

