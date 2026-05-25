using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// Pipeline data flowing through the RulesEngine rules, capturing decision outcomes at each stage.
/// </summary>
public record PricingData(
    OrderContext Order,
    decimal NetPrice = 0m,
    bool IsEligibleForDiscount = false,
    decimal DiscountPercent = 0m,
    decimal FinalPrice = 0m);
