using EvalApp.Solid.Starter.Orders.Services;

namespace EvalApp.Solid.Starter.Orders.Steps;

/// <summary>
/// Charges customer payment via IPaymentService.
/// AsyncStep: makes external service call (payment API).
/// SRP: Single responsibility = charge payment.
/// Stores ChargeAmount for compensation (refund).
/// </summary>
public class ChargePaymentStep : AsyncStep<OrderSagaData>
{
    private readonly IPaymentService _paymentService;
    private readonly decimal _orderAmount;

    public ChargePaymentStep(IPaymentService paymentService, decimal orderAmount)
    {
        _paymentService = paymentService;
        _orderAmount = orderAmount;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var chargeAmount = await _paymentService.ChargeAsync(data.CustomerId, _orderAmount, ct);

        if (chargeAmount == null)
            throw new InvalidOperationException($"Payment failed for customer {data.CustomerId}");

        return data with
        {
            State = OrderState.Charged,
            ChargeAmount = chargeAmount.Value
        };
    }
}

