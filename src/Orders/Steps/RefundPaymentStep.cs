using EvalApp.Solid.Starter.Orders.Services;

namespace EvalApp.Solid.Starter.Orders.Steps;

/// <summary>
/// COMPENSATION step: Refunds charge if saga fails.
/// Triggered by saga framework if any subsequent step fails.
/// Uses ChargeAmount stored by ChargePaymentStep.
/// SRP: Single responsibility = refund (undo) payment.
/// </summary>
public class RefundPaymentStep : AsyncStep<OrderSagaData>
{
    private readonly IPaymentService _paymentService;

    public RefundPaymentStep(IPaymentService paymentService)
    {
        _paymentService = paymentService;
    }

    public override async ValueTask<OrderSagaData> ExecuteAsync(OrderSagaData data, CancellationToken ct)
    {
        if (data.ChargeAmount == null || data.ChargeAmount <= 0)
            throw new InvalidOperationException("ChargeAmount missing or invalid; cannot refund");

        ct.ThrowIfCancellationRequested();
        var refunded = await _paymentService.RefundAsync(data.ChargeAmount.Value, ct);

        if (!refunded)
            throw new InvalidOperationException($"Failed to refund ${data.ChargeAmount:F2}");

        return data with
        {
            State = OrderState.Cancelled,
            ChargeAmount = null
        };
    }
}

