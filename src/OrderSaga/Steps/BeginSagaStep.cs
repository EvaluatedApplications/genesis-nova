namespace EvalApp.Solid.Starter.Features.OrderSaga.Steps;

/// <summary>
/// Begins saga transaction: validates order (non-empty items, valid customer).
/// Pure step: single responsibility, no I/O, no side effects.
/// SRP: Only validates order structure.
/// </summary>
public class BeginSagaStep : PureStep<OrderSagaData>
{
    public override OrderSagaData Execute(OrderSagaData data)
    {
        if (data.Items == null || data.Items.Count == 0)
            throw new InvalidOperationException("Order must contain at least one item");

        if (string.IsNullOrWhiteSpace(data.CustomerId))
            throw new InvalidOperationException("CustomerId is required");

        return data with { State = OrderState.Pending };
    }
}
