namespace EvalApp.Solid.Starter.Features.OrderSaga.Steps;

/// <summary>
/// Ends saga transaction: marks final state.
/// Pure step: captures final outcome without side effects.
/// SRP: Single responsibility = finalize order state.
/// </summary>
public class EndSagaStep : PureStep<OrderSagaData>
{
    public override OrderSagaData Execute(OrderSagaData data)
    {
        return data with
        {
            State = data.State == OrderState.Shipped ? OrderState.Shipped : OrderState.Cancelled
        };
    }
}
