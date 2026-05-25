using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Platform.Steps;

public sealed class SagaGateCompensationStep : IStep<ApiSurfaceData>
{
    public ValueTask<ApiSurfaceData> ExecuteAsync(ApiSurfaceData data, CancellationToken ct = default)
        => ValueTask.FromResult(
            data.AppendTrace("Saga:Gate:Compensate") with
            {
                SagaCounter = data.SagaCounter - 5,
                SagaCompensated = true
            });
}

