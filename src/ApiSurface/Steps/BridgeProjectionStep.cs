using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Features.ApiSurface.Steps;

public sealed class BridgeProjectionStep : IStep<ApiSurfaceData>
{
    public ValueTask<ApiSurfaceData> ExecuteAsync(ApiSurfaceData data, CancellationToken ct = default)
        => ValueTask.FromResult(
            data.AppendTrace("Bridge:Projected") with
            {
                BridgedValue = data.Input + 1000
            });
}
