using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.ApiSurface.Support;

namespace EvalApp.Solid.Starter.Features.ApiSurface.Steps;

public sealed class FactoryResolvedStep(BonusService bonusService) : IStep<ApiSurfaceData>
{
    private readonly BonusService _bonusService = bonusService;

    public ValueTask<ApiSurfaceData> ExecuteAsync(ApiSurfaceData data, CancellationToken ct = default)
        => ValueTask.FromResult(
            data.AppendTrace("Factory:ResolvedStep") with
            {
                SagaCounter = data.SagaCounter + _bonusService.Bonus
            });
}
