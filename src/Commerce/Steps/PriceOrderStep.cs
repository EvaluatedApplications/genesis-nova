using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;

namespace EvalApp.Solid.Starter.Commerce.Steps;

public sealed class PriceOrderStep : AsyncStep<CommerceWorkflowData>
{
    private readonly ICompiledPipeline<CommerceWorkflowData> _pricingPipeline;

    public PriceOrderStep(ICompiledPipeline<CommerceWorkflowData> pricingPipeline)
    {
        _pricingPipeline = pricingPipeline;
    }

    public override async ValueTask<CommerceWorkflowData> ExecuteAsync(CommerceWorkflowData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var result = await _pricingPipeline.RunAsync(data, ct);
        return result switch
        {
            PipelineResult<CommerceWorkflowData>.Success success => success.Data,
            PipelineResult<CommerceWorkflowData>.Failure failure =>
                throw new InvalidOperationException($"Pricing pipeline failed: {failure.Message}"),
            PipelineResult<CommerceWorkflowData>.Skipped skipped =>
                throw new InvalidOperationException($"Pricing pipeline skipped: {skipped.Reason}"),
            _ => throw new InvalidOperationException("Pricing pipeline returned an unexpected result")
        };
    }
}

