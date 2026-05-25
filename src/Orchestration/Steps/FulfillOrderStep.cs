using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.Orchestration;

namespace EvalApp.Solid.Starter.Features.Orchestration.Steps;

public sealed class FulfillOrderStep : AsyncStep<CommerceWorkflowData>
{
    private readonly ICompiledPipeline<CommerceWorkflowData> _fulfillmentPipeline;

    public FulfillOrderStep(ICompiledPipeline<CommerceWorkflowData> fulfillmentPipeline)
    {
        _fulfillmentPipeline = fulfillmentPipeline;
    }

    public override async ValueTask<CommerceWorkflowData> ExecuteAsync(CommerceWorkflowData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var result = await _fulfillmentPipeline.RunAsync(data, ct);
        return result switch
        {
            PipelineResult<CommerceWorkflowData>.Success success => success.Data,
            PipelineResult<CommerceWorkflowData>.Failure failure =>
                throw new InvalidOperationException($"Fulfillment pipeline failed: {failure.Message}"),
            PipelineResult<CommerceWorkflowData>.Skipped skipped =>
                throw new InvalidOperationException($"Fulfillment pipeline skipped: {skipped.Reason}"),
            _ => throw new InvalidOperationException("Fulfillment pipeline returned an unexpected result")
        };
    }
}
