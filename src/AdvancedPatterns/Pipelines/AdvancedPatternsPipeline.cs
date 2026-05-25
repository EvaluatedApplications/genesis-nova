using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.AdvancedPatterns.Middleware;

namespace EvalApp.Solid.Starter.Features.AdvancedPatterns.Pipelines;

public static class AdvancedPatternsPipeline
{
    public static ICompiledPipeline<AdvancedDemoData> Build(
        ForEachFailureMode failureMode = ForEachFailureMode.ContinueOnError)
    {
        ICompiledPipeline<AdvancedDemoData> pipeline = null!;

        Eval.App("AdvancedPatterns")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Network, new TunableConfig(Min: 1, Max: 8, Default: 4))
            .WithResource(ResourceKind.Cpu, new TunableConfig(Min: 1, Max: 4, Default: 2))
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithWindowBudget(30)
            .WithTuning()
            .DefineDomain("Advanced", NullGlobalContext.Instance)
                .DefineTask<AdvancedDemoData>("DemonstrateEvalApp")
                    .WithMiddleware(new TraceMiddleware("Trace"))
                    .WithMiddleware(new RetryOnceMiddleware())
                    .WithMiddleware(new TimeoutGuardMiddleware(TimeSpan.FromSeconds(10)))
                    .AddStep("SeedMeta", data => data with
                    {
                        Meta = data.Meta ?? new AdvancedMeta("Seeded", DateTime.UtcNow)
                    })
                    .Materialize(
                        "MaterializeInput",
                        data => AdvancedPatternHelpers.StreamItemsAsync(data.InputItems),
                        (data, items) => data.AppendTrace($"Materialize:{items.Count}") with
                        {
                            MaterializedItems = items
                        })
                    .AddSubTaskFor(
                        data => data.Meta ?? new AdvancedMeta(),
                        (data, meta) => data with { Meta = meta },
                        "Meta",
                        subTask => subTask.AddStep(
                            "StampMeta",
                            meta => meta with
                            {
                                Stage = "Prepared",
                                LastUpdatedUtc = DateTime.UtcNow
                            }))
                    .Gate(
                        ResourceKind.Network,
                        _ => { },
                        gate => gate.AddStepWithFallback(
                            "FetchQuote",
                            primary: data =>
                            {
                                if (data.ForcePrimaryQuoteFailure)
                                    throw new InvalidOperationException("Primary quote endpoint unavailable.");

                                return data.AppendTrace("Quote:Primary") with
                                {
                                    Quote = 125m,
                                    QuoteSource = "primary"
                                };
                            },
                            fallback: data => data.AppendTrace("Quote:Fallback") with
                            {
                                Quote = 100m,
                                QuoteSource = "fallback"
                            }))
                    .ForEach<int>(
                        data => data.MaterializedItems ?? [],
                        (data, items) =>
                        {
                            var transformed = items.ToList();
                            var totalInput = data.MaterializedItems?.Count ?? transformed.Count;
                            var errorCount = Math.Max(0, totalInput - transformed.Count);

                            return data.AppendTrace($"ForEach:{failureMode}:{transformed.Count}") with
                            {
                                MaterializedItems = transformed,
                                SuccessCount = transformed.Count,
                                ErrorCount = errorCount
                            };
                        },
                        "TransformItems",
                        Tunable.ForItems(),
                        failureMode,
                        item => item.AddStep(
                            "TransformItem",
                            value =>
                            {
                                if (value < 0)
                                    throw new InvalidOperationException($"Negative item {value} is invalid.");
                                return value * 2;
                            }))
                    .WindowBudget(scope => scope.AddStep(
                        "BudgetAwareMarker",
                        data => data.AppendTrace("WindowBudget:Applied")))
                    .Gate(
                        ResourceKind.Cpu,
                        _ => { },
                        gate => gate.AddStep(
                            "ComputeDigest",
                            async (data, ct) => await AdvancedPatternHelpers.ComputeDigestAsync(data, ct)))
                    .Gate(
                        ResourceKind.DiskIO,
                        _ => { },
                        gate => gate.AddStep(
                            "PersistSnapshot",
                            async (data, ct) => await AdvancedPatternHelpers.PersistSnapshotAsync(data, ct)))
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}
