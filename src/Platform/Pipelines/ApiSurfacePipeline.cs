using EvalApp.Consumer;
using EvalApp.Solid.Starter.Platform.Events;
using EvalApp.Solid.Starter.Platform.Merge;
using EvalApp.Solid.Starter.Platform.Steps;
using EvalApp.Solid.Starter.Platform.Support;

namespace EvalApp.Solid.Starter.Platform.Pipelines;

public static class ApiSurfacePipeline
{
    public static ICompiledPipeline<ApiSurfaceData> Build(List<string>? eventLog = null)
        => BuildInternal(eventLog, useBayesianTuning: false, tunerSettingsPath: null);

    public static ICompiledPipeline<ApiSurfaceData> BuildWithBayesianTuning(
        string tunerSettingsPath,
        List<string>? eventLog = null)
        => BuildInternal(eventLog, useBayesianTuning: true, tunerSettingsPath);

    private static ICompiledPipeline<ApiSurfaceData> BuildInternal(
        List<string>? eventLog,
        bool useBayesianTuning,
        string? tunerSettingsPath)
    {
        ICompiledPipeline<ApiSurfaceData> pipeline = null!;

        var eventSink = eventLog ?? new List<string>();
        var serviceProvider = new LocalServiceProvider(new Dictionary<Type, object>
        {
            [typeof(FactoryResolvedStep)] = new FactoryResolvedStep(new BonusService(3))
        });

        var appBuilder = Eval.App("ApiSurface")
            .WithContext(NullGlobalContext.Instance)
            .WithStepFactory(new ServiceProviderStepFactory(serviceProvider))
            .WithResource(ResourceKind.Network)
            .WithPressure(new TutorialPressureResource("tutorial-pressure", 0.35f))
            .WithWindowBudget(60)
            .WithWindowBudget("ui-budget", 60)
            .WithTuning();

        if (useBayesianTuning)
        {
            appBuilder = appBuilder.WithBayesianTuning(
                Path.Combine(Path.GetTempPath(), "evalapp-solid-api-surface-bayes.json"));
        }

        if (!string.IsNullOrWhiteSpace(tunerSettingsPath))
        {
            appBuilder = appBuilder.WithTunerSettings(tunerSettingsPath);
        }

        appBuilder
            .DefineDomain("Coverage", NullGlobalContext.Instance)
                .DefineTask<ApiSurfaceData>("RunApiSurface")
                    .WithEvents(new ApiSurfaceEvents(eventSink))
                    .AddStep<FactoryResolvedStep>("FactoryResolved")
                    .AddParallelGroup(
                        parallel => parallel
                            .AddBranch("ParallelLeft", data => data with { ParallelLeft = data.Input + 10 })
                            .AddBranch("ParallelRight", data => data with { ParallelRight = data.Input * 3 }),
                        new ApiSurfaceMergeStrategy())
                    .AddReadOnlyBridge(
                        "ReadOnlyBridge",
                        new BridgeProjectionStep(),
                        new ApiSurfaceData(Input: 7),
                        (current, bridge) => current.AppendTrace("Bridge:Merged") with
                        {
                            BridgedValue = bridge.BridgedValue
                        })
                    .BeginSaga()
                        .AddMaterialize(
                            "SagaMaterialize",
                            data => ApiSurfaceHelpers.StreamSagaItemsAsync(data),
                            (data, items) => data.AppendTrace($"Saga:Materialized:{items.Count}") with
                            {
                                SagaItems = items
                            })
                        .AddForEach<int>(
                            "SagaForEach",
                            select: data => data.SagaItems ?? [],
                            parallelism: Tunable.ForItems(2),
                            configure: item => item.AddStep("ScaleItem", value => value * 10),
                            merge: (data, items) => data.AppendTrace($"Saga:ForEach:{items.Count}") with
                            {
                                ProcessedSagaItems = items.ToList()
                            })
                        .AddStepWithCompensation(
                            "SagaReserve",
                            forward: data => data.AppendTrace("Saga:Reserve") with
                            {
                                SagaCounter = data.SagaCounter + 10
                            },
                            compensate: data => data.AppendTrace("Saga:Reserve:Compensate") with
                            {
                                SagaCounter = data.SagaCounter - 10,
                                SagaCompensated = true
                            })
                        .AddGate(
                            ResourceKind.Network,
                            onWaiting: null,
                            configure: gate => gate.AddStep(
                                "SagaNetworkCall",
                                data =>
                                {
                                    if (data.TriggerSagaFailure)
                                        throw new InvalidOperationException("Simulated saga gate failure.");

                                    return data.AppendTrace("Saga:GateForward") with
                                    {
                                        SagaCounter = data.SagaCounter + 5
                                    };
                                }),
                            compensate: new SagaGateCompensationStep())
                    .EndSaga()
                    .Pressure(
                        "tutorial-pressure",
                        pressure => pressure.AddStep(
                            "PressureMarker",
                            data => data.AppendTrace("Pressure:CustomScope") with
                            {
                                PressureScoped = true
                            }))
                    .WindowBudget(
                        window => window.AddStep(
                            "WindowBudgetMarker",
                            data => data.AppendTrace("Pressure:WindowBudgetScope")))
                    .AddStep("Finalize", data => data.AppendTrace("ApiSurface:Completed"))
                    .Run(out pipeline)
            .DefineDomain("Supplemental", NullGlobalContext.Instance)
                .DefineTask<ApiSurfaceData>("SupplementalTask")
                    .AddStep("SupplementalNoOp", data => data.AppendTrace("Supplemental:NoOp"))
                    .Run()
                .Build();

        return pipeline;
    }
}

