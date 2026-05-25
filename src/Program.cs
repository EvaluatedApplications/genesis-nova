using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.ApiSurface;
using EvalApp.Solid.Starter.Features.ApiSurface.Events;
using EvalApp.Solid.Starter.Features.ApiSurface.Merge;
using EvalApp.Solid.Starter.Features.ApiSurface.Pipelines;
using EvalApp.Solid.Starter.Features.ApiSurface.Steps;
using EvalApp.Solid.Starter.Features.ApiSurface.Support;
using EvalApp.Solid.Starter.Features.AdvancedPatterns;
using EvalApp.Solid.Starter.Features.AdvancedPatterns.Middleware;
using EvalApp.Solid.Starter.Features.AdvancedPatterns.Pipelines;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Contexts;
using EvalApp.Solid.Starter.Features.Orchestration.Pipelines;
using EvalApp.Solid.Starter.Features.Orchestration.Steps;
using EvalApp.Solid.Starter.Features.RulesEngine.Context;
using EvalApp.Solid.Starter.Features.RulesEngine.Pipelines;
using EvalApp.Solid.Starter.Features.OrderSaga.Steps;
using EvalApp.Solid.Starter.Features.BatchSync.Pipelines;
using EvalApp.Solid.Starter.Features.Ingestion.Pipelines;
using System.Collections.Immutable;

// ═══════════════════════════════════════════════════════════════════════════════
//  EvalApp SOLID Starter — Unified Application Manifest
//  All 7 features wired in a single Eval.App() declaration.
//  Each domain below maps to one SOLID tutorial chapter.
// ═══════════════════════════════════════════════════════════════════════════════

// Pipeline handles — set via Run(out ...) during builder evaluation
ICompiledPipeline<PricingData>         rulesPipeline            = null!;
ICompiledPipeline<BatchSyncData>       batchPipeline            = null!;
ICompiledPipeline<IngestionData>       ingestionPipeline        = null!;
ICompiledPipeline<OrderSagaData>       sagaPipeline             = null!;
ICompiledPipeline<CommerceWorkflowData> commercePricingPipeline  = null!;
ICompiledPipeline<CommerceWorkflowData> commerceFulfillPipeline  = null!;
ICompiledPipeline<CommerceWorkflowData> orchestrationPipeline    = null!;
ICompiledPipeline<AdvancedDemoData>    advancedPipeline         = null!;
ICompiledPipeline<ApiSurfaceData>      apiSurfacePipeline       = null!;

// Services for OrderSaga (DIP: depend on interfaces, inject concrete impls here)
IInventoryService inventoryService = new MockInventoryService();
IPaymentService   paymentService   = new MockPaymentService(chargeAmount: 250m);
IShipmentService  shipmentService  = new MockShipmentService();

// ApiSurface: event sink + service provider (demonstrates WithStepFactory / DI resolution)
var eventLog = new List<string>();
var serviceProvider = new LocalServiceProvider(new Dictionary<Type, object>
{
    [typeof(FactoryResolvedStep)] = new FactoryResolvedStep(new BonusService(3))
});

// ─────────────────────────────────────────────────────────────────────────────
//  App-level configuration — superset of all domain resource requirements
// ─────────────────────────────────────────────────────────────────────────────
Eval.App("SolidStarter")
    .WithContext(NullGlobalContext.Instance)
    .WithStepFactory(new ServiceProviderStepFactory(serviceProvider))   // DI first, Activator fallback
    .WithResource(ResourceKind.Network,  new TunableConfig(Min: 1, Max: 8,  Default: 4))
    .WithResource(ResourceKind.Cpu,      new TunableConfig(Min: 1, Max: 4,  Default: 2))
    .WithResource(ResourceKind.DiskIO,   new TunableConfig(Min: 1, Max: 2,  Default: 1))
    .WithResource(ResourceKind.Database, new TunableConfig(Min: 1, Max: 4,  Default: 2))
    .WithPressure(new TutorialPressureResource("tutorial-pressure", 0.35f))
    .WithWindowBudget(60)
    .WithWindowBudget("ui-budget", 60)
    .WithTuning()

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 1 · RulesEngine
//  Pure business rules pipeline. No I/O — all PureStep / ContextPureStep.
//  Teaches: SRP (one step = one rule), OCP (add rules without topology changes),
//           DIP (ContextPureStep depends on PricingContext abstraction, not config).
//  Flow: CalculateNetPrice → EvaluateEligibility → ApplyPromoRules →
//        ApplyTax [ContextPureStep] → CalculateFinalPrice
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Pricing", PricingContext.Default)
        .DefineTask<PricingData>("CalculatePrice")
            .AddStep("CalculateNetPrice",   new CalculateNetPriceStep())
            .AddStep("EvaluateEligibility", new EvaluateDiscountEligibilityStep())
            .AddStep<ApplyPromotionRulesStep>("ApplyPromoRules")
            .AddStep<ApplyTaxStep>("ApplyTax")
            .AddStep("CalculateFinalPrice", new CalculateFinalPriceStep())
            .Run(out rulesPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 2 · BatchSync
//  Async I/O with partial failure. Items that fail are tracked, not fatal.
//  Teaches: SRP (fetch / process / summarize separation),
//           OCP (swap ProcessBatchStep without changing topology).
//  Flow: FetchItems → ProcessBatch → CalculateSummary
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Processing")
        .DefineTask<BatchSyncData>("SyncBatch")
            .AddStep("FetchItems",       new FetchItemsStep())
            .AddStep("ProcessBatch",     new ProcessBatchStep(successRate: 0.85))
            .AddStep("CalculateSummary", new CalculateSummaryStep())
            .Run(out batchPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 3 · Ingestion
//  Stream validation with collect-all-errors semantics.
//  Teaches: SRP (materialize / validate / summarize), ISP (no forced dependencies).
//  Flow: Materialize → ProcessAllItems → Summarize
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("BatchProcessing")
        .DefineTask<IngestionData>("ProcessStream")
            .AddStep("Materialize",    new MaterializeStep())
            .AddStep("ProcessAllItems", new ProcessAllItemsStep())
            .AddStep("Summarize",      new SummarizeResultsStep())
            .Run(out ingestionPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 4 · OrderSaga
//  Sequential distributed transaction. Each step records its side-effect ID so
//  the caller can compensate on partial failure by inspecting the result data.
//  Teaches: DIP (inject IInventoryService / IPaymentService / IShipmentService),
//           LSP (all saga steps honour the same AsyncStep<T> contract).
//  Flow: ReserveInventory → ChargePayment → Ship
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Fulfillment", NullGlobalContext.Instance)
        .DefineTask<OrderSagaData>("ProcessOrder")
            .AddStep("ReserveInventory",
                async (data, ct) => await new ReserveInventoryStep(inventoryService).ExecuteAsync(data, ct))
            .AddStep("ChargePayment",
                async (data, ct) => await new ChargePaymentStep(paymentService, 250m).ExecuteAsync(data, ct))
            .AddStep("Ship",
                async (data, ct) => await new ShipStep(shipmentService).ExecuteAsync(data, ct))
            .Run(out sagaPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 5a · CommercePricing  (sub-pipeline — piped into CommerceOrchestration)
//  Prices a CommerceWorkflowData through a domain-specific quote calculation.
//  Flow: CalculateQuote [ContextPureStep]
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("CommercePricing", PricingDomainContext.Default)
        .DefineTask<CommerceWorkflowData>("PriceOrder")
            .AddStep<CalculateQuoteStep>("CalculateQuote")
            .Run(out commercePricingPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 5b · CommerceFulfillment  (sub-pipeline — piped into CommerceOrchestration)
//  Demonstrates ForEach parallelism, If/Else branching, and two resource gates.
//  Flow: PrepareLines → ForEach(PackLine) → If(FreeShipping?) →
//        Gate:Network(GenerateLabel) → Gate:Database(ArchiveOrder)
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("CommerceFulfillment", FulfillmentDomainContext.Default)
        .DefineTask<CommerceWorkflowData>("FulfillOrder")
            .AddStep<PrepareFulfillmentLinesStep>("PrepareLines")
            .ForEach<CommerceLineItem>(
                data => data.Lines,
                (data, lines) => data with
                {
                    Lines = lines.ToImmutableList(),
                    Trace = data.AppendTrace($"Fulfillment:Packed:{lines.Count}").Trace
                },
                "FulfillmentLines",
                Tunable.ForItems(),
                ForEachFailureMode.ContinueOnError,
                item => item.AddStep<PackLineStep>("PackLine"))
            .If(
                data => data.FinalTotal >= FulfillmentDomainContext.Default.FreeShippingThreshold,
                then:  branch => branch.AddStep<SelectShippingStep>("SelectExpressShipping"),
                @else: branch => branch.AddStep<SelectShippingStep>("SelectStandardShipping"))
            .Gate(ResourceKind.Network,   data => { },
                gate => gate.AddStep<GenerateShippingLabelStep>("GenerateLabel"))
            .Gate(ResourceKind.Database,  data => { },
                gate => gate.AddStep<ArchiveOrderStep>("ArchiveOrder"))
            .Run(out commerceFulfillPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 5c · CommerceOrchestration
//  Composes the two sub-pipelines as recursive pipeline dependencies.
//  The output of CommercePricing is the input to CommerceFulfillment.
//  Teaches: pipeline composition, recursive data flow, cross-domain orchestration.
//  Flow: PriceOrder(→ commercePricingPipeline) → FulfillOrder(→ commerceFulfillPipeline)
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Orchestration", NullGlobalContext.Instance)
        .DefineTask<CommerceWorkflowData>("RunCommerce")
            .AddStep("PriceOrder",   new PriceOrderStep(commercePricingPipeline))
            .AddStep("FulfillOrder", new FulfillOrderStep(commerceFulfillPipeline))
            .Run(out orchestrationPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 6 · AdvancedPatterns
//  Full middleware stack, Materialize, SubTask, Fallback, ForEach, WindowBudget,
//  CPU gate (SHA-256 digest), and DiskIO gate (temp-file snapshot).
//  Teaches: middleware pipeline, adaptive tuning, fault-tolerance via fallback.
//  Flow: [TraceMiddleware → RetryOnce → TimeoutGuard] →
//        SeedMeta → Materialize → SubTask(StampMeta) →
//        Gate:Network(FetchQuote+Fallback) → ForEach(TransformItem) →
//        WindowBudget → Gate:Cpu(ComputeDigest) → Gate:DiskIO(PersistSnapshot)
// ─────────────────────────────────────────────────────────────────────────────
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
                subTask => subTask.AddStep("StampMeta",
                    meta => meta with { Stage = "Prepared", LastUpdatedUtc = DateTime.UtcNow }))
            .Gate(ResourceKind.Network, _ => { },
                gate => gate.AddStepWithFallback(
                    "FetchQuote",
                    primary: data =>
                    {
                        if (data.ForcePrimaryQuoteFailure)
                            throw new InvalidOperationException("Primary quote endpoint unavailable.");
                        return data.AppendTrace("Quote:Primary") with { Quote = 125m, QuoteSource = "primary" };
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
                    var errorCount  = Math.Max(0, (data.MaterializedItems?.Count ?? 0) - transformed.Count);
                    return data.AppendTrace($"ForEach:ContinueOnError:{transformed.Count}") with
                    {
                        MaterializedItems = transformed,
                        SuccessCount      = transformed.Count,
                        ErrorCount        = errorCount
                    };
                },
                "TransformItems",
                Tunable.ForItems(),
                ForEachFailureMode.ContinueOnError,
                item => item.AddStep("TransformItem", value =>
                {
                    if (value < 0) throw new InvalidOperationException($"Negative item {value} is invalid.");
                    return value * 2;
                }))
            .WindowBudget(scope => scope.AddStep("BudgetAwareMarker",
                data => data.AppendTrace("WindowBudget:Applied")))
            .Gate(ResourceKind.Cpu,    _ => { },
                gate => gate.AddStep("ComputeDigest",
                    async (data, ct) => await AdvancedPatternHelpers.ComputeDigestAsync(data, ct)))
            .Gate(ResourceKind.DiskIO, _ => { },
                gate => gate.AddStep("PersistSnapshot",
                    async (data, ct) => await AdvancedPatternHelpers.PersistSnapshotAsync(data, ct)))
            .Run(out advancedPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  DOMAIN 7 · ApiSurface
//  Full API coverage lab — exercises every builder surface in a single task.
//  WithEvents, WithStepFactory (DI-resolved step), AddParallelGroup (custom merge),
//  AddReadOnlyBridge, BeginSaga / EndSaga (Materialize + ForEach + StepWithCompensation
//  + AddGate with compensation), Pressure scope, WindowBudget scope.
//  Flow: FactoryResolved → ParallelGroup(L+R, CustomMerge) → ReadOnlyBridge →
//        Saga[ Materialize → ForEach(ScaleItem) → StepWithCompensation(Reserve) →
//              Gate:Network(SagaCall) ] → Pressure(Marker) → WindowBudget(Marker) →
//        Finalize
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Coverage", NullGlobalContext.Instance)
        .DefineTask<ApiSurfaceData>("RunApiSurface")
            .WithEvents(new ApiSurfaceEvents(eventLog))
            .AddStep<FactoryResolvedStep>("FactoryResolved")
            .AddParallelGroup(
                parallel => parallel
                    .AddBranch("ParallelLeft",  data => data with { ParallelLeft  = data.Input + 10 })
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
                    select:      data => data.SagaItems ?? [],
                    parallelism: Tunable.ForItems(2),
                    configure:   item => item.AddStep("ScaleItem", value => value * 10),
                    merge: (data, items) => data.AppendTrace($"Saga:ForEach:{items.Count}") with
                    {
                        ProcessedSagaItems = items.ToList()
                    })
                .AddStepWithCompensation(
                    "SagaReserve",
                    forward:    data => data.AppendTrace("Saga:Reserve") with
                    {
                        SagaCounter = data.SagaCounter + 10
                    },
                    compensate: data => data.AppendTrace("Saga:Reserve:Compensate") with
                    {
                        SagaCounter   = data.SagaCounter - 10,
                        SagaCompensated = true
                    })
                .AddGate(
                    ResourceKind.Network,
                    onWaiting: null,
                    configure: gate => gate.AddStep("SagaNetworkCall", data =>
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
            .Pressure("tutorial-pressure",
                pressure => pressure.AddStep("PressureMarker",
                    data => data.AppendTrace("Pressure:CustomScope") with { PressureScoped = true }))
            .WindowBudget(window => window.AddStep("WindowBudgetMarker",
                data => data.AppendTrace("Pressure:WindowBudgetScope")))
            .AddStep("Finalize", data => data.AppendTrace("ApiSurface:Completed"))
            .Run(out apiSurfacePipeline)

    .DefineDomain("Supplemental", NullGlobalContext.Instance)
        .DefineTask<ApiSurfaceData>("SupplementalTask")
            .AddStep("SupplementalNoOp", data => data.AppendTrace("Supplemental:NoOp"))
            .Run()
        .Build();

// ═══════════════════════════════════════════════════════════════════════════════
//  Execution
// ═══════════════════════════════════════════════════════════════════════════════

Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║   EvalApp SOLID Starter Tutorial - All 7 Features Demo        ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// 1 · RulesEngine
Console.WriteLine("📋 [1/7] RulesEngine: Pure Logic & Business Rules");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var items = ImmutableList.Create(
        new Item("TSHIRT-1",  "Blue T-Shirt",    45m,  ItemCategory.Standard),
        new Item("JEANS-1",   "Denim Jeans",     120m, ItemCategory.Standard),
        new Item("HAT-CLEAR", "Summer Hat",      15m,  ItemCategory.Clearance));
    var shopper  = new ShopperProfile("SHOPPER-42", 12, 5000m, IsVip: true);
    var order    = new OrderContext(shopper, items, "SUMMER20");
    var result   = await rulesPipeline.RunAsync(new PricingData(order));
    var data     = result is PipelineResult<PricingData>.Success s ? s.Data : ((PipelineResult<PricingData>.Failure)result).Data;
    Console.WriteLine($"  Shopper: {shopper.CustomerId} (VIP: {shopper.IsVip})");
    Console.WriteLine($"  Net: ${data.NetPrice:F2}  Discount: {data.DiscountPercent:P0}  Final: ${data.FinalPrice:F2}");
    Console.WriteLine($"✅ RulesEngine completed. Final price: {data.FinalPrice:C}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ RulesEngine failed: {ex.Message}\n"); }

// 2 · BatchSync
Console.WriteLine("📦 [2/7] BatchSync: Async I/O & Partial Failure Handling");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var result = await batchPipeline.RunAsync(new BatchSyncData(new List<int>()));
    var data   = result is PipelineResult<BatchSyncData>.Success bs ? bs.Data : ((PipelineResult<BatchSyncData>.Failure)result).Data;
    Console.WriteLine($"  Processed: {data.ItemIds.Count}  Success: {data.SuccessCount}  Failed: {data.ErrorCount}");
    Console.WriteLine($"✅ BatchSync completed. Success: {data.SuccessCount}, Failed: {data.ErrorCount}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ BatchSync failed: {ex.Message}\n"); }

// 3 · Ingestion
Console.WriteLine("🔄 [3/7] Ingestion: Stream Processing & Validation");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var rawItems = new List<RawRecord>
    {
        new(1, "Item-A", 100m), new(2, "Item-B", 250m),
        new(3, "", 150m),       // Invalid: empty name
        new(4, "Item-D", -50m), // Invalid: negative amount
        new(5, "Item-E", 75m),  new(6, "Item-F", 200m)
    };
    var result = await ingestionPipeline.RunAsync(new IngestionData(rawItems));
    var data   = result is PipelineResult<IngestionData>.Success ins ? ins.Data : ((PipelineResult<IngestionData>.Failure)result).Data;
    Console.WriteLine($"  Total: {data.TotalProcessed}  Valid: {data.SuccessCount}  Invalid: {data.ErrorCount}");
    Console.WriteLine($"✅ Ingestion completed. Valid: {data.SuccessCount}, Invalid: {data.ErrorCount}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Ingestion failed: {ex.Message}\n"); }

// 4 · OrderSaga
Console.WriteLine("💳 [4/7] OrderSaga: Distributed Transactions & Compensation");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var order = new OrderSagaData(
        OrderId:    "ORD-12345",
        Items:      new List<LineItem> { new("SKU-LAPTOP", 1), new("SKU-MOUSE", 2) },
        CustomerId: "CUST-ABC");
    var result = await sagaPipeline.RunAsync(order);
    var data   = result is PipelineResult<OrderSagaData>.Success ss ? ss.Data : ((PipelineResult<OrderSagaData>.Failure)result).Data;
    Console.WriteLine($"  Order: {data.OrderId}  State: {data.State}  Charge: ${data.ChargeAmount:F2}");
    Console.WriteLine($"✅ OrderSaga completed. Order: {data.OrderId}, Status: {data.State}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ OrderSaga failed: {ex.Message}\n"); }

// 5 · Commerce Orchestration (sub-pipelines composed recursively)
Console.WriteLine("🧭 [5/7] Commerce Orchestration: Multiple Domains & Pipeline Composition");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var order = new OrderContext(
        Shopper: new ShopperProfile("SHOP-99", 24, 6200m, IsVip: true),
        Items:   ImmutableList.Create(
            new Item("SKU-PHONE", "Phone",          699m, ItemCategory.Premium),
            new Item("SKU-CASE",  "Phone Case",      29m, ItemCategory.Standard),
            new Item("SKU-CLEAR", "Clearance Cable", 12m, ItemCategory.Clearance)),
        PromotionCode: "VIPSHIP");
    var result = await orchestrationPipeline.RunAsync(new CommerceWorkflowData(order, ImmutableList<CommerceLineItem>.Empty));
    var data   = result is PipelineResult<CommerceWorkflowData>.Success cs ? cs.Data : ((PipelineResult<CommerceWorkflowData>.Failure)result).Data;
    Console.WriteLine($"  Net: {data.NetTotal:C}  Discount: {data.Discount:C}  Tax: {data.Tax:C}");
    Console.WriteLine($"  Shipping: {data.ShippingMethod} / {data.Shipping:C}  Label: {data.LabelId}");
    Console.WriteLine($"✅ Commerce orchestration completed. Final total: {data.FinalTotal:C}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Commerce orchestration failed: {ex.Message}\n"); }

// 6 · Advanced Patterns
Console.WriteLine("🧪 [6/7] Advanced Patterns: Tuning, Fallback, Materialize, Middleware, CPU/Disk Gates");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var input  = new AdvancedDemoData(InputItems: [2, 4, -1, 8], ForcePrimaryQuoteFailure: true);
    var result = await advancedPipeline.RunAsync(input);
    var data   = result is PipelineResult<AdvancedDemoData>.Success ads ? ads.Data : ((PipelineResult<AdvancedDemoData>.Failure)result).Data;
    Console.WriteLine($"  Quote: {data.Quote:C} ({data.QuoteSource})  Items: {data.SuccessCount} ok / {data.ErrorCount} err");
    Console.WriteLine($"  Digest: {data.CpuDigest?[..Math.Min(12, data.CpuDigest?.Length ?? 0)]}...");
    Console.WriteLine($"✅ Advanced patterns completed. Snapshot: {data.SnapshotPath}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Advanced patterns failed: {ex.Message}\n"); }

// 7 · API Surface
Console.WriteLine("🧩 [7/7] API Surface: Events, Pressure, Parallel Group, ReadOnlyBridge, Saga APIs");
Console.WriteLine("─────────────────────────────────────────────────");
try
{
    var result = await apiSurfacePipeline.RunAsync(new ApiSurfaceData(Input: 5));
    var data   = result is PipelineResult<ApiSurfaceData>.Success aps ? aps.Data : ((PipelineResult<ApiSurfaceData>.Failure)result).Data;
    Console.WriteLine($"  Parallel: L={data.ParallelLeft}, R={data.ParallelRight}  Bridged: {data.BridgedValue}");
    Console.WriteLine($"  Saga Counter: {data.SagaCounter}  Pressure Scoped: {data.PressureScoped}");
    Console.WriteLine($"  Events captured: {eventLog.Count}");
    Console.WriteLine($"✅ API surface completed.\n");
}
catch (Exception ex) { Console.WriteLine($"❌ API surface failed: {ex.Message}\n"); }

Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║                   All Features Completed ✨                   ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");
