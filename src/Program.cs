using EvalApp.Consumer;
using EvalApp.Solid.Starter.Platform;
using EvalApp.Solid.Starter.Platform.Events;
using EvalApp.Solid.Starter.Platform.Merge;
using EvalApp.Solid.Starter.Platform.Steps;
using EvalApp.Solid.Starter.Platform.Support;
using EvalApp.Solid.Starter.Analytics;
using EvalApp.Solid.Starter.Analytics.Middleware;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;
using EvalApp.Solid.Starter.Commerce.Steps;
using EvalApp.Solid.Starter.Pricing.Context;
using EvalApp.Solid.Starter.Orders.Steps;
using EvalApp.Solid.Starter.Accounting;
using EvalApp.Solid.Starter.Catalog;
using System.Collections.Immutable;

// ═══════════════════════════════════════════════════════════════════════════════
//  Northstar Commerce Platform — Application Manifest
//
//  Single unified Eval.App declaration orchestrating all business domains:
//  Pricing, Orders, Catalog, Accounting, Commerce, Analytics, Platform.
//
//  This file is the system's source of truth. Resource management, tuning,
//  and cross-domain orchestration are all visible here.
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
//  PRICING DOMAIN
//  Deterministic discount and tax calculation. All rules are pure (no I/O).
//  Policy adjustments flow through context injection — no code changes needed.
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
//  ACCOUNTING DOMAIN
//  Nightly partner settlement reconciliation. Partial success is normal.
//  Retry/failure tracking support controlled rollback and audit workflows.
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Processing")
        .DefineTask<BatchSyncData>("SyncBatch")
            .AddStep("FetchItems",       new FetchItemsStep())
            .AddStep("ProcessBatch",     new ProcessBatchStep(successRate: 0.85))
            .AddStep("CalculateSummary", new CalculateSummaryStep())
            .Run(out batchPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  CATALOG DOMAIN
//  Real-time product feed validation. Bad records are quarantined, good ones flow.
//  Error-as-data model allows single-pass processing of mixed validity.
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("BatchProcessing")
        .DefineTask<IngestionData>("ProcessStream")
            .AddStep("Materialize",    new MaterializeStep())
            .AddStep("ProcessAllItems", new ProcessAllItemsStep())
            .AddStep("Summarize",      new SummarizeResultsStep())
            .Run(out ingestionPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  ORDERS DOMAIN
//  Three-stage distributed transaction: reserve → charge → ship.
//  Each step records IDs for controlled rollback if any stage fails.
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
//  COMMERCE · Pricing Subdomain
//  Computes net total, discounts, tax for CommerceWorkflowData.
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("CommercePricing", PricingDomainContext.Default)
        .DefineTask<CommerceWorkflowData>("PriceOrder")
            .AddStep<CalculateQuoteStep>("CalculateQuote")
            .Run(out commercePricingPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  COMMERCE · Fulfillment Subdomain
//  Prepares lines, applies per-item packaging rules, selects shipping method.
//  Parallelizes line packing; branches on order total; gates external integrations.
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
//  COMMERCE DOMAIN
//  End-to-end order flow: price → fulfill. Composes two sub-pipelines
//  demonstrating recursive pipeline dependencies and cross-domain orchestration.
// ─────────────────────────────────────────────────────────────────────────────
    .DefineDomain("Orchestration", NullGlobalContext.Instance)
        .DefineTask<CommerceWorkflowData>("RunCommerce")
            .AddStep("PriceOrder",   new PriceOrderStep(commercePricingPipeline))
            .AddStep("FulfillOrder", new FulfillOrderStep(commerceFulfillPipeline))
            .Run(out orchestrationPipeline)

// ─────────────────────────────────────────────────────────────────────────────
//  ANALYTICS DOMAIN
//  Real-time metrics collection under load. Demonstrates resilient quoting
//  with middleware, fallback paths, resource gates, and ForEach failure modes.
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
//  PLATFORM DOMAIN
//  Internal validation suite exercising all EvalApp builder surfaces in one task.
//  Service factory resolution, saga compensation, tuning variants, resource gates,
//  pressure scopes, and event logging — all in a single controlled pipeline.
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
//  Demo Execution — Northstar Commerce Platform
// ═══════════════════════════════════════════════════════════════════════════════

Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║   Northstar Commerce — Platform Verification                 ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝\n");

// Pricing subsystem
Console.WriteLine("📋 Pricing: Discount eligibility and tax calculation");
Console.WriteLine("─────────────────────────────────────────────────────");
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
    Console.WriteLine($"✅ Completed. Final price: {data.FinalPrice:C}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Accounting subsystem
Console.WriteLine("📦 Accounting: Partner settlement reconciliation");
Console.WriteLine("─────────────────────────────────────────────────────");
try
{
    var result = await batchPipeline.RunAsync(new BatchSyncData(new List<int>()));
    var data   = result is PipelineResult<BatchSyncData>.Success bs ? bs.Data : ((PipelineResult<BatchSyncData>.Failure)result).Data;
    Console.WriteLine($"  Processed: {data.ItemIds.Count}  Success: {data.SuccessCount}  Failed: {data.ErrorCount}");
    Console.WriteLine($"✅ Completed. Success: {data.SuccessCount}, Failed: {data.ErrorCount}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Catalog subsystem
Console.WriteLine("🔄 Catalog: Product intake and validation");
Console.WriteLine("─────────────────────────────────────────────────────");
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
    Console.WriteLine($"✅ Completed. Valid: {data.SuccessCount}, Invalid: {data.ErrorCount}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Orders subsystem
Console.WriteLine("💳 Orders: Distributed transaction (inventory → payment → shipment)");
Console.WriteLine("─────────────────────────────────────────────────────");
try
{
    var order = new OrderSagaData(
        OrderId:    "ORD-12345",
        Items:      new List<LineItem> { new("SKU-LAPTOP", 1), new("SKU-MOUSE", 2) },
        CustomerId: "CUST-ABC");
    var result = await sagaPipeline.RunAsync(order);
    var data   = result is PipelineResult<OrderSagaData>.Success ss ? ss.Data : ((PipelineResult<OrderSagaData>.Failure)result).Data;
    Console.WriteLine($"  Order: {data.OrderId}  State: {data.State}  Charge: ${data.ChargeAmount:F2}");
    Console.WriteLine($"✅ Completed. Order: {data.OrderId}, Status: {data.State}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Commerce orchestration
Console.WriteLine("🧭 Commerce: End-to-end order (pricing + fulfillment)");
Console.WriteLine("─────────────────────────────────────────────────────");
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
    Console.WriteLine($"✅ Completed. Final total: {data.FinalTotal:C}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Analytics subsystem
Console.WriteLine("🧪 Analytics: Quote retrieval with resilience and tuning");
Console.WriteLine("─────────────────────────────────────────────────────");
try
{
    var input  = new AdvancedDemoData(InputItems: [2, 4, -1, 8], ForcePrimaryQuoteFailure: true);
    var result = await advancedPipeline.RunAsync(input);
    var data   = result is PipelineResult<AdvancedDemoData>.Success ads ? ads.Data : ((PipelineResult<AdvancedDemoData>.Failure)result).Data;
    Console.WriteLine($"  Quote: {data.Quote:C} ({data.QuoteSource})  Items: {data.SuccessCount} ok / {data.ErrorCount} err");
    Console.WriteLine($"  Digest: {data.CpuDigest?[..Math.Min(12, data.CpuDigest?.Length ?? 0)]}...");
    Console.WriteLine($"✅ Completed. Snapshot: {data.SnapshotPath}\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

// Platform validation
Console.WriteLine("🧩 Platform: Internal capability verification");
Console.WriteLine("─────────────────────────────────────────────────────");
try
{
    var result = await apiSurfacePipeline.RunAsync(new ApiSurfaceData(Input: 5));
    var data   = result is PipelineResult<ApiSurfaceData>.Success aps ? aps.Data : ((PipelineResult<ApiSurfaceData>.Failure)result).Data;
    Console.WriteLine($"  Parallel: L={data.ParallelLeft}, R={data.ParallelRight}  Bridged: {data.BridgedValue}");
    Console.WriteLine($"  Saga Counter: {data.SagaCounter}  Pressure Scoped: {data.PressureScoped}");
    Console.WriteLine($"  Events captured: {eventLog.Count}");
    Console.WriteLine($"✅ Completed.\n");
}
catch (Exception ex) { Console.WriteLine($"❌ Failed: {ex.Message}\n"); }

Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║                   Platform Verified ✨                       ║");
Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");

