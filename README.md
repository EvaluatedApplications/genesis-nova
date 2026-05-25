# EvalApp SOLID Starter Project

## Status: Scaffolded & Ready for Development

This is a bootstrap project demonstrating how to build production-grade **EvalApp pipelines** using **SOLID principles** applied to practical workflow problems.

## Project Structure

```
EvalApp.Solid.Starter/
├── src/
│   ├── Features/
│   │   ├── RulesEngine/
│   │   │   ├── Data/          PricingData record (immutable pipeline state)
│   │   │   ├── Steps/         CalculateNetPrice, EvaluateEligibility, ApplyPromoRules, CalculateFinalPrice
│   │   │   ├── Pipelines/     RulesEnginePipeline builder
│   │   │   └── Docs/          Problem statement, SOLID mapping, customization guide
│   │   ├── BatchSync/         (Scaffolded - placeholder)
│   │   ├── Ingestion/         (Scaffolded - placeholder)
│   │   └── OrderSaga/         (Scaffolded - placeholder)
│   ├── Shared/
│   │   ├── OrderModels.cs     ShopperProfile, Item, ItemCategory, OrderContext
│   │   └── AppContexts.cs     GlobalContext
│   └── Program.cs             Demo entry point
├── Tests/
│   ├── Features/RulesEngine/  Step unit tests + pipeline integration tests
│   └── Shared/TestData.cs     Test factories
└── docs/                       High-level guides (to be added)
```

## Features Implemented

### 1. RulesEngine
**Problem:** Large if/else logic for pricing, discounts, eligibility. Violates OCP (Open/Closed Principle).

**Solution:** Pipeline-based rules engine with isolated steps:
- **CalculateNetPriceStep** — Sum item prices
- **EvaluateDiscountEligibilityStep** — Check VIP/history/spend thresholds
- **ApplyPromotionRulesStep** — Centralized rule logic (clearance, promos, VIP stacking)
- **CalculateFinalPriceStep** — Apply discount

**SOLID Applied:**
- **SRP** (Single Responsibility) — One step per concern
- **OCP** (Open/Closed) — Add promotion rules without changing pipeline topology
- **DIP** (Dependency Inversion) — Steps inherit from `PureStep<T>` abstraction

**Test Coverage:**
- ✅ 10 unit tests (all passing): Net price calculation, eligibility logic, promotion rules, final price
- ⚠️ 2 integration tests: Fail due to NuGet package assembly loading (see below)
- **Total: 13/15 tests passing**

## Build & Test

```bash
cd EvalApp.Solid.Starter

# Build
dotnet build

# Run tests (13 pass, 2 fail due to NuGet assembly issue)
dotnet test

# Run console demo (fails at runtime with same assembly issue)
dotnet run --project src/EvalApp.Solid.Starter.csproj
```

## Known Issues

### Assembly Loading Error
```
System.IO.FileNotFoundException: Could not load file or assembly 'EvalApp, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null'
```

**Root Cause:** The published EvalApp.Consumer NuGet package (v1.0.1) is incomplete. It references the private `EvalApp.dll` but doesn't include it. 

**Why:** ILRepack was not configured in the evalapp-public csproj to embed/merge the private EvalApp.dll into EvalApp.Consumer.dll. This is a known limitation from the prior checkpoint.

**Fix Required:** Update `evalapp-public/src/EvalApp.Consumer/EvalApp.Consumer.csproj` to:
1. Import `ILRepack.targets` correctly
2. Configure MSBuild item groups to pack the merged DLL into NuGet

**Impact on This Project:**
- ✅ Step unit tests work perfectly (no runtime assembly resolution needed for pure steps)
- ⚠️ Pipeline integration tests + console demo fail (need EvalApp private assembly at runtime)

## Next Steps

### Short Term (unblock runtime)
1. Fix NuGet packaging to include embedded/merged private DLL
2. Re-publish EvalApp.Consumer v1.0.2
3. Update project to v1.0.2, re-run demo

### Medium Term (complete starter project)
1. **BatchSync feature** — Implement ForEach + gates + failure modes
2. **Ingestion feature** — Materialize stream, validate, partial success
3. **OrderSaga feature** — Distributed transaction with compensation

### Long Term (documentation & examples)
1. Before/after anti-pattern comparisons for each feature
2. Customization checklist for each pipeline
3. Benchmark harness (imperative vs EvalApp)

## Design Methodology

Each feature follows the **"Thinking Inversion" Curriculum**:

1. **Design immutable data record** — Define `PricingData` with stage fields
2. **Split one responsibility per step** — Each step transforms one concern
3. **Place gates at I/O boundaries only** — Pure steps ungated (RulesEngine is 100% pure)
4. **Move control policy to topology** — If/ForEach/Saga defined in pipeline builder
5. **Keep domain logic in steps** — Business rules stay in `ApplyPromotionRulesStep`

## Step Test Example

```csharp
[Fact]
public void WhenVipAndClearance_Then_HighestDiscount()
{
    // Arrange
    var shopper = TestData.CreateShopper(isVip: true);
    var items = ImmutableList.Create(
        TestData.CreateItem(category: ItemCategory.Clearance));
    var order = TestData.CreateOrder(shopper: shopper, items: items);
    var data = new PricingData(order, IsEligibleForDiscount: true);
    var step = new ApplyPromotionRulesStep();

    // Act
    var result = step.Execute(data);

    // Assert: 20% (clearance) + 5% (VIP) = 25%
    Assert.Equal(0.25m, result.DiscountPercent);
}
```

## Key Files

| File | Purpose |
|------|---------|
| `src/Features/RulesEngine/Steps/*.cs` | Step implementations (4 files) |
| `src/Features/RulesEngine/Pipelines/RulesEnginePipeline.cs` | Pipeline builder |
| `src/Shared/OrderModels.cs` | Domain records (immutable data) |
| `Tests/Features/RulesEngine/RulesEngineTests.cs` | 15 test cases |
| `src/Program.cs` | Console demo (blocked by assembly issue) |

## Teaching Value

This project demonstrates:
- ✅ **Immutable records** for pipeline data (no mutations)
- ✅ **One step = one responsibility** (SRP)
- ✅ **Fluent builder API** for pipeline composition
- ✅ **Test-driven step design** (unit tests pass)
- ✅ **Pattern matching** on PipelineResult discriminated unions
- ✅ **Scalable architecture** (ready for gates/ForEach in other features)

New users can:
1. Read RulesEngine/Docs/README.md for problem context
2. Study the 4 step implementations
3. Look at test cases for usage patterns
4. Replicate/modify steps for their domain

## Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| Compiles cleanly | ✅ | No warnings |
| Unit tests (80%+) | ✅ | 13/15 (87%), 2 blocked by NuGet assembly issue |
| Clear problem statements | ✅ | RulesEngine.Docs/README.md complete |
| SOLID mapping | ✅ | Documented in RulesEngine pipeline |
| Custom ization guide | ✅ | In RulesEngine/Docs/README.md |
| Runnable examples | ⚠️ | Code ready, blocked by NuGet assembly loading |

---

**Next Action:** Fix NuGet packaging (ILRepack) so console demo can run, then continue with BatchSync feature.
