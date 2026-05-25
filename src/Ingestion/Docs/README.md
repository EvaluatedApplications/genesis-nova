# Ingestion Feature — Stream-to-Batch Processing with Partial Success

## Problem Statement

Ad hoc ingestion pipelines often struggle with:
- **Data loss** — If one item fails validation, do we lose the entire batch or skip that item silently?
- **Error context** — When an item fails, how do we report *which* item and *why*?
- **Inconsistent handling** — Mixing validation logic, transformation, and error collection in one method leads to spaghetti code.

Typical code smells:
```csharp
// ❌ Antipattern: mixed responsibilities
var validItems = new List<ValidatedRecord>();
var errors = new List<string>();
try
{
    foreach (var raw in stream)
    {
        if (string.IsNullOrWhiteSpace(raw.Name))
        {
            errors.Add($"Item {raw.Id}: Name required");
        }
        else if (raw.Amount <= 0)
        {
            errors.Add($"Item {raw.Id}: Amount must be positive");
        }
        else
        {
            validItems.Add(new ValidatedRecord(raw.Id, raw.Name, raw.Amount, DateTime.UtcNow));
        }
    }
}
catch (Exception ex)
{
    errors.Add($"Unexpected error: {ex.Message}");
}

Console.WriteLine($"Processed: {validItems.Count} valid, {errors.Count} invalid");
```

This approach conflates:
- Validation logic (spread through if/else)
- Transformation logic (building ValidatedRecord inline)
- Error handling (strings added to list)

**EvalApp solution**: Separate concerns into focused steps, use immutable records for clean data flow, and let the pipeline orchestrate the sequence.

---

## EvalApp Solution

### Data Models

Three records model the domain:

```csharp
public record RawRecord(int Id, string Name, decimal Amount);
public record ValidatedRecord(int Id, string Name, decimal Amount, DateTime ProcessedAt);
public record ValidationError(int Id, string Reason);
public record IngestionData(
    List<RawRecord> InputStream,
    List<ValidatedRecord>? ValidItems = null,
    List<ValidationError>? InvalidItems = null,
    int TotalProcessed = 0,
    int SuccessCount = 0,
    int ErrorCount = 0,
    string? Summary = null);
```

**Key insight**: Both `ValidItems` and `InvalidItems` are populated together. Partial success means we don't abandon the entire batch if some items fail.

### Pipeline Topology

```
InputStream (raw records)
    ↓
[Materialize] — initialize ValidItems, InvalidItems collections
    ↓
[ProcessAllItems] — iterate stream, validate each, collect results
    ├─ ValidateItemStep: Name.Length > 0 && Amount > 0
    ├─ ProcessItemStep: Add ProcessedAt timestamp
    └─ Collect errors for invalid items
    ↓
ValidItems: [ValidatedRecord, ValidatedRecord, ...]
InvalidItems: [ValidationError, ValidationError, ...]
    ↓
[SummarizeResults] — count successes/errors, build summary
    ↓
IngestionData { SuccessCount, ErrorCount, Summary }
```

### Step Responsibilities

| Step | Input | Output | Responsibility |
|------|-------|--------|-----------------|
| **Materialize** | IngestionData with InputStream | IngestionData with empty ValidItems/InvalidItems | Prepare for iteration |
| **ProcessAllItems** | IngestionData | IngestionData with populated ValidItems/InvalidItems | Iterate, validate, transform, collect errors |
| **SummarizeResults** | IngestionData | IngestionData with SuccessCount, ErrorCount, Summary | Aggregate outcomes |

### SOLID Mapping

#### Single Responsibility Principle (SRP)
Each step has **one reason to change**:
- **Materialize** — only if output collection initialization changes
- **ProcessAllItems** — only if the iteration/validation/transformation logic changes
- **SummarizeResults** — only if summary format changes

#### Open/Closed Principle (OCP)
New validation rules are **easy to add without changing the pipeline**:
```csharp
// In ValidateItemStep, add a new method:
public string? ValidateRecord(RawRecord record)
{
    // ... existing checks ...
    
    // NEW: Check for duplicate IDs (passed from external state)
    if (_existingIds.Contains(record.Id))
        return "Duplicate ID";
    
    return null; // Valid
}
```

The pipeline topology doesn't change — we just extend the validation step.

#### Liskov Substitution Principle (LSP)
All steps are `PureStep<IngestionData>`. Clients don't know or care which specific step is running — they trust that each returns consistent error contracts:
- Invalid items → `ValidationError` with reason
- Valid items → `ValidatedRecord` with timestamp
- Errors are **reported, never thrown** from steps

#### Dependency Inversion Principle (DIP)
Steps depend on the abstraction `PureStep<T>`, not concrete implementations. The pipeline composes them without tight coupling.

#### Interface Segregation Principle (ISP)
Each step has a minimal interface: just `Execute(data)`. No bloated dependencies.

---

## Before/After Comparison

### ❌ Manual Try/Catch Loop

```csharp
public class IngestionService
{
    public (List<ValidatedRecord> valid, List<string> errors) ProcessStream(List<RawRecord> items)
    {
        var valid = new List<ValidatedRecord>();
        var errors = new List<string>();

        foreach (var item in items)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(item.Name))
                    throw new ArgumentException("Name required");
                if (item.Amount <= 0)
                    throw new ArgumentException("Amount must be positive");

                valid.Add(new ValidatedRecord(item.Id, item.Name, item.Amount, DateTime.UtcNow));
            }
            catch (Exception ex)
            {
                errors.Add($"Item {item.Id}: {ex.Message}");
            }
        }

        return (valid, errors);
    }
}

// Usage: scattered across the codebase
var service = new IngestionService();
var (validRecords, errorMessages) = service.ProcessStream(rawItems);
// Now what? How do we count them? Report them? Store them?
```

**Problems**:
- Exception-based control flow (slow)
- Error reporting ad hoc
- No structured result
- Mixes concerns in one method
- Hard to test individual validation rules
- Hard to extend validation

### ✅ EvalApp Pipeline

```csharp
// Define data and steps — pure, testable, composable
public record IngestionData(...);
public class MaterializeStep : PureStep<IngestionData> { ... }
public class ProcessAllItemsStep : PureStep<IngestionData> { ... }
public class SummarizeResultsStep : PureStep<IngestionData> { ... }

// Assemble pipeline declaratively
var pipeline = IngestionPipeline.Build();

// Run it
var result = await pipeline.RunAsync(data);
var finalData = result.GetData();

// Result is structured and complete
Console.WriteLine($"Success: {finalData.SuccessCount}");
Console.WriteLine($"Errors: {finalData.ErrorCount}");
Console.WriteLine(finalData.Summary);
foreach (var error in finalData.InvalidItems)
{
    Console.WriteLine($"  Item {error.Id}: {error.Reason}");
}
```

**Benefits**:
- Clean separation of concerns
- Easy to test each step independently
- Structured result record
- Declarative pipeline is self-documenting
- Immutable data flow
- Errors are data, not exceptions
- Extensible without modifying existing code

---

## Customization Checklist

### Add a New Validation Rule

1. **Extend ValidateItemStep**:
   ```csharp
   public string? ValidateRecord(RawRecord record)
   {
       if (string.IsNullOrWhiteSpace(record.Name))
           return "Name cannot be empty";
       if (record.Amount <= 0)
           return "Amount must be greater than zero";
       
       // NEW RULE:
       if (record.Name.Length > 100)
           return "Name exceeds 100 characters";
       
       return null;
   }
   ```

2. **No pipeline changes needed** — ProcessAllItemsStep automatically calls the updated validation.

3. **Add test**:
   ```csharp
   [Fact]
   public void WhenNameTooLong_Then_ValidationFails()
   {
       var data = new IngestionData(new List<RawRecord>
       {
           new RawRecord(1, new string('x', 101), 100m)
       });
       var pipeline = IngestionPipeline.Build();
       var result = await pipeline.RunAsync(data);
       var finalData = result.GetData();
       Assert.Single(finalData.InvalidItems!);
       Assert.Contains("exceeds", finalData.InvalidItems[0].Reason);
   }
   ```

### Change Enrichment Logic

Modify ProcessItemStep to add new fields:

```csharp
public class ProcessItemStep : PureStep<IngestionData>
{
    public ValidatedRecord ProcessRecord(RawRecord record)
    {
        // NEW: Add processing region field based on name prefix
        var region = record.Name.StartsWith("US-") ? "USA" : "Other";
        
        return new ValidatedRecord(
            record.Id,
            record.Name,
            record.Amount,
            DateTime.UtcNow,
            Region: region);  // NEW field in ValidatedRecord
    }
}
```

Update the data record:
```csharp
public record ValidatedRecord(
    int Id, 
    string Name, 
    decimal Amount, 
    DateTime ProcessedAt,
    string Region = "Unknown");  // NEW
```

### Add Custom Summary Logic

Extend SummarizeResultsStep:

```csharp
public class SummarizeResultsStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        var summary = BuildCustomSummary(data);
        return data with { Summary = summary, ... };
    }
    
    private string BuildCustomSummary(IngestionData data)
    {
        if (data.ValidItems?.Count == 0)
            return $"❌ All {data.TotalProcessed} items failed";
        
        var totalAmount = data.ValidItems?.Sum(v => v.Amount) ?? 0m;
        return $"✅ {data.SuccessCount} items, ${totalAmount:F2} total (❌ {data.ErrorCount} failed)";
    }
}
```

### Parallel Processing (With License)

The current implementation uses synchronous processing. For licensed EvalApp, use ForEach:

```csharp
Eval.App("Ingestion")
    .DefineDomain("BatchProcessing")
        .DefineTask<IngestionData>("ProcessStream")
            .AddStep("Materialize", new MaterializeStep())
            .ForEach<RawRecord>(
                select: data => data.InputStream.AsEnumerable(),
                merge: (data, results) => MergeResults(data, results),
                collectionName: "RawRecords",
                maxParallelism: Environment.ProcessorCount,
                failureMode: ForEachFailureMode.ContinueOnError,
                configure: b => b
                    .AddStep("Validate", new ValidateItemStep())
                    .AddStep("Process", new ProcessItemStep()))
            .AddStep("Summarize", new SummarizeResultsStep())
            .Run(out pipeline)
        .Build();
```

---

## Testing Strategy

### Unit Tests (Steps)

Test each step in isolation:

```csharp
[Fact]
public void ValidateItemStep_WhenEmptyName_ReturnsError()
{
    var step = new ValidateItemStep();
    var record = new RawRecord(1, "", 100m);
    var error = step.ValidateRecord(record);
    Assert.NotNull(error);
    Assert.Contains("Name", error);
}
```

### Integration Tests (Pipeline)

Test the end-to-end flow:

```csharp
[Fact]
public async Task WhenMixed_Then_PartialSuccessWithReasons()
{
    var data = new IngestionData(new List<RawRecord>
    {
        new RawRecord(1, "Valid", 100m),
        new RawRecord(2, "", 100m),
        new RawRecord(3, "Also Valid", 50m)
    });
    
    var pipeline = IngestionPipeline.Build();
    var result = await pipeline.RunAsync(data);
    var finalData = result.GetData();
    
    Assert.Equal(2, finalData.ValidItems?.Count);
    Assert.Single(finalData.InvalidItems!);
    Assert.Contains("Partial success", finalData.Summary);
}
```

### Test Scenarios

✅ **All valid** — verify all items processed, success count = input count  
✅ **All invalid** — verify no valid items, error count = input count  
✅ **Mixed** — verify partial success, both collections populated  
✅ **Empty stream** — verify zero processed  
✅ **Individual validation rules** — Name, Amount constraints  
✅ **Timestamp enrichment** — verify ProcessedAt populated correctly  
✅ **Error reasons** — verify error messages are informative  

---

## Key Takeaways

1. **Immutable records** — All data is immutable; transformations return new instances via `with`.
2. **Separation of concerns** — Each step has one responsibility.
3. **Structured errors** — Errors are data (ValidationError), not exceptions.
4. **Partial success** — Both valid and invalid items are collected; the entire pipeline doesn't fail.
5. **Composability** — Steps can be reused, extended, or tested independently.
6. **Declarative pipeline** — The pipeline topology is self-documenting.
7. **Easy extensibility** — New validation rules, enrichment logic, or summaries don't require pipeline changes.

---

## References

- **Data Models** — `src/Ingestion/IngestionData.cs`
- **Steps** — `src/Ingestion/Steps/`
  - MaterializeStep
  - ValidateItemStep
  - ProcessItemStep
  - ProcessAllItemsStep
  - SummarizeResultsStep
- **Pipeline** — `src/Ingestion/Pipelines/IngestionPipeline.cs`
- **Tests** — `Tests/Features/Ingestion/`
- **Shared Contexts** — `src/AppContexts.cs` (GlobalContext, DomainContext — not used for CPU-bound processing)

---

## FAQ

**Q: Why not use exceptions for validation errors?**  
A: Exceptions are for *exceptional* conditions (out of memory, file not found). Validation failures are *expected* outcomes that should be handled as data. Using `ValidationError` records is faster and clearer.

**Q: How do I add async I/O?**  
A: Replace `PureStep<T>` with `SideEffectStep<T>` and mark methods `async`. Add gates (`ResourceKind.Network`, `ResourceKind.Database`) for the I/O steps.

**Q: Can I parallel process items?**  
A: Yes, with licensed EvalApp. Use `ForEach` with `maxParallelism: Environment.ProcessorCount`. The current implementation uses synchronous processing (max parallelism: 1) for clarity.

**Q: What if validation needs external state?**  
A: Use `ContextPureStep<TGlobal, TDomain, T>` instead of `PureStep<T>`. Inject the state via global or domain context.

**Q: How do I report results?**  
A: The final `IngestionData` record has `ValidItems`, `InvalidItems`, `SuccessCount`, `ErrorCount`, and `Summary`. Use these fields to render reports, logs, or API responses.
