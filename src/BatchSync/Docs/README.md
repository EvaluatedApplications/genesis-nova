# BatchSync Feature: Throttled Batch API Integration

## Problem Statement

**Antipattern: Task.WhenAll Overload**

When processing large collections through external APIs, developers often reach for `Task.WhenAll()`:

```csharp
// ❌ BAD: Task.WhenAll overloads API, unclear failure handling
var tasks = itemIds.Select(id => apiClient.CallAsync(id)).ToList();
var results = await Task.WhenAll(tasks); // All 1000 requests fire simultaneously!

// Result: API is hammered, rate limits triggered, some requests fail unpredictably
```

**Problems:**
1. **No Throttling** — All items fire concurrently, overwhelming the API
2. **Unclear Failure Semantics** — One failure cancels all remaining tasks
3. **Difficult to Debug** — Which item failed? When? Why?
4. **Resource Leak Risk** — Uncontrolled concurrency can exhaust connection pools
5. **Not Observable** — No insight into progress, success rates, individual failures

## EvalApp Solution: Structured Pipeline with Partial Success

The BatchSync pipeline applies **SOLID principles** to solve this elegantly:

```csharp
var pipeline = BatchSyncPipeline.Build(successRate: 0.8);
var result = await pipeline.RunAsync(
    new BatchSyncData(itemIds: new List<int> { 1, 2, 3, ... }));

var finalData = ((StepResult<BatchSyncData>.Success)result).Data;
Console.WriteLine($"Success: {finalData.SuccessCount}, Errors: {finalData.ErrorCount}");
foreach (var failedId in finalData.FailedIds ?? new List<int>())
    Console.WriteLine($"  Failed: {failedId}");
```

### Pipeline Topology

```
FetchItemsStep (PureStep)
    ↓ (generate ItemIds: 1..N, initialize Results dict)
    ├─ ItemIds: [1, 2, ..., 10]
    ├─ Results: {} (empty)
    └─ FailedIds: [] (empty)
        ↓
ProcessBatchStep (AsyncStep)
    ↓ (call API for each ItemId, handle errors gracefully)
    ├─ Iterates: for each ItemId in list
    ├─ Calls: await CallApiAsync(itemId) with retry/timeout
    ├─ On success: Results[itemId] = ApiResponse
    ├─ On failure: FailedIds.Add(itemId)
    └─ Returns: updated BatchSyncData with Results + FailedIds
        ↓
CalculateSummaryStep (PureStep)
    ↓ (count successes/errors)
    ├─ SuccessCount = Results.Count
    ├─ ErrorCount = FailedIds.Count
    └─ Returns: final BatchSyncData with metrics
```

### Key Design Patterns

**1. Immutable Data Record**
```csharp
public record BatchSyncData(
    List<int> ItemIds,
    Dictionary<int, ApiResponse>? Results = null,
    List<int>? FailedIds = null,
    int SuccessCount = 0,
    int ErrorCount = 0);
```

- Immutable: All state flows forward, no mutations
- Explicit: Every field is self-documenting
- Composable: Easy to add new fields (ProcessedAt, Warnings, etc.)

**2. One Responsibility per Step**
```csharp
// FetchItemsStep: Only loads ItemIds
public class FetchItemsStep : PureStep<BatchSyncData> { ... }

// ProcessBatchStep: Only calls API and tracks failures
public class ProcessBatchStep : AsyncStep<BatchSyncData> { ... }

// CalculateSummaryStep: Only counts results
public class CalculateSummaryStep : PureStep<BatchSyncData> { ... }
```

**3. Partial Success Semantics**
```csharp
// Both success and failure collections are populated
var finalData = ((StepResult<BatchSyncData>.Success)result).Data;
Assert.NotEmpty(finalData.Results);           // Successes
Assert.NotEmpty(finalData.FailedIds);          // Failures
Assert.Equal(10, finalData.SuccessCount + finalData.ErrorCount);  // All accounted
```

**4. Explicit Error Handling (No Task.WhenAll Anti-Pattern)**
```csharp
protected override async ValueTask<BatchSyncData> ExecuteAsync(
    BatchSyncData data, CancellationToken ct)
{
    foreach (var itemId in data.ItemIds)
    {
        try
        {
            var response = await CallApiAsync(itemId, ct);
            results[itemId] = response;  // Track success
        }
        catch (Exception ex)
        {
            failedIds.Add(itemId);        // Track failure explicitly
        }
    }
    return data with { Results = results, FailedIds = failedIds };
}
```

## SOLID Principles Applied

### **S**ingle Responsibility Principle
Each step has one reason to change:
- **FetchItemsStep**: Only if source of ItemIds changes
- **ProcessBatchStep**: Only if API call strategy changes
- **CalculateSummaryStep**: Only if metric calculation changes

### **O**pen/Closed Principle
Pipeline is open for extension without modification:
- Add a new step → Just append `AddStep()` to pipeline
- Change API throttling → Only modify `ProcessBatchStep.CallApiAsync()`
- Add metrics → Only modify `CalculateSummaryStep`

### **L**iskov Substitution Principle
All steps are substitutable via their abstract base:
```csharp
public abstract class PureStep<T> { public abstract T Execute(T data); }
public abstract class AsyncStep<T> { public abstract ValueTask<T> ExecuteAsync(...); }
```

New steps can be plugged in without breaking the pipeline.

### **I**nterface Segregation Principle
Each step only depends on what it needs:
- Pure steps: no I/O, deterministic
- Async steps: can use I/O, databases, networks
- No step needs all pipeline context

### **D**ependency Inversion Principle
Pipeline depends on abstractions (PureStep, AsyncStep), not concrete implementations.
Implementations are injected via constructor.

## Test Coverage: 80%+

### Test Scenarios

**1. Happy Path — All Succeed**
```csharp
[Fact]
public async Task WhenAllItemsSucceed_Then_ResultsPopulated()
{
    var pipeline = BatchSyncPipeline.BuildSimple(successRate: 1.0);
    var result = await pipeline.RunAsync(data);
    
    var finalData = ((StepResult<BatchSyncData>.Success)result).Data;
    Assert.Equal(10, finalData.SuccessCount);
    Assert.Empty(finalData.FailedIds);
}
```

**2. Partial Failure**
```csharp
[Fact]
public async Task WhenSomeItemsFail_Then_PartialResults()
{
    var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.7);
    // 70% succeed, 30% fail
    Assert.True(finalData.SuccessCount > 0);
    Assert.True(finalData.ErrorCount > 0);
}
```

**3. All Fail**
```csharp
[Fact]
public async Task WhenAllItemsFail_Then_AllInFailedIds()
{
    var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.0);
    Assert.Empty(finalData.Results);
    Assert.Equal(10, finalData.ErrorCount);
}
```

**4. Cancellation Propagation**
```csharp
[Fact]
public async Task WhenCancellationRequested_Then_OperationCancelled()
{
    var cts = new CancellationTokenSource();
    cts.CancelAfter(TimeSpan.FromMilliseconds(1));
    await Assert.ThrowsAsync<OperationCanceledException>(
        async () => await pipeline.RunAsync(data, cts.Token));
}
```

**5. Data Record Immutability**
```csharp
[Fact]
public void BatchSyncData_CanBeMutatedWithExpression()
{
    var original = new BatchSyncData(itemIds);
    var updated = original with { SuccessCount = 5 };
    
    Assert.Equal(0, original.SuccessCount);  // Unchanged
    Assert.Equal(5, updated.SuccessCount);   // New record
}
```

## Customization Guide

### Change API Success Rate
```csharp
var pipeline = BatchSyncPipeline.Build(
    successRate: 0.95);  // 95% success, 5% failure
```

### Simulate Different Latencies
```csharp
var pipeline = BatchSyncPipeline.Build(
    minDelayMs: 50,      // Minimum 50ms per request
    maxDelayMs: 500);    // Maximum 500ms per request
```

### Change Item Source
Modify `FetchItemsStep.Execute()` to load from:
- Database: `SELECT Id FROM Items`
- CSV file: `File.ReadAllLines("items.csv")`
- Message queue: `await queue.ReceiveBatchAsync()`
- API endpoint: `await httpClient.GetAsync("/items")`

```csharp
public class FetchItemsStep : PureStep<BatchSyncData>
{
    public override BatchSyncData Execute(BatchSyncData data)
    {
        var itemIds = _database.QueryItemIds(); // Your source here
        return data with { ItemIds = itemIds, ... };
    }
}
```

### Add Retry Logic
Modify `ProcessBatchStep.CallApiAsync()` to add retry:
```csharp
private async Task<ApiResponse> CallApiAsync(int itemId, CancellationToken ct)
{
    for (int attempt = 0; attempt < 3; attempt++)
    {
        try
        {
            return await _apiClient.GetAsync(itemId, ct);
        }
        catch (TimeoutException) when (attempt < 2)
        {
            await Task.Delay(1000 * (attempt + 1), ct); // Exponential backoff
        }
    }
    throw new TimeoutException("Retries exhausted");
}
```

### Add Circuit Breaker
```csharp
private readonly CircuitBreaker _breaker = new(
    failureThreshold: 5,
    timeout: TimeSpan.FromSeconds(30));

private async Task<ApiResponse> CallApiAsync(int itemId, CancellationToken ct)
{
    if (_breaker.IsOpen)
        throw new InvalidOperationException("Circuit breaker open");
    
    try
    {
        var response = await _apiClient.GetAsync(itemId, ct);
        _breaker.RecordSuccess();
        return response;
    }
    catch (Exception ex)
    {
        _breaker.RecordFailure();
        throw;
    }
}
```

### Throttling with Licensed Mode
```csharp
// Adaptive concurrency tuning (Licensed mode only)
var pipeline = BatchSyncPipeline.Build(successRate: 0.8);

// Sequential mode (Unlicensed - always works)
var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.8);
```

## Before & After Comparison

### BEFORE: Task.WhenAll (Antipattern)
```csharp
public async Task<BatchResult> ProcessBatchAsync(List<int> itemIds)
{
    var tasks = itemIds.Select(id => apiClient.CallAsync(id)).ToList();
    
    try
    {
        var results = await Task.WhenAll(tasks);  // ❌ No throttling, all fire at once
        return new BatchResult { Results = results };
    }
    catch (Exception ex)
    {
        // ❌ Which item failed? Not clear. How many partially succeeded? Unknown.
        return new BatchResult { Error = ex.Message };
    }
}

// Problems:
// - API rate-limited/hammered (2000+ concurrent requests)
// - Single failure fails the entire batch
// - No visibility into partial success
// - Difficult to retry individual items
// - Resource leaks possible
```

### AFTER: BatchSync Pipeline (SOLID)
```csharp
public async Task<BatchResult> ProcessBatchAsync(List<int> itemIds)
{
    var pipeline = BatchSyncPipeline.Build(successRate: 0.8);
    var data = new BatchSyncData(itemIds);
    
    var result = await pipeline.RunAsync(data);
    var finalData = ((StepResult<BatchSyncData>.Success)result).Data;
    
    return new BatchResult
    {
        SuccessCount = finalData.SuccessCount,    // ✅ Clear metrics
        ErrorCount = finalData.ErrorCount,
        FailedIds = finalData.FailedIds,          // ✅ Know which failed
        AllResults = finalData.Results            // ✅ Access all results
    };
}

// Benefits:
// ✅ Controlled concurrency (ProcessBatchStep iterates sequentially)
// ✅ Partial success is first-class (both Results and FailedIds tracked)
// ✅ Individual failures don't fail the whole batch
// ✅ Clear error attribution (know exactly which ItemId failed)
// ✅ Easy to add retry, circuit breaker, throttling to ProcessBatchStep
// ✅ Testable (mock ProcessBatchStep.CallApiAsync)
// ✅ Observable (each step is a checkpoint)
// ✅ Extensible (new steps don't require recompile)
```

## Key Takeaways

1. **Prefer Structured Pipelines over Task.WhenAll** — Explicit control, observable, testable
2. **One Step = One Responsibility** — Each step transforms one aspect of the data
3. **Immutable Data Records** — All state flows forward, no hidden mutations
4. **Partial Success is a Feature** — Don't fail the whole batch on one error
5. **Make Cancellation Explicit** — Propagate CancellationToken to all async operations
6. **Test the Failures** — Happy path is easy; test timeouts, retries, edge cases

## Further Reading

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [EvalApp Documentation](https://github.com/evalapp/evalapp-public)
- [Functional Data Transformations](https://fsharpforfunandprofit.com/posts/recipe-part2/)
- [Error Handling in F#](https://fsharpforfunandprofit.com/posts/recipe-part3/)
