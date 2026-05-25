namespace EvalApp.Solid.Starter.Features.BatchSync;

/// <summary>
/// Represents the result of a single API batch call.
/// </summary>
public record ApiResponse(int ItemId, string Status, string? Message = null);

/// <summary>
/// Pipeline data flowing through the BatchSync pipeline.
/// Demonstrates ForEach + Gate pattern for throttled batch processing.
///
/// Flow:
/// - ItemIds: list of IDs to process
/// - Results: dictionary mapping ItemId -> ApiResponse
/// - FailedIds: list of ItemIds that failed to process
/// - SuccessCount: number of successful API calls
/// - ErrorCount: number of failed API calls
/// </summary>
public record BatchSyncData(
    List<int> ItemIds,
    Dictionary<int, ApiResponse>? Results = null,
    List<int>? FailedIds = null,
    int SuccessCount = 0,
    int ErrorCount = 0);
