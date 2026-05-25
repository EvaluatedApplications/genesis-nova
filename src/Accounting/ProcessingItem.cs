namespace EvalApp.Solid.Starter.Accounting;

/// <summary>
/// Wrapper record for processing individual items in a ForEach-like pattern.
/// Contains the ItemId and its processing result.
/// </summary>
public record ProcessingItem(int ItemId, ApiResponse? Response = null, string? ErrorMessage = null);

