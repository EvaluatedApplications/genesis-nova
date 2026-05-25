namespace EvalApp.Solid.Starter.Features.Ingestion;

/// <summary>
/// ProcessAllItems — iterate stream and populate valid/invalid collections.
/// Demonstrates partial success: both collections are populated.
/// Pure step: one responsibility (process all items with error handling).
/// </summary>
public class ProcessAllItemsStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        if (data.InputStream == null || data.InputStream.Count == 0)
        {
            return data;
        }

        var validateStep = new ValidateItemStep();
        var processStep = new ProcessItemStep();
        var validItems = data.ValidItems ?? new List<ValidatedRecord>();
        var invalidItems = data.InvalidItems ?? new List<ValidationError>();

        foreach (var rawRecord in data.InputStream)
        {
            // Validate the record
            var errorReason = validateStep.ValidateRecord(rawRecord);

            if (errorReason == null)
            {
                // Validation passed, transform and add to valid items
                var validatedRecord = processStep.ProcessRecord(rawRecord);
                validItems.Add(validatedRecord);
            }
            else
            {
                // Validation failed, record error and add to invalid items
                invalidItems.Add(new ValidationError(rawRecord.Id, errorReason));
            }
        }

        return data with
        {
            ValidItems = validItems,
            InvalidItems = invalidItems
        };
    }
}
