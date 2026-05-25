namespace EvalApp.Solid.Starter.Catalog;

/// <summary>
/// Materialize — prepare input stream for iteration.
/// Initializes empty collections for valid and invalid items.
/// Pure step: validates input structure before pipeline processing begins.
/// </summary>
public class MaterializeStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        // Validate input stream is not null
        if (data.InputStream == null || data.InputStream.Count == 0)
        {
            return data with
            {
                ValidItems = new List<ValidatedRecord>(),
                InvalidItems = new List<ValidationError>(),
                TotalProcessed = 0
            };
        }

        // Initialize output collections
        return data with
        {
            ValidItems = new List<ValidatedRecord>(),
            InvalidItems = new List<ValidationError>(),
            TotalProcessed = data.InputStream.Count
        };
    }
}

