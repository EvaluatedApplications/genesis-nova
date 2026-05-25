using CoreAbstractions = EvalApp.Abstractions;

namespace EvalApp.Solid.Starter.Platform.Merge;

public sealed class ApiSurfaceMergeStrategy : CoreAbstractions.IMergeStrategy<ApiSurfaceData>
{
    public ApiSurfaceData Merge(ApiSurfaceData original, IReadOnlyList<ApiSurfaceData> results)
    {
        var left = original.ParallelLeft;
        var right = original.ParallelRight;

        foreach (var result in results)
        {
            left = Math.Max(left, result.ParallelLeft);
            right = Math.Max(right, result.ParallelRight);
        }

        return original.AppendTrace("ParallelGroup:CustomMerge") with
        {
            ParallelLeft = left,
            ParallelRight = right
        };
    }
}

