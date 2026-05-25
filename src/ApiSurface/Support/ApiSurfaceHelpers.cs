namespace EvalApp.Solid.Starter.Features.ApiSurface.Support;

public static class ApiSurfaceHelpers
{
    public static async IAsyncEnumerable<int> StreamSagaItemsAsync(ApiSurfaceData data)
    {
        for (var i = 0; i < 3; i++)
        {
            await Task.Yield();
            yield return data.Input + i;
        }
    }
}
