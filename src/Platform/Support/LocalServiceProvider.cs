namespace EvalApp.Solid.Starter.Platform.Support;

public sealed class LocalServiceProvider(Dictionary<Type, object> services) : IServiceProvider
{
    private readonly Dictionary<Type, object> _services = services;

    public object? GetService(Type serviceType)
        => _services.TryGetValue(serviceType, out var service) ? service : null;
}

