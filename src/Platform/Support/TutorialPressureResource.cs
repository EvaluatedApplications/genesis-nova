using EvalApp.Core.Pressure;

namespace EvalApp.Solid.Starter.Platform.Support;

public sealed class TutorialPressureResource(string name, float pressure) : IPressureResource
{
    private float _pressure = pressure;

    public string Name { get; } = name;
    public float Pressure => _pressure;

    public void Consume(float amount)
        => _pressure += amount;

    public void Reset()
        => _pressure = 0f;
}

