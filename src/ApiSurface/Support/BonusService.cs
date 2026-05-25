namespace EvalApp.Solid.Starter.Features.ApiSurface.Support;

public sealed class BonusService(int bonus)
{
    public int Bonus { get; } = bonus;
}
