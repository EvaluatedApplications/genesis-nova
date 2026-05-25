namespace EvalApp.Solid.Starter.Platform.Support;

public sealed class BonusService(int bonus)
{
    public int Bonus { get; } = bonus;
}

