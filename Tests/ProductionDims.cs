using GenesisNova.Core;

namespace GenesisNova.Tests;

/// <summary>
/// Production dimensions for the whole test suite — DERIVED from the runtime's single source of truth
/// (<see cref="GenesisNovaConfig"/>), never hardcoded. Tests therefore exercise the exact production
/// model width and the exact face-width convention (HiddenSize/2) the runtime uses; change the defaults
/// in <see cref="GenesisNovaConfig"/> and both runtime and tests move together. The platonic space is
/// only fully instantiated at this production face dimension — the old tiny test dims were degenerate.
/// </summary>
internal static class ProductionDims
{
    private static readonly GenesisNovaConfig Config = new();

    /// <summary>Production model width = <see cref="GenesisNovaConfig.HiddenSize"/> default.</summary>
    public static int HiddenSize => Config.HiddenSize;

    /// <summary>Production platonic face width = <see cref="GenesisNovaConfig.FaceDimension"/> (the single /2 definition).</summary>
    public static int FaceDimension => Config.FaceDimension;
}
