namespace GenesisNova.Train;

/// <summary>
/// Shared mastery-proximity LR anneal curve. Anneals RELATIVE to the lesson's own target so the curve works
/// for both the strict (0.95) and majority-mastery (0.85) bars: FULL steps until just below target, shrink in
/// the approach band, smallest at/above. Extracted verbatim from the two byte-identical target-relative copies
/// (<c>CoreBootstrapRegime.AnnealFactor</c> and <c>GenesisModularTrainingOrchestrator.AnnealFactor</c>).
/// (NOTE: <c>GenesisTrainingOrchestrator.AnnealedLearningRateFactor</c> is a one-arg variant with FIXED literal
/// breakpoints 0.92/0.97 — it is intentionally NOT routed here, because <c>0.95 - 0.03</c> / <c>0.95 + 0.02</c>
/// are not guaranteed bit-identical to the literals 0.92/0.97, which could flip a boundary comparison.)
/// </summary>
internal static class MasteryAnneal
{
    internal static double Factor(double success, double target)
    {
        if (success < target - 0.03) return 1.00;  // still climbing (incl. the plateau zone): full steps
        if (success < target + 0.02) return 0.30;  // near the top: shrink to settle, stop overshooting
        return 0.10;                                 // at mastery: small steps to HOLD without bouncing out
    }
}
