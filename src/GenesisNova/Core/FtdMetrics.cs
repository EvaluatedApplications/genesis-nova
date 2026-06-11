namespace GenesisNova.Core;

/// <summary>
/// Pure mathematical metrics inspired by Foundational Ternary Dynamics (FTD).
/// Ported faithfully from the canonical source of truth
/// (Genesis.Engine.GenesisLearner.Training.FtdMetrics) so the nova transform audit
/// matches the engine's gate semantics exactly.
///
/// FTD axiom mapping:
///   PolarityCoherenceScore ← Gauss constraint (div J = s): T-vector polarity must align with per-pair delta signs.
///   SelfConsistencyScore   ← Gap equation fixed point: T must preserve pairwise geometry (isometric translation).
/// </summary>
public static class FtdMetrics
{
    /// <summary>
    /// FTD Gauss constraint analog: per-dimension sign agreement between T and actual deltas.
    /// For each dimension d where |T[d]| is non-trivial, measures the fraction of training pairs
    /// whose delta[d] shares the sign of T[d], weighted by |T[d]| so important dims dominate.
    /// Returns ∈ [0,1] where 1 = perfect polarity alignment, 0 = no signal or empty input.
    /// </summary>
    public static double PolarityCoherenceScore(
        double[][] inputEmbeds, double[][] outputEmbeds, double[] T, int dim)
    {
        int n = inputEmbeds.Length;
        if (n == 0) return 0.0;

        double tMagSum = 0;
        for (int d = 0; d < dim; d++) tMagSum += Math.Abs(T[d]);
        if (tMagSum < 1e-12) return 0.0; // zero T-vector

        double weightedAgreement = 0;
        double totalWeight = 0;

        for (int d = 0; d < dim; d++)
        {
            double absT = Math.Abs(T[d]);
            if (absT < 1e-12) continue; // skip negligible dimensions

            double signT = Math.Sign(T[d]);

            int agree = 0;
            int countNonZero = 0;
            for (int i = 0; i < n; i++)
            {
                double delta = outputEmbeds[i][d] - inputEmbeds[i][d];
                if (Math.Abs(delta) < 1e-12) continue; // zero delta = no info
                countNonZero++;
                if (Math.Sign(delta) == signT) agree++;
            }

            if (countNonZero == 0) continue;
            double agreement = (double)agree / countNonZero;
            weightedAgreement += agreement * absT;
            totalWeight += absT;
        }

        return totalWeight > 1e-12 ? weightedAgreement / totalWeight : 0.0;
    }

    /// <summary>
    /// FTD gap equation analog: measures whether T is an isometric translation — i.e. the
    /// pairwise geometry of the inputs is preserved in the outputs. Returns the Pearson
    /// correlation of pairwise input distances vs. pairwise output distances.
    /// Returns ∈ [0,1] where 1 = perfectly self-consistent (isometric). 1.0 for 0 or 1 pair.
    /// </summary>
    public static double SelfConsistencyScore(
        double[][] inputEmbeds, double[][] outputEmbeds, double[] T, int dim)
    {
        int n = inputEmbeds.Length;
        if (n <= 1) return 1.0; // vacuously consistent

        int pairCount = n * (n - 1) / 2;
        var inputDists = new double[pairCount];
        var outputDists = new double[pairCount];

        int idx = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double inD2 = 0, outD2 = 0;
                for (int d = 0; d < dim; d++)
                {
                    double iDiff = inputEmbeds[i][d] - inputEmbeds[j][d];
                    inD2 += iDiff * iDiff;

                    double oDiff = outputEmbeds[i][d] - outputEmbeds[j][d];
                    outD2 += oDiff * oDiff;
                }
                inputDists[idx] = Math.Sqrt(inD2);
                outputDists[idx] = Math.Sqrt(outD2);
                idx++;
            }
        }

        return PearsonCorrelation(inputDists, outputDists);
    }

    /// <summary>
    /// Pearson correlation coefficient between two equal-length arrays.
    /// Returns 1.0 if arrays are constant (no variance) or have ≤1 element.
    /// Clamped to [0,1] (matches canonical — anti-correlation reads as 0).
    /// </summary>
    private static double PearsonCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        if (n <= 1) return 1.0;

        double sumX = 0, sumY = 0;
        for (int i = 0; i < n; i++) { sumX += x[i]; sumY += y[i]; }
        double meanX = sumX / n, meanY = sumY / n;

        double cov = 0, varX = 0, varY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - meanX, dy = y[i] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        if (varX < 1e-20 || varY < 1e-20) return 1.0; // constant = no distortion
        return Math.Clamp(cov / (Math.Sqrt(varX) * Math.Sqrt(varY)), 0, 1);
    }
}
