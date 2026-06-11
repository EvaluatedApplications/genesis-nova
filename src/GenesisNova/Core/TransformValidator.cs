using System.Diagnostics;

namespace GenesisNova.Core;

/// <summary>
/// Structural audit gate for learned transforms. Ported from the canonical source of truth
/// (Genesis.Engine.GenesisLearner.TransformValidator.AuditTransform) so nova's transform
/// confidence and lifecycle conform to the engine's rules instead of a fake +0.05 increment.
///
/// The audit runs five independent checks over the transform's stored (input, output) embedding
/// sample pairs:
///   1. Round-trip integrity — apply +T forward and −T backward; both directions must land close.
///   2. Neighborhood consistency — cosine(stored T, per-pair delta) ≥ 0.45 (mapped to [0,1]).
///   3. Collision avoidance — predicted outputs must span a non-trivial region (spread ≥ 0.05).
///   4. Minimum support — at least 5 sample pairs.
///   5. FTD self-consistency — Pearson correlation of pairwise distances ≥ 0.0 (not anti-correlated).
///
/// State machine: Candidate → Provisional (≥3 passes) → Stable (≥6 passes AND FTD polarity ≥ 0.5
/// AND self-consistency ≥ 0.4) → Retired (audit fails AND confidence &lt; 0.3 AND 0 prior passes
/// AND ≥20 examples). A Retired transform recovers to Candidate on the next passing audit.
///
/// FTD polarity coherence is NOT a hard gate (arithmetic has structurally mixed polarity in the
/// log face) — it only gates the Provisional→Stable advance, matching canonical.
/// </summary>
public sealed class TransformValidator
{
    private const double RoundTripThreshold = 0.45;
    private const double NeighborhoodThreshold = 0.45;
    private const double SelfConsistencyGateThreshold = 0.0;   // Pearson ≥ 0 — not anti-correlated
    private const double CollisionSpreadThreshold = 0.05;
    private const int MinSupport = 5;

    private const double StablePolarityThreshold = 0.5;
    private const double StableSelfConsistencyThreshold = 0.4;

    private readonly int _dim;

    public TransformValidator(int embeddingDim)
    {
        _dim = embeddingDim;
    }

    /// <summary>
    /// Run the 5-gate structural audit and return an updated <see cref="Transform"/> with refreshed
    /// audit scores, pass count, lifecycle state, and (gate-driven) confidence. Pure: returns a new
    /// record rather than mutating in place. If there is insufficient support (&lt;5 pairs) the input
    /// transform is returned unchanged — matching canonical (audit is a no-op below minimum support).
    /// </summary>
    public Transform AuditTransform(Transform transform)
    {
        var inputs = transform.InputSamples;
        var outputs = transform.OutputSamples;
        if (inputs is null || outputs is null) return transform;

        int n = Math.Min(inputs.Count, outputs.Count);
        if (n < MinSupport) return transform; // Check 4: minimum support

        int dim = _dim;
        var T = transform.Vector;

        // ── Check 1: Round-trip integrity ──────────────────────────────────
        double totalRoundTrip = 0;
        for (int i = 0; i < n; i++)
        {
            var fwd = new double[dim];
            for (int d = 0; d < dim; d++) fwd[d] = inputs[i][d] + T[d];
            double fwdError = EuclideanDistance(fwd, outputs[i]);

            var inv = new double[dim];
            for (int d = 0; d < dim; d++) inv[d] = outputs[i][d] - T[d];
            double invError = EuclideanDistance(inv, inputs[i]);

            totalRoundTrip += Math.Max(fwdError, invError);
        }
        double roundTripScore = 1.0 - Math.Clamp(totalRoundTrip / n / 2.0, 0, 1);

        // ── Check 2: Neighborhood consistency ──────────────────────────────
        double tNorm = 0;
        for (int d = 0; d < dim; d++) tNorm += T[d] * T[d];
        tNorm = Math.Sqrt(tNorm);

        double totalCosine = 0;
        for (int i = 0; i < n; i++)
        {
            double dot = 0, eTNorm = 0;
            for (int d = 0; d < dim; d++)
            {
                double delta = outputs[i][d] - inputs[i][d];
                dot += T[d] * delta;
                eTNorm += delta * delta;
            }
            eTNorm = Math.Sqrt(eTNorm);
            totalCosine += (tNorm > 1e-10 && eTNorm > 1e-10) ? dot / (tNorm * eTNorm) : 0;
        }
        double neighborhoodScore = Math.Clamp((totalCosine / n + 1.0) / 2.0, 0, 1);

        // ── Check 3: Collision check (predicted-output spread) ──────────────
        var predictions = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var predicted = new double[dim];
            for (int d = 0; d < dim; d++) predicted[d] = inputs[i][d] + T[d];
            predictions[i] = predicted;
        }
        double outputSpread = 0;
        if (n >= 2)
        {
            for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                outputSpread += EuclideanDistance(predictions[i], predictions[j]);
            outputSpread /= (n * (n - 1) / 2.0);
        }
        bool collisionPass = outputSpread >= CollisionSpreadThreshold || n == 1;

        // ── FTD metrics (Gauss polarity + gap-equation self-consistency) ────
        var inArr = ToArray(inputs, n);
        var outArr = ToArray(outputs, n);
        double polarityCoherence = FtdMetrics.PolarityCoherenceScore(inArr, outArr, T, dim);
        double selfConsistency = FtdMetrics.SelfConsistencyScore(inArr, outArr, T, dim);

        // ── Gate decision ──────────────────────────────────────────────────
        bool auditPassed = roundTripScore >= RoundTripThreshold
                        && neighborhoodScore >= NeighborhoodThreshold
                        && collisionPass
                        && selfConsistency >= SelfConsistencyGateThreshold;

        if (auditPassed)
        {
            int newPassCount = transform.AuditPassCount + 1;
            bool ftdStableReady = polarityCoherence >= StablePolarityThreshold
                               && selfConsistency >= StableSelfConsistencyThreshold;
            TransformState newState = transform.State switch
            {
                TransformState.Candidate when newPassCount >= 3 => TransformState.Provisional,
                TransformState.Provisional when newPassCount >= 6 && ftdStableReady => TransformState.Stable,
                TransformState.Retired => TransformState.Candidate, // allow recovery
                _ => transform.State
            };

            // Confidence is gate-driven: blend of the structural quality scores, monotone with passes.
            double quality = (roundTripScore + neighborhoodScore + Math.Max(0, selfConsistency)) / 3.0;
            double newConfidence = Math.Clamp(0.5 * quality + 0.5 * Math.Min(1.0, newPassCount / 6.0), 0.0, 1.0);

            Trace.WriteLine($"Genesis(nova) audit PASS [{transform.FunctionName}] rt={roundTripScore:F2} nb={neighborhoodScore:F2} spread={outputSpread:F3} pol={polarityCoherence:F2} sc={selfConsistency:F2} → {newState}");

            return transform with
            {
                AuditPassCount = newPassCount,
                RoundTripScore = roundTripScore,
                NeighborhoodScore = neighborhoodScore,
                PolarityCoherenceScore = polarityCoherence,
                SelfConsistencyScore = selfConsistency,
                Confidence = newConfidence,
                State = newState
            };
        }
        else
        {
            // Fail: penalise confidence; retire if persistently broken.
            double newConfidence = Math.Max(0.0, transform.Confidence - 0.08);
            bool shouldRetire = newConfidence < 0.3
                             && transform.AuditPassCount == 0
                             && transform.ObservationCount >= 20;
            TransformState newState = shouldRetire ? TransformState.Retired : transform.State;

            Trace.WriteLine($"Genesis(nova) audit FAIL [{transform.FunctionName}] rt={roundTripScore:F2} nb={neighborhoodScore:F2} spread={outputSpread:F3} pol={polarityCoherence:F2} sc={selfConsistency:F2} collision={!collisionPass}{(shouldRetire ? " → RETIRED" : "")}");

            return transform with
            {
                RoundTripScore = roundTripScore,
                NeighborhoodScore = neighborhoodScore,
                PolarityCoherenceScore = polarityCoherence,
                SelfConsistencyScore = selfConsistency,
                Confidence = newConfidence,
                State = newState
            };
        }
    }

    private static double[][] ToArray(IReadOnlyList<double[]> samples, int n)
    {
        var arr = new double[n][];
        for (int i = 0; i < n; i++) arr[i] = samples[i];
        return arr;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        int dim = Math.Min(a.Length, b.Length);
        double sum = 0;
        for (int i = 0; i < dim; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
