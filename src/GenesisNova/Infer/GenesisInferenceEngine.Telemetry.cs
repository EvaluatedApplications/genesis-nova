using GenesisNova.Model;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Core;

namespace GenesisNova.Infer;

public sealed partial class GenesisInferenceEngine
{
    public void ReportTelemetryOutcome(bool isSuccessful, GenerationResult result)
    {
        lock (_telemetryLock)
        {
            _telemetrySuccessEma = BlendEma(_telemetrySuccessEma, isSuccessful ? 1.0 : 0.0, TelemetryEmaAlpha);

            var confidenceSignal = Math.Clamp(result.PlatonicConfidence, 0.0, 1.0);
            var fallbackSignal = result.UsedNeuralFallback ? 0.0 : 1.0;
            var biasSignal = result.AppliedBiasCount > 0
                ? Math.Clamp(result.AverageBiasMagnitude / 0.9, 0.0, 1.0)
                : 0.5;
            var routeSignal = result.UsedPlatonicQuery ? confidenceSignal : 0.5;
            var efficacySignal = Math.Clamp(
                (_telemetrySuccessEma * 0.60) +
                (fallbackSignal * 0.20) +
                (routeSignal * 0.10) +
                (biasSignal * 0.10),
                0.0,
                1.0);

            _checkpointConceptEfficacyEma = BlendEma(_checkpointConceptEfficacyEma, efficacySignal, TelemetryEmaAlpha);

            var scaleTarget = DefaultNeuralBiasScale * (0.75 + (0.55 * _checkpointConceptEfficacyEma));
            _adaptiveBiasScale = Math.Clamp(scaleTarget, MinAdaptiveBiasScale, MaxAdaptiveBiasScale);
        }
    }

    public void ApplyTelemetryHint(InferenceTelemetryHint hint)
    {
        lock (_telemetryLock)
            _trainerHint = new InferenceTelemetryHint(
                BiasScale: Math.Clamp(hint.BiasScale, 0.7, 1.4));
    }

    private void ResetRouteTelemetry()
    {
        lock (_routeTelemetryLock)
            _lastRouteDecisions.Clear();
    }

    private void RecordRouteDecision(
        int predictedRoute,
        int finalRoute,
        bool platonicAttempted,
        bool platonicSucceeded,
        bool predictedRouteProducedFinalAnswer,
        int? supervisionLabel,
        string decisionPath,
        double routeConfidence,
        int platonicAssistInvocations = 0,
        int platonicAssistFired = 0)
    {
        lock (_routeTelemetryLock)
        {
            _lastRouteDecisions.Add(new RouteDecisionTelemetry(
                PredictedRoute: predictedRoute,
                FinalRoute: finalRoute,
                PlatonicRouteAttempted: platonicAttempted,
                PlatonicRouteSucceeded: platonicSucceeded,
                PredictedRouteProducedFinalAnswer: predictedRouteProducedFinalAnswer,
                SupervisionLabel: supervisionLabel,
                DecisionPath: decisionPath,
                RouteConfidence: routeConfidence,
                PlatonicAssistInvocations: platonicAssistInvocations,
                PlatonicAssistFired: platonicAssistFired));
        }
    }

    private static IReadOnlyList<PlatonicEvidence> CollapseEvidence(IEnumerable<PlatonicEvidence> evidence)
    {
        var collapsed = evidence
            .Where(e => !string.IsNullOrWhiteSpace(e.Concept))
            .GroupBy(e => (
                Concept: e.Concept.Trim().ToLowerInvariant(),
                Related: string.IsNullOrWhiteSpace(e.RelatedConcept) ? null : e.RelatedConcept.Trim().ToLowerInvariant(),
                e.Hop))
            .Select(g => new PlatonicEvidence(
                g.Key.Concept,
                g.Key.Related,
                g.Sum(e => e.Contribution),
                g.Key.Hop))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(32)
            .ToArray();
        return collapsed;
    }

    private static double BlendEma(double current, double sample, double alpha)
        => (current * (1.0 - alpha)) + (sample * alpha);
}
