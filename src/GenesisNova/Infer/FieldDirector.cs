using System;
using System.Collections.Generic;

namespace GenesisNova.Infer;

/// <summary>
/// THE LEARNED DIRECTOR (PLATONIC_MIND.md, Stage 2). Deciding WHEN a generative op should fire (compose the meanings)
/// vs plain retrieval is NOT a hand-heuristic problem — substrate-confidence arbitration was tried and FAILS to
/// separate a good composition ("red fruit") from a bad one ("describe otter"). It is a judgment the controller must
/// LEARN, self-supervised, from the OUTCOME (which route actually produced the target). This is the keep-core control
/// path the reckoning pointed at: a general controller over general routes, trained by prediction of what was correct
/// — NOT a classifier over a fixed gym-shape taxonomy.
///
/// A small ONLINE logistic regression over substrate features (relation degree of the query's concepts, how many
/// there are, the compose vs retrieve settle confidences). One example teaches it; it generalises the decision
/// boundary. Tiny + transparent on purpose — the point is that the decision is LEARNED from outcome, not coded.
/// </summary>
public sealed class FieldDirector
{
    private readonly double[] _w;
    private double _b;
    private readonly double _lr;

    public FieldDirector(int featureCount, double learningRate = 0.3, double initialBias = 0.0)
    {
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));
        _w = new double[featureCount];
        _lr = learningRate;
        _b = initialBias; // a CONSERVATIVE negative prior makes the untrained director default to retrieval (safe) and
                          // LEARN its way to compose — so wiring it in never opens the gate before it has earned it.
    }

    public int FeatureCount => _w.Length;

    /// <summary>P(this query wants a generative COMPOSE rather than plain retrieval) ∈ (0,1).</summary>
    public double ComposeProbability(IReadOnlyList<double> features)
    {
        var z = _b;
        for (var i = 0; i < _w.Length && i < features.Count; i++) z += _w[i] * features[i];
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    /// <summary>The director's decision: fire the generative compose, or defer to plain retrieval.</summary>
    public bool ShouldCompose(IReadOnlyList<double> features) => ComposeProbability(features) >= 0.5;

    /// <summary>SELF-SUPERVISED update: a single labelled outcome (was compose the route that produced the target?).
    /// Logistic-loss gradient step — recent outcomes shape the learned boundary.</summary>
    public void Observe(IReadOnlyList<double> features, bool composeWasCorrect)
    {
        var err = ComposeProbability(features) - (composeWasCorrect ? 1.0 : 0.0);
        for (var i = 0; i < _w.Length && i < features.Count; i++) _w[i] -= _lr * err * features[i];
        _b -= _lr * err;
    }

    /// <summary>The learned weights (for inspection / persistence).</summary>
    public IReadOnlyList<double> Weights => _w;
    public double Bias => _b;
}
