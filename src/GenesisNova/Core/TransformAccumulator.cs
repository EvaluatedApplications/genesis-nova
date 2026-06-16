using System.Collections.Immutable;

namespace GenesisNova.Core;

/// <summary>
/// Lifecycle state of a learned transform (canonical: Genesis.Engine TransformState).
/// Candidate → Provisional (>=3 audit passes) → Stable (>=6 passes + FTD polarity/self-consistency)
/// → Retired (confidence collapsed and persistently failing audit).
/// </summary>
public enum TransformState
{
    Candidate = 0,
    Provisional = 1,
    Stable = 2,
    Retired = 3
}

/// <summary>
/// A learned transform from input embedding space to output embedding space.
///
/// Core learning algorithm from Genesis Engine:
/// - One example teaches one transform (no gradient descent, pure vector arithmetic)
/// - Transforms average as new examples arrive
/// - Embeddings adapt to make transforms consistent
///
/// Confidence and lifecycle state are driven by the canonical 5-gate structural audit
/// (<see cref="TransformValidator"/>), not a fake per-observation increment.
/// <see cref="InputSamples"/> / <see cref="OutputSamples"/> hold a bounded ring of recent
/// (input, output) embedding pairs — the sufficient statistics the audit gates require.
/// </summary>
public record Transform(
    string FunctionName,
    double[] Vector,
    int ObservationCount,
    double Confidence = 0.0,
    TransformState State = TransformState.Candidate,
    int AuditPassCount = 0,
    double RoundTripScore = 0.0,
    double NeighborhoodScore = 0.0,
    double PolarityCoherenceScore = 0.0,
    double SelfConsistencyScore = 0.0,
    // The numeric face this function is a CONSTANT translation in — the one whose per-example delta is
    // consistent across training (poly=1 for additive +k, log=2 for multiplicative ×k). 0 = unknown/auto.
    // Decoding the applied transform in this face avoids the spurious-but-clean reading in the other face.
    int PreferredFace = 0,
    // EARNED downstream reliability (distinct from the structural audit Confidence): how often APPLYING this
    // transform actually predicted the target better than identity, across training. Feeds ReliabilityUcb →
    // route perception so the route head learns which transforms are genuinely useful vs noisy. Persisted.
    int SuccessCount = 0,
    int AttemptCount = 0,
    IReadOnlyList<double[]>? InputSamples = null,
    IReadOnlyList<double[]>? OutputSamples = null);

/// <summary>
/// JSON-serializable snapshot of a single learned <see cref="Transform"/> for checkpoint persistence.
/// Omits the audit sample rings (<see cref="Transform.InputSamples"/>/<see cref="Transform.OutputSamples"/>),
/// which are re-warmed on the next training pass; only the durable learned vector + audit scores survive.
/// All fields are System.Text.Json-friendly (primitives, double[], enum).
/// </summary>
public sealed record TransformEntrySnapshot(
    string FunctionName,
    double[] Vector,
    int ObservationCount,
    double Confidence,
    TransformState State,
    double RoundTripScore,
    double NeighborhoodScore,
    double PolarityCoherenceScore,
    double SelfConsistencyScore,
    int AuditPassCount,
    int SuccessCount = 0,
    int AttemptCount = 0);

/// <summary>
/// JSON-serializable snapshot of an entire <see cref="TransformAccumulator"/> for checkpoint persistence.
/// </summary>
public sealed record TransformAccumulatorSnapshot(
    int EmbeddingDim,
    IReadOnlyList<TransformEntrySnapshot> Transforms);

/// <summary>
/// Accumulates and manages learned transforms.
/// 
/// This is the heart of the Genesis Engine approach:
/// T(f) = avg(embed(output_i) - embed(input_i)) for all examples of f
/// </summary>
public class TransformAccumulator
{
    /// <summary>Bounded ring size for stored (input, output) embedding samples used by the audit gates.</summary>
    private const int MaxAuditSamples = 64;

    /// <summary>Canonical audit cadence — run the 5-gate structural audit every N examples.</summary>
    private const int AuditEvery = 20;

    private readonly Dictionary<string, Transform> _transforms = new();
    private readonly int _embeddingDim;
    private readonly TransformValidator _validator;

    public TransformAccumulator(int embeddingDim)
    {
        _embeddingDim = embeddingDim;
        _validator = new TransformValidator(embeddingDim);
    }
    
    public IReadOnlyDictionary<string, Transform> Transforms => _transforms;
    public int EmbeddingDimension => _embeddingDim;
    
    /// <summary>
    /// Learn a transform from a single example.
    /// (output_embedding - input_embedding) is averaged with existing transform.
    /// </summary>
    public void Learn(
        string functionName,
        double[] inputEmbedding,
        double[] outputEmbedding)
    {
        if (inputEmbedding.Length != _embeddingDim)
            throw new ArgumentException($"Expected embedding dim {_embeddingDim}, got {inputEmbedding.Length}");
        if (outputEmbedding.Length != _embeddingDim)
            throw new ArgumentException($"Expected embedding dim {_embeddingDim}, got {outputEmbedding.Length}");
        
        // Compute delta (transform)
        var delta = new double[_embeddingDim];
        for (int i = 0; i < _embeddingDim; i++)
            delta[i] = outputEmbedding[i] - inputEmbedding[i];
        
        if (_transforms.TryGetValue(functionName, out var existing))
        {
            // F8 canonical (PlatonicCompute.cs:675-681): harmonic/recency-biased alpha
            //   alpha = max(0.02, 2/(ExampleCount + 2))
            // Base is the PRE-increment count (existing.ObservationCount), matching canonical semantics.
            // The 0.02 floor prevents transform freeze when ObservationCount accumulates across epochs;
            // recent examples get ~2× the weight of older ones.
            var alpha = Math.Max(0.02, 2.0 / (existing.ObservationCount + 2));

            var blended = new double[_embeddingDim];
            for (int i = 0; i < _embeddingDim; i++)
                blended[i] = existing.Vector[i] * (1.0 - alpha) + delta[i] * alpha;

            var newCount = existing.ObservationCount + 1;
            var newInputs = AppendSample(existing.InputSamples, inputEmbedding);
            var newOutputs = AppendSample(existing.OutputSamples, outputEmbedding);
            var updated = existing with
            {
                Vector = blended,
                ObservationCount = newCount,
                PreferredFace = ComputePreferredFace(newInputs, newOutputs),
                InputSamples = newInputs,
                OutputSamples = newOutputs,
            };

            // Drive Confidence + lifecycle State from the canonical 5-gate audit, not a fake increment.
            // Canonical cadence: audit every 20 examples (TransformValidator.AuditTransform).
            _transforms[functionName] = (newCount % AuditEvery == 0)
                ? _validator.AuditTransform(updated)
                : updated;
        }
        else
        {
            // First observation of this transform — Candidate, confidence seeded low.
            _transforms[functionName] = new Transform(
                functionName,
                delta,
                1,
                Confidence: 0.1,
                State: TransformState.Candidate,
                InputSamples: new[] { (double[])inputEmbedding.Clone() },
                OutputSamples: new[] { (double[])outputEmbedding.Clone() });
        }
    }

    /// <summary>
    /// Append an embedding sample to a bounded ring buffer (keeps the most recent <see cref="MaxAuditSamples"/>).
    /// Stores defensive copies so later mutation of the source arrays cannot corrupt audit statistics.
    /// </summary>
    private static IReadOnlyList<double[]> AppendSample(IReadOnlyList<double[]>? existing, double[] sample)
    {
        var copy = (double[])sample.Clone();
        if (existing is null || existing.Count == 0)
            return new List<double[]> { copy };

        var list = new List<double[]>(Math.Min(existing.Count, MaxAuditSamples) + 1);
        // Keep only the most recent (MaxAuditSamples - 1) so the appended sample fits within the cap.
        int skip = Math.Max(0, existing.Count - (MaxAuditSamples - 1));
        for (int i = skip; i < existing.Count; i++)
            list.Add(existing[i]);
        list.Add(copy);
        return list;
    }
    
    /// <summary>
    /// Apply a learned transform to an embedding.
    /// predicted = input_embedding + transform_vector
    /// </summary>
    public double[]? Apply(string functionName, double[] inputEmbedding)
    {
        if (!_transforms.TryGetValue(functionName, out var transform))
            return null;
        
        var result = new double[_embeddingDim];
        for (int i = 0; i < _embeddingDim; i++)
            result[i] = inputEmbedding[i] + transform.Vector[i];
        
        return result;
    }
    
    /// <summary>
    /// Look up the transform embedding (operation representation) by name.
    /// Returns the transform vector and confidence.
    /// </summary>
    public (double[] Embedding, double Confidence)? LookupOperation(string operationName)
    {
        if (_transforms.TryGetValue(operationName, out var t))
            return (t.Vector, t.Confidence);
        return null;
    }

    public bool TryGetTransform(string functionName, out Transform transform)
        => _transforms.TryGetValue(functionName, out transform!);

    // ── EARNED reliability (the "bubble up successes to the route head" mechanism) ───────────────────────────
    // A transform's reliability is its measured downstream success rate — does applying it actually help —
    // NOT a hand-set label. The route head reads it (via ComputeRoutePerception) so it learns to trust the
    // function route when proven transforms exist and distrust it when they're noisy.

    /// <summary>Record one APPLICATION outcome (Laplace-smoothed success rate is what callers read back).
    /// Immutable record → replace the entry with incremented counts. No-op for an unknown transform.</summary>
    public void RecordOutcome(string functionName, bool correct)
    {
        if (!_transforms.TryGetValue(functionName, out var t))
            return;
        _transforms[functionName] = t with
        {
            AttemptCount = t.AttemptCount + 1,
            SuccessCount = t.SuccessCount + (correct ? 1 : 0),
        };
    }

    /// <summary>Did applying the CURRENT transform predict <paramref name="output"/> closer than doing nothing
    /// (identity)? A scale-free success criterion: a consistent constant-translation improves over identity, a
    /// noisy/inconsistent one does not. False when the transform is unknown.</summary>
    public bool ApplyImprovesOverIdentity(string functionName, double[] input, double[] output)
    {
        var predicted = Apply(functionName, input);
        if (predicted is null)
            return false;
        return EuclideanDistance(predicted, output) < EuclideanDistance(input, output);
    }

    /// <summary>Laplace-smoothed success rate (success+1)/(attempts+2) in [0,1]; 0.5 for a known-but-untried
    /// transform, 0 for an unknown one.</summary>
    public double Reliability(string functionName)
        => _transforms.TryGetValue(functionName, out var t)
            ? (t.SuccessCount + 1.0) / (t.AttemptCount + 2.0)
            : 0.0;

    /// <summary>UCB1-style reliability: success rate + an exploration bonus that decays with attempts, so an
    /// under-tried transform is optimistically surfaced (gets routed enough to earn or disprove itself) instead
    /// of being frozen out. Clamped to [0,1]. 0 for an unknown transform.</summary>
    public double ReliabilityUcb(string functionName)
    {
        if (!_transforms.TryGetValue(functionName, out var t))
            return 0.0;
        var totalAttempts = 0;
        foreach (var x in _transforms.Values) totalAttempts += x.AttemptCount;
        var bonus = Math.Sqrt(2.0 * Math.Log(totalAttempts + 1.0) / (t.AttemptCount + 1.0));
        return Math.Clamp(Reliability(functionName) + bonus, 0.0, 1.0);
    }

    /// <summary>The most-reliable (UCB) transform's score — a target-agnostic "do I have a proven transform
    /// capability here?" signal for the route head. 0 when no transforms exist.</summary>
    public double BestReliabilityUcb()
    {
        var best = 0.0;
        foreach (var name in _transforms.Keys)
            best = Math.Max(best, ReliabilityUcb(name));
        return best;
    }
    
    /// <summary>
    /// Find the most similar operation to a given embedding (KNN over operation embeddings).
    /// Used when the model's operation query embedding is ambiguous.
    /// </summary>
    public (string Name, double Confidence)? FindSimilarOperation(double[] queryEmbedding)
    {
        if (_transforms.Count == 0)
            return null;
        
        double bestDist = double.MaxValue;
        Transform? best = null;
        
        foreach (var t in _transforms.Values)
        {
            var dist = EuclideanDistance(queryEmbedding, t.Vector);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = t;
            }
        }
        
        if (best is null)
            return null;
        
        // Distance → similarity conversion
        var similarity = 1.0 / (1.0 + bestDist);
        return (best.FunctionName, similarity);
    }
    
    /// <summary>
    /// Export a JSON-serializable snapshot of all learned transforms for checkpoint persistence.
    /// The audit sample rings are intentionally omitted (re-warm on next training); only the learned
    /// vector and audit scores/lifecycle are captured. Defensive copies of vectors are taken.
    /// </summary>
    public TransformAccumulatorSnapshot ExportSnapshot()
    {
        var entries = new List<TransformEntrySnapshot>(_transforms.Count);
        foreach (var t in _transforms.Values)
        {
            entries.Add(new TransformEntrySnapshot(
                t.FunctionName,
                (double[])t.Vector.Clone(),
                t.ObservationCount,
                t.Confidence,
                t.State,
                t.RoundTripScore,
                t.NeighborhoodScore,
                t.PolarityCoherenceScore,
                t.SelfConsistencyScore,
                t.AuditPassCount,
                t.SuccessCount,
                t.AttemptCount));
        }
        return new TransformAccumulatorSnapshot(_embeddingDim, entries);
    }

    /// <summary>
    /// Rebuild the learned transform table from a checkpoint snapshot. Clears any existing transforms.
    /// If the snapshot's embedding dimension does not match this accumulator's, the import is skipped
    /// (the table is left empty) rather than producing dimension-mismatched vectors — graceful degradation.
    /// Restored transforms start with empty audit sample rings (re-warmed on next training).
    /// </summary>
    public void ImportSnapshot(TransformAccumulatorSnapshot snapshot)
    {
        if (snapshot is null)
            return;

        _transforms.Clear();

        // Graceful skip: a checkpoint trained at a different embedding dim cannot be reused.
        if (snapshot.EmbeddingDim != _embeddingDim)
            return;

        foreach (var entry in snapshot.Transforms ?? Array.Empty<TransformEntrySnapshot>())
        {
            if (entry is null || string.IsNullOrEmpty(entry.FunctionName))
                continue;
            if (entry.Vector is null || entry.Vector.Length != _embeddingDim)
                continue;

            _transforms[entry.FunctionName] = new Transform(
                entry.FunctionName,
                (double[])entry.Vector.Clone(),
                entry.ObservationCount,
                Confidence: entry.Confidence,
                State: entry.State,
                AuditPassCount: entry.AuditPassCount,
                RoundTripScore: entry.RoundTripScore,
                NeighborhoodScore: entry.NeighborhoodScore,
                PolarityCoherenceScore: entry.PolarityCoherenceScore,
                SelfConsistencyScore: entry.SelfConsistencyScore,
                SuccessCount: entry.SuccessCount,
                AttemptCount: entry.AttemptCount,
                InputSamples: null,
                OutputSamples: null);
        }
    }

    /// <summary>
    /// Determine which numeric face this function is a CONSTANT translation in, by comparing how consistent
    /// the per-example delta (out−in) is in the poly region [0,nd) vs the log region [nd,2nd) across the
    /// stored samples. Additive functions (+k) are constant in poly; multiplicative (×k) constant in log.
    /// Returns 1 (poly), 2 (log), or 0 when undecidable (too few samples / no signal). Uses the dominant
    /// dim of each region (index 0 and nd) and the coefficient of variation — lower CoV = the true face.
    /// </summary>
    private int ComputePreferredFace(IReadOnlyList<double[]>? inputs, IReadOnlyList<double[]>? outputs)
    {
        if (inputs is null || outputs is null)
            return 0;
        var m = Math.Min(inputs.Count, outputs.Count);
        if (m < 2)
            return 0;

        var nd = Math.Min(_embeddingDim / 2, 21);
        if (nd < 1 || nd >= _embeddingDim)
            return 0;

        var polyCoV = DeltaCoV(inputs, outputs, m, dim: 0);
        var logCoV = DeltaCoV(inputs, outputs, m, dim: nd);
        if (double.IsNaN(polyCoV) && double.IsNaN(logCoV))
            return 0;
        if (double.IsNaN(logCoV))
            return 1;
        if (double.IsNaN(polyCoV))
            return 2;
        return polyCoV <= logCoV ? 1 : 2;
    }

    // Coefficient of variation (std/|mean|) of the per-example delta out[k][dim]-in[k][dim] across samples.
    // NaN when the mean delta is ~0 (no signal in this face). Lower CoV ⇒ a more constant translation here.
    private static double DeltaCoV(IReadOnlyList<double[]> inputs, IReadOnlyList<double[]> outputs, int m, int dim)
    {
        double sum = 0;
        var deltas = new double[m];
        for (var k = 0; k < m; k++)
        {
            deltas[k] = outputs[k][dim] - inputs[k][dim];
            sum += deltas[k];
        }
        var mean = sum / m;
        if (Math.Abs(mean) < 1e-12)
            return double.NaN;
        double varSum = 0;
        for (var k = 0; k < m; k++)
        {
            var d = deltas[k] - mean;
            varSum += d * d;
        }
        return Math.Sqrt(varSum / m) / Math.Abs(mean);
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
