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
    IReadOnlyList<double[]>? InputSamples = null,
    IReadOnlyList<double[]>? OutputSamples = null);

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
            var updated = existing with
            {
                Vector = blended,
                ObservationCount = newCount,
                InputSamples = AppendSample(existing.InputSamples, inputEmbedding),
                OutputSamples = AppendSample(existing.OutputSamples, outputEmbedding),
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
