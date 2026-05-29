using System.Collections.Immutable;

namespace GenesisNova.Core;

/// <summary>
/// A learned transform from input embedding space to output embedding space.
/// 
/// Core learning algorithm from Genesis Engine:
/// - One example teaches one transform (no gradient descent, pure vector arithmetic)
/// - Transforms average as new examples arrive
/// - Embeddings adapt to make transforms consistent
/// </summary>
public record Transform(
    string FunctionName,
    double[] Vector,
    int ObservationCount,
    double Confidence = 0.0);

/// <summary>
/// Accumulates and manages learned transforms.
/// 
/// This is the heart of the Genesis Engine approach:
/// T(f) = avg(embed(output_i) - embed(input_i)) for all examples of f
/// </summary>
public class TransformAccumulator
{
    private readonly Dictionary<string, Transform> _transforms = new();
    private readonly int _embeddingDim;
    
    public TransformAccumulator(int embeddingDim)
    {
        _embeddingDim = embeddingDim;
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
            // Average with existing transform (confidence-weighted)
            var newCount = existing.ObservationCount + 1;
            var alpha = 1.0 / newCount;  // Running average
            
            var blended = new double[_embeddingDim];
            for (int i = 0; i < _embeddingDim; i++)
                blended[i] = existing.Vector[i] * (1.0 - alpha) + delta[i] * alpha;
            
            _transforms[functionName] = new Transform(
                functionName,
                blended,
                newCount,
                Math.Min(1.0, existing.Confidence + 0.05));  // Confidence grows with observations
        }
        else
        {
            // First observation of this transform
            _transforms[functionName] = new Transform(
                functionName,
                delta,
                1,
                0.1);  // Low confidence initially
        }
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
