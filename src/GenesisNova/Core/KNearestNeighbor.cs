namespace GenesisNova.Core;

/// <summary>
/// Maps embeddings back to symbols via k-nearest neighbor search.
/// </summary>
public class KNearestNeighbor
{
    private readonly Dictionary<string, double[]> _symbolEmbeddings = new();
    private readonly int _embeddingDim;
    private const int K = 1;  // For now, use nearest neighbor only
    
    public KNearestNeighbor(int embeddingDim)
    {
        _embeddingDim = embeddingDim;
    }
    
    public IReadOnlyDictionary<string, double[]> SymbolEmbeddings => _symbolEmbeddings;
    
    /// <summary>
    /// Register a symbol with its embedding (typically during encoding).
    /// </summary>
    public void Register(string symbol, double[] embedding)
    {
        if (embedding.Length != _embeddingDim)
            throw new ArgumentException($"Expected embedding dim {_embeddingDim}, got {embedding.Length}");
        
        _symbolEmbeddings[symbol] = embedding;
    }
    
    /// <summary>
    /// Find the nearest symbol to a query embedding.
    /// Returns (symbol, distance).
    /// </summary>
    public (string Symbol, double Distance)? FindNearest(double[] queryEmbedding)
    {
        if (_symbolEmbeddings.Count == 0)
            return null;
        
        if (queryEmbedding.Length != _embeddingDim)
            throw new ArgumentException($"Expected embedding dim {_embeddingDim}, got {queryEmbedding.Length}");
        
        double bestDist = double.MaxValue;
        string? bestSymbol = null;
        
        foreach (var (symbol, embedding) in _symbolEmbeddings)
        {
            var dist = EuclideanDistance(queryEmbedding, embedding);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestSymbol = symbol;
            }
        }
        
        return bestSymbol is null ? null : (bestSymbol, bestDist);
    }
    
    /// <summary>
    /// Find k-nearest neighbors (not just 1).
    /// </summary>
    public List<(string Symbol, double Distance)> FindKNearest(double[] queryEmbedding, int k = 5)
    {
        var results = new List<(string, double)>();
        
        if (_symbolEmbeddings.Count == 0)
            return results;
        
        foreach (var (symbol, embedding) in _symbolEmbeddings)
        {
            var dist = EuclideanDistance(queryEmbedding, embedding);
            results.Add((symbol, dist));
        }
        
        results.Sort((a, b) => a.Item2.CompareTo(b.Item2));
        
        return results.Take(Math.Min(k, results.Count)).ToList();
    }
    
    /// <summary>
    /// Confidence score: inverse of distance (0 to 1).
    /// </summary>
    public double DistanceToConfidence(double distance)
    {
        return 1.0 / (1.0 + distance);
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
