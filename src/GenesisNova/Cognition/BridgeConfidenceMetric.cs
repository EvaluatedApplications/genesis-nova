using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// BridgeConfidence metric: measures alignment between embedding space and symbolic graph.
/// Based on ionization energy from original Genesis platonic theory.
/// 
/// High confidence = concept is stable (embedding matches graph structure)
/// Low confidence = concept is ionizable (accepts new learning)
/// </summary>
public class BridgeConfidenceMetric
{
    private readonly IReadOnlyDictionary<string, ImmutableHashSet<string>> _symbolToNeighbors;
    private readonly Func<string, IReadOnlyList<(string Symbol, double Distance)>> _getEmbeddingNeighbors;
    
    public BridgeConfidenceMetric(
        IReadOnlyDictionary<string, ImmutableHashSet<string>> symbolToNeighbors,
        Func<string, IReadOnlyList<(string Symbol, double Distance)>> getEmbeddingNeighbors)
    {
        _symbolToNeighbors = symbolToNeighbors;
        _getEmbeddingNeighbors = getEmbeddingNeighbors;
    }
    
    /// <summary>
    /// Compute BridgeConfidence for a concept.
    /// Returns 0.5 if concept is unknown.
    /// Returns Jaccard(embedding neighbors, graph neighbors) otherwise.
    /// </summary>
    public double ComputeForConcept(string symbol)
    {
        // If concept has no symbolic relations, we're uncertain
        if (!_symbolToNeighbors.TryGetValue(symbol, out var graphNeighbors) || graphNeighbors.Count == 0)
            return 0.5;
        
        // Get k nearest neighbors in embedding space
        var embeddingNeighbors = _getEmbeddingNeighbors(symbol);
        if (embeddingNeighbors.Count == 0)
            return 0.5;
        
        // Convert to sets for Jaccard
        var embeddingSet = embeddingNeighbors.Select(n => n.Symbol).ToHashSet();
        var graphSet = new HashSet<string>(graphNeighbors);
        
        // Jaccard similarity = |A ∩ B| / |A ∪ B|
        var intersection = embeddingSet.Intersect(graphSet).Count();
        var union = embeddingSet.Union(graphSet).Count();
        
        if (union == 0) return 0.5;
        
        return (double)intersection / union;
    }
    
    /// <summary>
    /// Compute average BridgeConfidence across all concepts.
    /// </summary>
    public double ComputeAverageForConcepts(IEnumerable<string> concepts)
    {
        var confidences = concepts
            .Select(ComputeForConcept)
            .ToList();
        
        return confidences.Count > 0 ? confidences.Average() : 0.5;
    }
    
    /// <summary>
    /// Compute ionization energy (inverse of bridge confidence).
    /// Low energy = high confidence = stable
    /// High energy = low confidence = ionizable
    /// </summary>
    public double ComputeIonizationEnergy(string symbol)
        => 1.0 - ComputeForConcept(symbol);
    
    /// <summary>
    /// Compute ionization energy from a confidence value (0 to 1).
    /// </summary>
    public double ComputeIonizationEnergyFromConfidence(double confidence)
        => 1.0 - Math.Clamp(confidence, 0.0, 1.0);
    
    /// <summary>
    /// Determine if space needs "portal bridge" (realignment) operation.
    /// Triggers when average confidence drops below threshold.
    /// </summary>
    public bool ShouldExecutePortalBridge(IEnumerable<string> concepts, double threshold = 0.3)
        => ComputeAverageForConcepts(concepts) < threshold;
}
