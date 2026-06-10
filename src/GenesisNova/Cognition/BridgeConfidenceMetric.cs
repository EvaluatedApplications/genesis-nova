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
    private const double UnknownConfidence = 0.5;
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
        return ComputeJaccard(symbol);
    }

    /// <summary>
    /// Compute BridgeConfidence for only a scoped concept subset.
    /// Keeps signal semantics unchanged while avoiding full-space scans.
    /// </summary>
    public IReadOnlyDictionary<string, double> ComputeForConceptSubset(IEnumerable<string> concepts)
    {
        var result = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var concept in concepts ?? Enumerable.Empty<string>())
        {
            if (string.IsNullOrWhiteSpace(concept))
                continue;
            if (result.ContainsKey(concept))
                continue;
            result[concept] = ComputeJaccard(concept);
        }

        return result;
    }

    /// <summary>
    /// Expand a frontier through symbolic neighborhoods.
    /// Useful for incremental per-turn confidence recomputation on affected concepts.
    /// </summary>
    public IReadOnlyCollection<string> ExpandFrontierConcepts(
        IEnumerable<string> frontierConcepts,
        int maxHops = 1,
        int maxConcepts = 256)
    {
        var limit = Math.Clamp(maxConcepts, 1, 4096);
        var hops = Math.Clamp(maxHops, 0, 4);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var frontier = new Queue<(string Symbol, int Hop)>();

        foreach (var concept in frontierConcepts ?? Enumerable.Empty<string>())
        {
            if (string.IsNullOrWhiteSpace(concept))
                continue;
            if (!seen.Add(concept))
                continue;
            frontier.Enqueue((concept, 0));
            if (seen.Count >= limit)
                return seen.ToArray();
        }

        while (frontier.Count > 0 && seen.Count < limit)
        {
            var (symbol, hop) = frontier.Dequeue();
            if (hop >= hops)
                continue;
            if (!_symbolToNeighbors.TryGetValue(symbol, out var neighbors) || neighbors.Count == 0)
                continue;

            foreach (var neighbor in neighbors)
            {
                if (!seen.Add(neighbor))
                    continue;
                frontier.Enqueue((neighbor, hop + 1));
                if (seen.Count >= limit)
                    break;
            }
        }

        return seen.ToArray();
    }

    /// <summary>
    /// Compute confidence on an incremental scope derived from a frontier (+ optional affected concepts).
    /// </summary>
    public BridgeConfidenceScopeResult ComputeForFrontier(
        IEnumerable<string> frontierConcepts,
        IEnumerable<string>? affectedConcepts = null,
        int maxHops = 1,
        int maxConcepts = 256)
    {
        var frontier = (frontierConcepts ?? Enumerable.Empty<string>())
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var scope = new HashSet<string>(
            ExpandFrontierConcepts(frontier, maxHops, maxConcepts),
            StringComparer.OrdinalIgnoreCase);

        foreach (var concept in affectedConcepts ?? Enumerable.Empty<string>())
        {
            if (string.IsNullOrWhiteSpace(concept))
                continue;
            if (scope.Count >= Math.Clamp(maxConcepts, 1, 4096))
                break;
            scope.Add(concept);
        }

        var scopedScores = ComputeForConceptSubset(scope);
        var frontierAverage = ComputeAverageFromMap(frontier, scopedScores);
        var scopeAverage = scopedScores.Count > 0 ? scopedScores.Values.Average() : UnknownConfidence;

        return new BridgeConfidenceScopeResult(scopedScores, frontierAverage, scopeAverage);
    }
    
    /// <summary>
    /// Compute average BridgeConfidence across all concepts.
    /// </summary>
    public double ComputeAverageForConcepts(IEnumerable<string> concepts)
    {
        var scores = ComputeForConceptSubset(concepts);
        return scores.Count > 0 ? scores.Values.Average() : UnknownConfidence;
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

    private double ComputeJaccard(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
            return UnknownConfidence;

        if (!_symbolToNeighbors.TryGetValue(symbol, out var graphNeighbors) || graphNeighbors.Count == 0)
            return UnknownConfidence;

        var embeddingNeighbors = _getEmbeddingNeighbors(symbol);
        if (embeddingNeighbors.Count == 0)
            return UnknownConfidence;

        var embeddingSet = embeddingNeighbors
            .Select(n => n.Symbol)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        if (embeddingSet.Count == 0)
            return UnknownConfidence;

        var graphSet = new HashSet<string>(graphNeighbors, StringComparer.OrdinalIgnoreCase);
        var intersection = embeddingSet.Intersect(graphSet).Count();
        var union = embeddingSet.Union(graphSet).Count();
        return union == 0 ? UnknownConfidence : (double)intersection / union;
    }

    private static double ComputeAverageFromMap(
        IEnumerable<string> concepts,
        IReadOnlyDictionary<string, double> scores)
    {
        var values = new List<double>();
        foreach (var concept in concepts)
        {
            if (!scores.TryGetValue(concept, out var score))
                continue;
            values.Add(score);
        }

        return values.Count > 0 ? values.Average() : UnknownConfidence;
    }
}

public sealed record BridgeConfidenceScopeResult(
    IReadOnlyDictionary<string, double> ConceptConfidences,
    double FrontierAverageConfidence,
    double ScopeAverageConfidence);
