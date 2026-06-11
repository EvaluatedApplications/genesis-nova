using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using TorchSharp;

namespace GenesisNova.Cognition;

/// <summary>
/// Complement pair manager: tracks explicit negations for every concept.
/// Based on conservation principle from original Genesis (G4).
/// 
/// For every concept x, ensures ¬x exists and embed(x) + embed(¬x) ≈ 0
/// </summary>
public class ComplementPairManager
{
    private Dictionary<string, string> _complement = new();
    private readonly object _lock = new();
    
    /// <summary>
    /// Get the complement of a concept.
    /// If not explicitly registered, computes it as "¬" prefix.
    /// </summary>
    public string GetComplement(string concept)
    {
        if (string.IsNullOrWhiteSpace(concept))
            return concept;
        
        // Handle already-negated concepts
        if (concept.StartsWith("¬", StringComparison.Ordinal))
            return concept[1..]; // Double negation: ¬(¬x) = x
        
        lock (_lock)
        {
            if (_complement.TryGetValue(concept, out var comp))
                return comp;
            
            var negated = $"¬{concept}";
            _complement[concept] = negated;
            _complement[negated] = concept;
            return negated;
        }
    }
    
    /// <summary>
    /// Register a complement pair explicitly.
    /// Useful when embeddings have known complementary structure.
    /// </summary>
    public void RegisterComplement(string concept, string complement)
    {
        if (string.IsNullOrWhiteSpace(concept) || string.IsNullOrWhiteSpace(complement))
            return;
        
        lock (_lock)
        {
            _complement[concept] = complement;
            _complement[complement] = concept;
        }
    }
    
    /// <summary>
    /// Check if two concepts are complements.
    /// </summary>
    public bool AreComplements(string concept1, string concept2)
        => GetComplement(concept1) == concept2;
    
    /// <summary>
    /// Get all registered complements.
    /// </summary>
    public IReadOnlyDictionary<string, string> GetAllComplements()
    {
        lock (_lock)
        {
            return _complement.ToImmutableDictionary();
        }
    }
    
    /// <summary>
    /// Clear all registered complements (for testing).
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _complement.Clear();
        }
    }
}

/// <summary>
/// Verifies conservation law: complement embeddings should sum to approximately zero.
/// Used for validating that the platonic space maintains G4 (Conservation).
/// </summary>
public class ConservationVerifier
{
    private readonly Func<string, torch.Tensor?> _getEmbedding;
    private readonly double _tolerance;
    private readonly ComplementPairManager _complements;
    
    public ConservationVerifier(
        Func<string, torch.Tensor?> getEmbedding,
        ComplementPairManager complements,
        double tolerance = 0.01)
    {
        _getEmbedding = getEmbedding;
        _complements = complements;
        _tolerance = tolerance;
    }
    
    /// <summary>
    /// Verify that complement pairs satisfy conservation: x + ¬x ≈ 0
    /// Returns true if conservation is maintained within tolerance.
    /// </summary>
    public bool VerifyConservation(string concept)
    {
        var embedding = _getEmbedding(concept);
        if (embedding is null)
            return false;
        
        var complement = _complements.GetComplement(concept);
        var complementEmbedding = _getEmbedding(complement);
        if (complementEmbedding is null)
            return false;
        
        // Compute sum and check if approximately zero
        try
        {
            using var sum = embedding + complementEmbedding;
            var norm = sum.norm().item<double>();
            return norm < _tolerance;
        }
        catch
        {
            return false;
        }
    }
    
    /// <summary>
    /// Verify conservation across multiple concepts.
    /// Returns fraction of concepts that satisfy conservation.
    /// </summary>
    public double VerifyConservationAcross(IEnumerable<string> concepts)
    {
        var conceptList = concepts?.ToList() ?? new();
        if (conceptList.Count == 0)
            return 0.0;
        
        var verified = conceptList.Count(VerifyConservation);
        return (double)verified / conceptList.Count;
    }
    
    /// <summary>
    /// Get conservation deficit for a concept.
    /// Returns the norm of (x + ¬x), ideally close to 0.
    /// </summary>
    public double GetConservationDeficit(string concept)
    {
        var embedding = _getEmbedding(concept);
        if (embedding is null)
            return double.NaN;
        
        var complement = _complements.GetComplement(concept);
        var complementEmbedding = _getEmbedding(complement);
        if (complementEmbedding is null)
            return double.NaN;
        
        try
        {
            using var sum = embedding + complementEmbedding;
            return sum.norm().item<double>();
        }
        catch
        {
            return double.NaN;
        }
    }
}

/// <summary>
/// Enforces conservation law: ensures all concepts have explicit complements with proper embeddings.
/// </summary>
public class ConservationEnforcer
{
    private readonly ComplementPairManager _complements;
    private readonly Func<string, torch.Tensor> _getOrCreateEmbedding;
    private readonly Action<string, torch.Tensor> _registerEmbedding;
    
    public ConservationEnforcer(
        ComplementPairManager complements,
        Func<string, torch.Tensor> getOrCreateEmbedding,
        Action<string, torch.Tensor> registerEmbedding)
    {
        _complements = complements;
        _getOrCreateEmbedding = getOrCreateEmbedding;
        _registerEmbedding = registerEmbedding;
    }
    
    /// <summary>
    /// Enforce the hard G4 conservation law for a concept and its complement: embed(¬x) = −embed(x),
    /// so embed(x) + embed(¬x) = 0 exactly. This is RE-PROJECTION (it rewrites the complement's
    /// embedding to conserve), not mere validation — conforming to the canonical hard-negation rule
    /// (EmbeddingSpace.NegateEmbedding) rather than the previous soft 0.95/0.05 coupling.
    /// </summary>
    public void EnforceForConcept(string concept)
    {
        var embedding = _getOrCreateEmbedding(concept);
        var complement = _complements.GetComplement(concept);

        // Hard negation: re-project the complement onto −embed(x) so the pair sums to 0.
        using var complementEmbedding = -embedding;
        _registerEmbedding(complement, complementEmbedding);
    }

    /// <summary>
    /// Enforce conservation across all provided concepts (re-projecting each complement).
    /// </summary>
    public void EnforceForAll(IEnumerable<string> concepts)
    {
        foreach (var concept in concepts ?? Enumerable.Empty<string>())
        {
            EnforceForConcept(concept);
        }
    }

    /// <summary>
    /// Re-project a concept/complement pair to conserve charge when BOTH already carry signal.
    /// Rather than blindly negating one side, it averages x and −(¬x) into a single canonical
    /// positive face, then re-projects: embed(x) ← mean, embed(¬x) ← −mean. This keeps the pair
    /// on the conservation manifold (x + ¬x = 0) while respecting evidence accumulated on both faces.
    /// </summary>
    public void ReprojectPair(string concept)
    {
        var complementName = _complements.GetComplement(concept);
        var pos = _getOrCreateEmbedding(concept);
        var neg = _getOrCreateEmbedding(complementName);

        // Canonical positive face = mean(pos, −neg); negative face = −canonical.
        using var negFlipped = -neg;
        using var canonical = (pos + negFlipped) * 0.5;
        using var canonicalNeg = -canonical;
        _registerEmbedding(concept, canonical);
        _registerEmbedding(complementName, canonicalNeg);
    }
}
