using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;

namespace GenesisNova.Cognition;

/// <summary>
/// Explicit symbolic graph tracking concept relations.
/// Used to compute BridgeConfidence (alignment with embedding space).
/// Based on original Genesis platonic reasoning architecture.
/// </summary>
public class ConceptGraph
{
    private Dictionary<string, ImmutableHashSet<string>> _relations = new();
    private readonly ReaderWriterLockSlim _lock = new();
    
    /// <summary>
    /// Add a directed relation from source to target.
    /// Thread-safe.
    /// </summary>
    public void AddRelation(string source, string target)
    {
        if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
            return;
        
        _lock.EnterWriteLock();
        try
        {
            if (!_relations.TryGetValue(source, out var targets))
            {
                targets = ImmutableHashSet<string>.Empty;
                _relations[source] = targets;
            }
            
            if (!targets.Contains(target))
                _relations[source] = targets.Add(target);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// Get all neighbors of a concept in the graph.
    /// Returns empty set if concept unknown.
    /// Thread-safe.
    /// </summary>
    public ImmutableHashSet<string> GetNeighbors(string concept)
    {
        if (string.IsNullOrWhiteSpace(concept))
            return ImmutableHashSet<string>.Empty;
        
        _lock.EnterReadLock();
        try
        {
            return _relations.TryGetValue(concept, out var neighbors)
                ? neighbors
                : ImmutableHashSet<string>.Empty;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
    
    /// <summary>
    /// Check if a relation exists.
    /// </summary>
    public bool HasRelation(string source, string target)
        => GetNeighbors(source).Contains(target);
    
    /// <summary>
    /// Get total number of concepts in the graph.
    /// </summary>
    public int ConceptCount
    {
        get
        {
            _lock.EnterReadLock();
            try
            {
                return _relations.Count;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }
    
    /// <summary>
    /// Get total number of relations in the graph.
    /// </summary>
    public int RelationCount
    {
        get
        {
            _lock.EnterReadLock();
            try
            {
                return _relations.Values.Sum(set => set.Count);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }
    
    /// <summary>
    /// Get all concepts in the graph.
    /// </summary>
    public IEnumerable<string> AllConcepts
    {
        get
        {
            _lock.EnterReadLock();
            try
            {
                return _relations.Keys.ToList();
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }
    
    /// <summary>
    /// Clear all relations (for testing/reset).
    /// </summary>
    public void Clear()
    {
        _lock.EnterWriteLock();
        try
        {
            _relations.Clear();
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// Get a snapshot of the entire graph structure.
    /// Used for diagnostics and testing.
    /// </summary>
    public ImmutableDictionary<string, ImmutableHashSet<string>> GetSnapshot()
    {
        _lock.EnterReadLock();
        try
        {
            return _relations.ToImmutableDictionary();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
    
    /// <summary>
    /// Observe a training example by adding concept relations.
    /// Called during training to build the symbolic graph.
    /// </summary>
    public void ObserveExample(IEnumerable<string> inputConcepts, IEnumerable<string> outputConcepts)
    {
        var inputs = inputConcepts?.ToList() ?? new();
        var outputs = outputConcepts?.ToList() ?? new();
        
        // Create bidirectional relations
        foreach (var input in inputs)
        {
            foreach (var output in outputs)
            {
                AddRelation(input, output);
                AddRelation(output, input);
            }
        }
        
        // Create self-relations within same set (concepts appear together)
        for (int i = 0; i < inputs.Count; i++)
        {
            for (int j = i + 1; j < inputs.Count; j++)
            {
                AddRelation(inputs[i], inputs[j]);
                AddRelation(inputs[j], inputs[i]);
            }
        }
        
        for (int i = 0; i < outputs.Count; i++)
        {
            for (int j = i + 1; j < outputs.Count; j++)
            {
                AddRelation(outputs[i], outputs[j]);
                AddRelation(outputs[j], outputs[i]);
            }
        }
    }
    
    /// <summary>
    /// Get a structural report of the concept graph.
    /// </summary>
    public string GetStructuralReport()
    {
        var snapshot = GetSnapshot();
        if (snapshot.Count == 0)
            return "Graph is empty";
        
        var lines = new List<string> { $"Concepts: {snapshot.Count}, Relations: {RelationCount}" };
        
        // Show top-connected concepts
        var topConcepts = snapshot
            .OrderByDescending(kvp => kvp.Value.Count)
            .Take(10)
            .Select(kvp => $"  {kvp.Key}: {kvp.Value.Count} connections")
            .ToList();
        
        if (topConcepts.Count > 0)
        {
            lines.Add("Top-connected concepts:");
            lines.AddRange(topConcepts);
        }
        
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Get the total number of relations (alternative to RelationCount property).
    /// </summary>
    public int GetRelationCount() => RelationCount;
}
