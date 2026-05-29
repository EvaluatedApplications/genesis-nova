using System.Collections.Immutable;

namespace GenesisNova.Core;

/// <summary>
/// A relation in the symbolic graph.
/// Source --(operation)--> Target
/// </summary>
public record GraphRelation(
    string Source,
    string Target,
    string Operation);

/// <summary>
/// Tracks symbolic relations and can verify consistency of predictions.
/// </summary>
public class SymbolicGraph
{
    private readonly Dictionary<string, HashSet<GraphRelation>> _outgoing = new();
    private readonly Dictionary<string, HashSet<GraphRelation>> _incoming = new();
    private readonly HashSet<string> _symbols = new();
    
    /// <summary>
    /// Add a fact: Source --(operation)--> Target
    /// </summary>
    public void AddRelation(string source, string target, string operation)
    {
        _symbols.Add(source);
        _symbols.Add(target);
        
        var relation = new GraphRelation(source, target, operation);
        
        if (!_outgoing.ContainsKey(source))
            _outgoing[source] = new();
        _outgoing[source].Add(relation);
        
        if (!_incoming.ContainsKey(target))
            _incoming[target] = new();
        _incoming[target].Add(relation);
    }
    
    /// <summary>
    /// Check if a prediction (source + operation = target) is consistent with known facts.
    /// </summary>
    public bool VerifyRelation(string source, string target, string operation)
    {
        if (!_outgoing.TryGetValue(source, out var outgoing))
            return true;  // No constraints, so it's ok
        
        // Check if this exact relation exists
        if (outgoing.Any(r => r.Target == target && r.Operation == operation))
            return true;
        
        // Check if this operation exists (even to different target)
        if (outgoing.Any(r => r.Operation == operation))
            return true;  // Operation is known, different result might be due to input
        
        return false;  // Unknown operation or contradictory result
    }
    
    /// <summary>
    /// Get all known relations from source.
    /// </summary>
    public IReadOnlyList<GraphRelation>? GetOutgoing(string source)
    {
        return _outgoing.TryGetValue(source, out var rels)
            ? rels.ToList()
            : null;
    }
    
    /// <summary>
    /// Check if a symbol is known.
    /// </summary>
    public bool Contains(string symbol)
    {
        return _symbols.Contains(symbol);
    }
    
    /// <summary>
    /// Get all known symbols.
    /// </summary>
    public IReadOnlySet<string> AllSymbols => _symbols;
    
    /// <summary>
    /// Get all known relations.
    /// </summary>
    public IReadOnlyList<GraphRelation> AllRelations
    {
        get
        {
            var all = new List<GraphRelation>();
            foreach (var rels in _outgoing.Values)
                all.AddRange(rels);
            return all;
        }
    }
}
