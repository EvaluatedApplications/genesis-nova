using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// Hypothesis confidence states based on original Genesis epistemic model.
/// </summary>
public enum HypothesisStatus
{
    Conjecture,     // 1-10 confirmations: we're exploring
    Axiom,          // 10+ confirmations: we're confident
    Rejected        // More rejections than confirmations: disproven
}

/// <summary>
/// Represents a learned relation that the system is tracking confidence about.
/// Used for epistemic self-awareness: "what do I know for sure vs. what am I still learning?"
/// </summary>
public record Hypothesis(
    string SourceConcept,
    string TargetConcept,
    string RelationName,
    int ConfirmedCount = 0,
    int RejectedCount = 0)
{
    /// <summary>
    /// Normalized confidence: 0.0 = completely uncertain, 1.0 = completely confident
    /// </summary>
    public double Confidence =>
        (ConfirmedCount + RejectedCount) == 0
            ? 0.5
            : (double)ConfirmedCount / (ConfirmedCount + RejectedCount);
    
    /// <summary>
    /// Status based on confirmation count.
    /// </summary>
    public HypothesisStatus Status =>
        RejectedCount > ConfirmedCount ? HypothesisStatus.Rejected :
        ConfirmedCount >= 10 ? HypothesisStatus.Axiom :
        ConfirmedCount >= 1 ? HypothesisStatus.Conjecture :
        HypothesisStatus.Conjecture;
    
    /// <summary>
    /// Total times this hypothesis has been tested.
    /// </summary>
    public int TotalTests => ConfirmedCount + RejectedCount;
    
    /// <summary>
    /// Record a confirmation of this hypothesis.
    /// </summary>
    public Hypothesis Confirm() =>
        this with { ConfirmedCount = ConfirmedCount + 1 };
    
    /// <summary>
    /// Record a rejection of this hypothesis.
    /// </summary>
    public Hypothesis Reject() =>
        this with { RejectedCount = RejectedCount + 1 };
}

/// <summary>
/// Manages the lifecycle of hypotheses about learned relations.
/// Tracks axioms (high confidence), conjectures (medium), and rejections.
/// </summary>
public class HypothesisTracker
{
    private readonly Dictionary<(string, string), Hypothesis> _hypotheses = new();
    private readonly object _lock = new();
    
    /// <summary>
    /// Get or create a hypothesis about a relation.
    /// </summary>
    public Hypothesis GetOrCreateHypothesis(string source, string target, string relation = "related_to")
    {
        var key = (source, target);
        
        lock (_lock)
        {
            if (!_hypotheses.TryGetValue(key, out var hypothesis))
            {
                hypothesis = new Hypothesis(source, target, relation);
                _hypotheses[key] = hypothesis;
            }
            return hypothesis;
        }
    }
    
    /// <summary>
    /// Record a confirmation of a hypothesis.
    /// </summary>
    public void ConfirmHypothesis(string source, string target)
    {
        var key = (source, target);
        lock (_lock)
        {
            if (_hypotheses.TryGetValue(key, out var hypothesis))
                _hypotheses[key] = hypothesis.Confirm();
            else
                _hypotheses[key] = new Hypothesis(source, target, "related_to").Confirm();
        }
    }
    
    /// <summary>
    /// Record a rejection of a hypothesis.
    /// </summary>
    public void RejectHypothesis(string source, string target)
    {
        var key = (source, target);
        lock (_lock)
        {
            if (_hypotheses.TryGetValue(key, out var hypothesis))
                _hypotheses[key] = hypothesis.Reject();
            else
                _hypotheses[key] = new Hypothesis(source, target, "related_to").Reject();
        }
    }
    
    /// <summary>
    /// Get all axioms (high confidence relations).
    /// </summary>
    public IReadOnlyList<Hypothesis> GetAxioms()
    {
        lock (_lock)
        {
            return _hypotheses.Values
                .Where(h => h.Status == HypothesisStatus.Axiom)
                .ToList();
        }
    }
    
    /// <summary>
    /// Get all conjectures (medium confidence relations being explored).
    /// </summary>
    public IReadOnlyList<Hypothesis> GetConjectures()
    {
        lock (_lock)
        {
            return _hypotheses.Values
                .Where(h => h.Status == HypothesisStatus.Conjecture)
                .ToList();
        }
    }
    
    /// <summary>
    /// Get all rejected hypotheses (disproven relations).
    /// </summary>
    public IReadOnlyList<Hypothesis> GetRejected()
    {
        lock (_lock)
        {
            return _hypotheses.Values
                .Where(h => h.Status == HypothesisStatus.Rejected)
                .ToList();
        }
    }
    
    /// <summary>
    /// Get total number of tracked hypotheses.
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _hypotheses.Count;
            }
        }
    }
    
    /// <summary>
    /// Query what the system is certain about.
    /// </summary>
    public string QueryWhatICertainlyKnow()
    {
        var axioms = GetAxioms();
        if (axioms.Count == 0)
            return "✓ Nothing is yet confirmed as axiom-level knowledge";
        
        var lines = new List<string> { "✓ Knowledge I'm certain about (axioms):" };
        foreach (var axiom in axioms)
        {
            lines.Add($"  - {axiom.SourceConcept} → {axiom.TargetConcept} " +
                $"(confirmed: {axiom.ConfirmedCount}x, confidence: {axiom.Confidence:P0})");
        }
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Query what the system is exploring.
    /// </summary>
    public string QueryWhatIExploring()
    {
        var conjectures = GetConjectures();
        if (conjectures.Count == 0)
            return "? No conjectures currently being explored";
        
        var lines = new List<string> { "? Knowledge I'm currently exploring (conjectures):" };
        foreach (var conjecture in conjectures)
        {
            lines.Add($"  - {conjecture.SourceConcept} → {conjecture.TargetConcept} " +
                $"(tests: {conjecture.TotalTests}, confidence: {conjecture.Confidence:P0})");
        }
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Query what the system has ruled out.
    /// </summary>
    public string QueryWhatIRuledOut()
    {
        var rejected = GetRejected();
        if (rejected.Count == 0)
            return "✗ No relations have been disproven";
        
        var lines = new List<string> { "✗ Knowledge I've ruled out (rejected):" };
        foreach (var rej in rejected)
        {
            lines.Add($"  - {rej.SourceConcept} → {rej.TargetConcept} " +
                $"(rejections: {rej.RejectedCount}, confirmations: {rej.ConfirmedCount})");
        }
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Get the status of a hypothesis about a concept.
    /// </summary>
    public HypothesisStatus GetHypothesisStatus(string concept)
    {
        lock (_lock)
        {
            // Find any hypothesis involving this concept
            var related = _hypotheses.Values
                .FirstOrDefault(h => h.SourceConcept == concept || h.TargetConcept == concept);
            
            return related?.Status ?? HypothesisStatus.Conjecture;
        }
    }
    
    /// <summary>
    /// Get a comprehensive status report.
    /// </summary>
    public string GetStatusReport()
    {
        var axioms = GetAxioms().Count;
        var conjectures = GetConjectures().Count;
        var rejected = GetRejected().Count;
        
        var lines = new List<string>
        {
            $"Total hypotheses: {Count}",
            $"  ✓ Axioms (confident): {axioms}",
            $"  ? Conjectures (exploring): {conjectures}",
            $"  ✗ Rejected (disproven): {rejected}"
        };
        
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Clear all hypotheses (for testing/reset).
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _hypotheses.Clear();
        }
    }
}
