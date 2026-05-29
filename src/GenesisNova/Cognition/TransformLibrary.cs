using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using TorchSharp;

namespace GenesisNova.Cognition;

/// <summary>
/// Transform element: a learned capability made observable in the platonic space.
/// Based on original Genesis: transforms are first-class objects that can be introspected.
/// </summary>
public record TransformElement(
    string FunctionName,
    torch.Tensor TransformVector,
    double Confidence = 0.5,
    int SuccessfulApplications = 0,
    int FailedApplications = 0)
{
    /// <summary>
    /// Symbolic name in platonic space.
    /// </summary>
    public string Symbol => $"fn:{FunctionName}";
    
    /// <summary>
    /// The embedding of this transform (the vector itself).
    /// </summary>
    public torch.Tensor Embedding => TransformVector;
    
    /// <summary>
    /// Total applications of this transform.
    /// </summary>
    public int TotalApplications => SuccessfulApplications + FailedApplications;
    
    /// <summary>
    /// Success rate of this transform.
    /// </summary>
    public double SuccessRate =>
        TotalApplications == 0
            ? 0.5
            : (double)SuccessfulApplications / TotalApplications;
    
    /// <summary>
    /// Update confidence based on application result.
    /// </summary>
    public TransformElement RecordSuccess() =>
        this with
        {
            SuccessfulApplications = SuccessfulApplications + 1,
            Confidence = Math.Clamp(Confidence + 0.01, 0.0, 1.0)
        };
    
    /// <summary>
    /// Update confidence based on application failure.
    /// </summary>
    public TransformElement RecordFailure() =>
        this with
        {
            FailedApplications = FailedApplications + 1,
            Confidence = Math.Clamp(Confidence - 0.02, 0.0, 1.0)
        };
    
    /// <summary>
    /// Human-readable status of this transform's confidence.
    /// </summary>
    public string GetStatusLabel() =>
        Confidence > 0.8 ? "✓ Confident" :
        Confidence > 0.5 ? "◐ Exploring" :
        "✗ Uncertain";
}

/// <summary>
/// Manages observable transforms: learned capabilities tracked in the platonic space.
/// </summary>
public class TransformLibrary
{
    private Dictionary<string, TransformElement> _transforms = new();
    private readonly object _lock = new();
    
    /// <summary>
    /// Register a learned transform.
    /// </summary>
    public void RegisterTransform(string name, torch.Tensor transformVector)
    {
        if (string.IsNullOrWhiteSpace(name))
            return;
        
        lock (_lock)
        {
            if (_transforms.TryGetValue(name, out var existing))
            {
                _transforms[name] = existing with { TransformVector = transformVector };
            }
            else
            {
                _transforms[name] = new TransformElement(name, transformVector);
            }
        }
    }
    
    /// <summary>
    /// Get a transform by name.
    /// Returns null if not found.
    /// </summary>
    public TransformElement? GetTransform(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;
        
        lock (_lock)
        {
            return _transforms.TryGetValue(name, out var transform) ? transform : null;
        }
    }
    
    /// <summary>
    /// Record successful application of a transform.
    /// </summary>
    public void RecordSuccess(string name)
    {
        lock (_lock)
        {
            if (_transforms.TryGetValue(name, out var transform))
                _transforms[name] = transform.RecordSuccess();
        }
    }
    
    /// <summary>
    /// Record failed application of a transform.
    /// </summary>
    public void RecordFailure(string name)
    {
        lock (_lock)
        {
            if (_transforms.TryGetValue(name, out var transform))
                _transforms[name] = transform.RecordFailure();
        }
    }
    
    /// <summary>
    /// Get all registered transforms.
    /// </summary>
    public IReadOnlyList<TransformElement> GetAllTransforms()
    {
        lock (_lock)
        {
            return _transforms.Values.ToList();
        }
    }
    
    /// <summary>
    /// Get confident transforms (high success rate).
    /// </summary>
    public IReadOnlyList<TransformElement> GetConfidentTransforms(double threshold = 0.8)
    {
        lock (_lock)
        {
            return _transforms.Values
                .Where(t => t.Confidence >= threshold)
                .ToList();
        }
    }
    
    /// <summary>
    /// Get transforms being explored (medium confidence).
    /// </summary>
    public IReadOnlyList<TransformElement> GetExploringTransforms()
    {
        lock (_lock)
        {
            return _transforms.Values
                .Where(t => t.Confidence > 0.5 && t.Confidence < 0.8)
                .ToList();
        }
    }
    
    /// <summary>
    /// Check if system knows a capability.
    /// </summary>
    public bool HasCapability(string name)
    {
        lock (_lock)
        {
            return _transforms.ContainsKey(name);
        }
    }
    
    /// <summary>
    /// Total number of registered transforms.
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _transforms.Count;
            }
        }
    }
    
    /// <summary>
    /// Query what transforms the system knows.
    /// </summary>
    public string QueryKnownCapabilities()
    {
        var confident = GetConfidentTransforms();
        var exploring = GetExploringTransforms();
        
        var lines = new List<string>();
        
        if (confident.Count > 0)
        {
            lines.Add("✓ Confident capabilities:");
            foreach (var t in confident)
            {
                lines.Add($"  - {t.FunctionName}: {t.Confidence:P0} " +
                    $"({t.SuccessfulApplications} successes, {t.FailedApplications} failures)");
            }
        }
        
        if (exploring.Count > 0)
        {
            lines.Add("◐ Exploring capabilities:");
            foreach (var t in exploring)
            {
                lines.Add($"  - {t.FunctionName}: {t.Confidence:P0} " +
                    $"({t.SuccessfulApplications} successes, {t.FailedApplications} failures)");
            }
        }
        
        if (confident.Count == 0 && exploring.Count == 0)
            lines.Add("No transforms registered yet");
        
        return string.Join(Environment.NewLine, lines);
    }
    
    /// <summary>
    /// Clear all transforms (for testing).
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _transforms.Clear();
        }
    }
}
