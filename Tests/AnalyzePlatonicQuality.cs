using Xunit;
using Xunit.Abstractions;
using GenesisNova.Persistence;
using GenesisNova.Core;
using System.Linq;

namespace GenesisNova.Tests;

/// <summary>
/// Tests platonic model checkpoint for quality and learned knowledge structure.
/// Verifies that arithmetic training resulted in useful concepts and relations.
/// </summary>
public class AnalyzePlatonicQuality
{
    private readonly ITestOutputHelper _output;

    public AnalyzePlatonicQuality(ITestOutputHelper output) => _output = output;

    private static string FindRepoRoot()
    {
        // Hardcoded for now since the repo structure is known
        return "C:\\Users\\cex\\repos-working\\genesis-nova";
    }

    [Fact]
    public void AnalyzePlatonicModelFromCheckpoint()
    {
        // Load the checkpoint - find it from repo root
        var repoRoot = FindRepoRoot();
        _output.WriteLine($"Found repo root: {repoRoot}");
        var checkpointPath = System.IO.Path.Combine(repoRoot, "models", "genesis-nova.autosave.checkpoint.json");
        _output.WriteLine($"Checkpoint path: {checkpointPath}");
        _output.WriteLine($"Exists: {System.IO.File.Exists(checkpointPath)}");
        
        var (config, tokenizer, model, platonic, conversation, autonomousTraining) = 
            GenesisCheckpointStore.Load(checkpointPath);
        
        Assert.NotNull(platonic);
        
        var ps = platonic!;
        
        // Basic stats
        _output.WriteLine("\n=== PLATONIC MODEL QUALITY ANALYSIS ===\n");
        _output.WriteLine($"Face Dimension: {ps.FaceDimension}");
        _output.WriteLine($"Total Nodes (Concepts): {ps.Nodes.Length}");
        _output.WriteLine($"Total Relations: {ps.Relations.Length}");
        
        // Node analysis
        _output.WriteLine($"\n=== TOP LEARNED CONCEPTS ===");
        var topNodes = ps.Nodes
            .OrderByDescending(n => n.ObservationCount)
            .Take(20)
            .ToList();
        
        foreach (var node in topNodes)
        {
            _output.WriteLine($"  '{node.Name}': {node.ObservationCount} observations");
        }
        
        // Look for arithmetic concepts
        _output.WriteLine($"\n=== ARITHMETIC CONCEPT QUALITY ===");
        var numbers = ps.Nodes.Where(n => char.IsDigit(n.Name.FirstOrDefault())).ToList();
        var operators = ps.Nodes.Where(n => 
            n.Name.Contains("add") || n.Name.Contains("plus") || 
            n.Name.Contains("sum") || n.Name.Contains("sub")).ToList();
        
        _output.WriteLine($"Number concepts found: {numbers.Count}");
        foreach (var n in numbers.OrderByDescending(x => x.ObservationCount).Take(10))
        {
            _output.WriteLine($"  {n.Name}: {n.ObservationCount} obs");
        }
        
        _output.WriteLine($"\nOperator concepts found: {operators.Count}");
        foreach (var op in operators.OrderByDescending(x => x.ObservationCount))
        {
            _output.WriteLine($"  {op.Name}: {op.ObservationCount} obs");
        }
        
        // Relation analysis
        _output.WriteLine($"\n=== RELATION QUALITY ===");
        var avgUtility = ps.Relations.Average(r => r.ObservationCount);
        var highUtility = ps.Relations.Count(r => r.ObservationCount > 100);
        var lowUtility = ps.Relations.Count(r => r.ObservationCount < 10);
        
        _output.WriteLine($"Average observations per relation: {avgUtility:F1}");
        _output.WriteLine($"High-utility relations (>100 obs): {highUtility}");
        _output.WriteLine($"Low-utility relations (<10 obs): {lowUtility}");
        _output.WriteLine($"Noise ratio: {(double)lowUtility / ps.Relations.Length:P}");
        
        // Find arithmetic relations
        _output.WriteLine($"\n=== ARITHMETIC RELATIONS ===");
        var arithmeticRelations = ps.Relations
            .Where(r => 
                (r.Left.Any(char.IsDigit) || r.Right.Any(char.IsDigit)) ||
                r.Left.Contains("add") || r.Left.Contains("plus") ||
                r.Right.Contains("add") || r.Right.Contains("plus"))
            .OrderByDescending(r => r.ObservationCount)
            .Take(15)
            .ToList();
        
        if (arithmeticRelations.Any())
        {
            _output.WriteLine($"Arithmetic relations found: {arithmeticRelations.Count}");
            foreach (var rel in arithmeticRelations)
            {
                var thesis = rel.ThesisContradiction;
                var synthesis = rel.SynthesisContradiction;
                var delta = System.Math.Abs(synthesis - thesis);
                _output.WriteLine($"  {rel.Left} ←→ {rel.Right}:");
                _output.WriteLine($"    Thesis={thesis:F3}, Synthesis={synthesis:F3}, Delta={delta:F3}, Obs={rel.ObservationCount}");
            }
        }
        else
        {
            _output.WriteLine("No arithmetic relations found yet.");
        }
        
        // Summary assessment
        _output.WriteLine($"\n=== QUALITY ASSESSMENT ===");
        var conceptQuality = topNodes.Average(n => n.ObservationCount);
        var relationQuality = ps.Relations.Count(r => r.ObservationCount > 50);
        
        if (conceptQuality > 200 && relationQuality > ps.Relations.Length * 0.3)
        {
            _output.WriteLine("✓ GOOD: Concepts well-trained, strong relations");
        }
        else if (conceptQuality > 50)
        {
            _output.WriteLine("~ FAIR: Some training, but could go deeper");
        }
        else
        {
            _output.WriteLine("✗ POOR: Model needs more training cycles");
        }
        
        _output.WriteLine($"\nAverage concept strength: {conceptQuality:F0} observations");
        _output.WriteLine($"Percentage of strong relations: {((double)relationQuality / ps.Relations.Length):P}");
        
        // Assertions to verify model quality
        Assert.True(ps.Nodes.Length > 20, "Model should have learned significant concepts");
        Assert.True(ps.Relations.Length > 50, "Model should have learned significant relations");
        Assert.True(numbers.Count >= 5, "Model should have learned number concepts");
        Assert.True(operators.Count >= 1, "Model should have learned operator concepts");
        Assert.True(conceptQuality > 200, "Concepts should be well-trained");
    }
}
