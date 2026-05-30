using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using System.Collections.Generic;

class AnalyzePlatonic
{
    static void Main()
    {
        var checkpointPath = "models/genesis-nova.autosave.checkpoint.json";
        var jsonText = File.ReadAllText(checkpointPath);
        var doc = JsonDocument.Parse(jsonText);
        var root = doc.RootElement;

        Console.WriteLine("=== PLATONIC MODEL QUALITY ANALYSIS ===\n");

        // Get config
        var config = root.GetProperty("Config");
        Console.WriteLine($"HiddenSize: {config.GetProperty("HiddenSize").GetInt32()}");
        Console.WriteLine($"Routes: {config.GetProperty("RouteCount").GetInt32()}\n");

        // Get vocabulary
        var vocab = root.GetProperty("Vocabulary");
        var vocabArray = vocab.EnumerateArray().Select(v => v.GetString()).ToList();
        Console.WriteLine($"Vocabulary ({vocabArray.Count} tokens):");
        Console.WriteLine(string.Join(", ", vocabArray));
        Console.WriteLine();

        // Get Platonic Space
        var ps = root.GetProperty("PlatonicSpace");
        var nodes = ps.GetProperty("Nodes");
        var relations = ps.GetProperty("Relations");

        var nodeList = nodes.EnumerateObject().ToList();
        var relationList = relations.EnumerateObject().ToList();
        
        Console.WriteLine($"Platonic Space Structure:");
        Console.WriteLine($"  Total Nodes: {nodeList.Count}");
        Console.WriteLine($"  Total Relations: {relationList.Count}\n");

        // Analyze nodes by observation count
        Console.WriteLine("Top 20 Most-Observed Concepts:");
        var sortedNodes = nodeList
            .Select(n => new { 
                Name = n.Name,
                Obs = n.Value.GetProperty("ObservationCount").GetInt64()
            })
            .OrderByDescending(x => x.Obs)
            .Take(20)
            .ToList();

        foreach (var node in sortedNodes)
        {
            Console.WriteLine($"  '{node.Name}': {node.Obs} observations");
        }

        Console.WriteLine("\nAnalyzing Relation Quality:");
        
        // Analyze relations - look for arithmetic patterns
        var arithmeticRelations = relationList
            .Where(r => 
                r.Name.Contains("2") || r.Name.Contains("3") || r.Name.Contains("5") ||
                r.Name.Contains("add") || r.Name.Contains("plus") || r.Name.Contains("sum")
            )
            .Take(15)
            .ToList();

        if (arithmeticRelations.Any())
        {
            Console.WriteLine("Arithmetic-related relations found:");
            foreach (var rel in arithmeticRelations)
            {
                try
                {
                    var synthesis = rel.Value.GetProperty("SynthesisContradiction").GetDouble();
                    var obs = rel.Value.GetProperty("ObservationCount").GetInt64();
                    var lastObs = rel.Value.GetProperty("LastObservedContradiction").GetDouble();
                    Console.WriteLine($"  {rel.Name}:");
                    Console.WriteLine($"    Synthesis={synthesis:F3}, Last={lastObs:F3}, Obs={obs}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"    Error: {ex.Message}");
                }
            }
        }
        else
        {
            Console.WriteLine("No arithmetic relations found in first scan.");
            Console.WriteLine("\nSample of all relations:");
            foreach (var rel in relationList.Take(10))
            {
                try
                {
                    var synthesis = rel.Value.GetProperty("SynthesisContradiction").GetDouble();
                    var obs = rel.Value.GetProperty("ObservationCount").GetInt64();
                    Console.WriteLine($"  {rel.Name}: synthesis={synthesis:F3}, obs={obs}");
                }
                catch { }
            }
        }

        // Summary
        Console.WriteLine("\n=== QUALITY ASSESSMENT ===");
        Console.WriteLine($"✓ Model has {vocabArray.Count} tokens in vocabulary");
        Console.WriteLine($"✓ Platonic space contains {nodeList.Count} learned concepts");
        Console.WriteLine($"✓ Relations tracked: {relationList.Count}");
        Console.WriteLine($"✓ Total observations across graph: {sortedNodes.Sum(n => n.Obs):N0}");
        
        var avgObs = sortedNodes.Average(n => n.Obs);
        Console.WriteLine($"✓ Average observations per concept: {avgObs:F0}");
        
        if (avgObs > 100)
        {
            Console.WriteLine("✓ Concepts are well-trained (high observation count)");
        }
        else
        {
            Console.WriteLine("⚠ Concepts need more training");
        }
    }
}
