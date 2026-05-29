using GenesisNova.Core;
using System.Text.Json;

namespace GenesisNova.Tests;

/// <summary>
/// Inspects the learned symbolic structure from the active checkpoint.
/// Reveals patterns in platonic space, indicating what the model has learned.
/// </summary>
public sealed class TransformAccumulatorInspectionTests
{
    [Fact]
    public void WhenCheckpointLoaded_ThenAnalyzeLearnedStructure()
    {
        // Arrange
        var checkpointPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova", "state", "genesis-nova.autosave.checkpoint.json");

        if (!File.Exists(checkpointPath))
        {
            Console.WriteLine($"[SKIPPED] Checkpoint not found at {checkpointPath}");
            return;
        }

        var json = JsonDocument.Parse(File.ReadAllText(checkpointPath));
        var root = json.RootElement;

        // Extract platonic space info
        var psElem = root.GetProperty("PlatonicSpace");
        var nodesElem = psElem.GetProperty("Nodes");
        var relationsElem = psElem.GetProperty("Relations");

        var nodeCount = nodesElem.GetArrayLength();
        var relationCount = relationsElem.GetArrayLength();

        Console.WriteLine($"\n=== PLATONIC SPACE STRUCTURE ===");
        Console.WriteLine($"Total nodes: {nodeCount}");
        Console.WriteLine($"Total relations: {relationCount}");

        // Find numeric nodes
        var numericNodes = new List<(string name, int value)>();
        foreach (var node in nodesElem.EnumerateArray())
        {
            var name = node.GetProperty("Name").GetString() ?? "";
            if (int.TryParse(name, out var val))
            {
                numericNodes.Add((name, val));
            }
        }

        Console.WriteLine($"Numeric nodes: {numericNodes.Count}");
        if (numericNodes.Count > 0)
        {
            Console.WriteLine("  Examples: " + string.Join(", ", numericNodes.Take(15).Select(n => n.name)));
        }

        // Analyze relations for arithmetic patterns
        Console.WriteLine($"\n=== ARITHMETIC PATTERN ANALYSIS ===");
        var addTriples = new List<(int a, int b, int c)>();
        var relationsByOp = new Dictionary<string, int>();

        foreach (var rel in relationsElem.EnumerateArray())
        {
            var left = rel.GetProperty("Left").GetString() ?? "";
            var right = rel.GetProperty("Right").GetString() ?? "";
            var obsCount = rel.GetProperty("ObservationCount").GetInt64();

            // Track operations
            if (!relationsByOp.ContainsKey(left)) relationsByOp[left] = 0;
            relationsByOp[left]++;

            // Look for add patterns: a --left--> right, where left="add" and right is numeric
            if (left == "add" && int.TryParse(right, out var b))
            {
                // Find if there's a relation from some number to b: would indicate a + X = b
                foreach (var relB in relationsElem.EnumerateArray())
                {
                    if (relB.GetProperty("Right").GetString() == right && 
                        int.TryParse(relB.GetProperty("Left").GetString() ?? "", out var a) &&
                        relB.GetProperty("Left").GetString() == "add")
                    {
                        // Found: add ---> numeric, indicating strong association
                    }
                }
            }
        }

        Console.WriteLine("Top operations by relation count:");
        foreach (var (op, count) in relationsByOp.OrderByDescending(x => x.Value).Take(10))
        {
            Console.WriteLine($"  '{op}': {count} relations");
        }

        // Check ADD specifically
        var addRelations = 0;
        var addNumericRelations = 0;
        var addFaceRelations = 0;

        foreach (var rel in relationsElem.EnumerateArray())
        {
            if (rel.GetProperty("Left").GetString() == "add")
            {
                addRelations++;
                var right = rel.GetProperty("Right").GetString() ?? "";
                if (int.TryParse(right, out _))
                {
                    addNumericRelations++;
                }
                else if (right.StartsWith("face:"))
                {
                    addFaceRelations++;
                }
            }
        }

        Console.WriteLine($"\n=== ADD OPERATION ANALYSIS ===");
        Console.WriteLine($"Add relations total: {addRelations}");
        Console.WriteLine($"  → to numeric nodes: {addNumericRelations}");
        Console.WriteLine($"  → to face nodes: {addFaceRelations}");

        // Find strongest numeric relations that might indicate learned addition
        Console.WriteLine($"\n=== STRONGEST NUMERIC RELATIONS ===");
        var numericRels = new List<(string left, string right, long obs)>();
        foreach (var rel in relationsElem.EnumerateArray())
        {
            var left = rel.GetProperty("Left").GetString() ?? "";
            var right = rel.GetProperty("Right").GetString() ?? "";
            if (int.TryParse(left, out _) && int.TryParse(right, out _))
            {
                numericRels.Add((left, right, rel.GetProperty("ObservationCount").GetInt64()));
            }
        }

        if (numericRels.Count > 0)
        {
            Console.WriteLine($"Found {numericRels.Count} numeric-to-numeric relations");
            Console.WriteLine("Top 10:");
            foreach (var (left, right, obs) in numericRels.OrderByDescending(r => r.obs).Take(10))
            {
                Console.WriteLine($"  {left} → {right}: {obs} observations");
            }
        }

        // Summary
        Console.WriteLine($"\n=== INTERPRETATION ===");
        if (addNumericRelations > 100 && numericRels.Count > 5)
        {
            Console.WriteLine("✓ Model has learned arithmetic structure:");
            Console.WriteLine($"  - Strong add operation: {addRelations} total relations");
            Console.WriteLine($"  - Numeric associations: {addNumericRelations}");
            Console.WriteLine($"  - Direct numeric patterns: {numericRels.Count}");
            Console.WriteLine("✓ Single numeric-face add transform appears to have emerged");
        }
        else if (addRelations > 50)
        {
            Console.WriteLine("~ Partial arithmetic learning:");
            Console.WriteLine($"  - Add operation exists: {addRelations} relations");
            Console.WriteLine($"  - But limited numeric association: {addNumericRelations}");
            Console.WriteLine("  - May still be in early learning phase");
        }
        else
        {
            Console.WriteLine("✗ Limited arithmetic structure detected");
            Console.WriteLine("  - Training may not have focused on arithmetic");
        }

        Assert.True(nodeCount > 0, "Platonic space should have learned nodes");
    }
}
