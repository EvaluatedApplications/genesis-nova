using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// GEOMETRIC RETRIEVAL ROUTE (re-promoted "position IS identity"): after training category clusters, a
/// single-concept query is answered by the NEAREST stored concept in the semantic face (the lattice VP-Tree),
/// via the `platonic-geometric` decision path — NOT the relation edge. This only works because the push/pull
/// dynamics now SEPARATE the space (frozen-dilution fix + attraction floor): a collapsed cone has no nearest-
/// neighbour signal. So this test is the behavioural capture of the whole geometry-fix arc. Production dims.
/// </summary>
public sealed class GeometricRetrievalRouteTests
{
    private readonly ITestOutputHelper _out;
    public GeometricRetrievalRouteTests(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void GeometricRetrieval_EmergesAfterTraining_NearestConceptViaPosition()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, model, memory, config);
        var inference = new GenesisInferenceEngine(tok, model, memory, null,
            transformAccumulator: trainer.TransformAccumulator, foldPathDiscovery: trainer.FoldPathDiscovery);
        trainer.SetInferencePolicy(inference);

        // Category clusters: each item → its category (a shared hub). Members of a cluster should end up NEAR
        // each other / their category, and FAR from other clusters, once the dynamics separate the space.
        var clusters = new Dictionary<string, string[]>
        {
            ["fruit"] = new[] { "apple", "banana", "cherry", "grape" },
            ["animal"] = new[] { "cat", "dog", "fox", "owl" },
            ["metal"] = new[] { "iron", "gold", "copper", "zinc" },
            ["color"] = new[] { "crimson", "azure", "violet", "amber" },
        };
        var train = clusters.SelectMany(kv => kv.Value.Select(m => new GenesisExample(m, kv.Key))).ToList();
        var inCluster = new Dictionary<string, HashSet<string>>();
        foreach (var kv in clusters)
            foreach (var m in kv.Value)
                inCluster[m] = new HashSet<string>(kv.Value.Append(kv.Key), StringComparer.OrdinalIgnoreCase);

        var rng = new Random(7);
        void Shuffle(List<GenesisExample> d)
        { for (var i = d.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (d[i], d[j]) = (d[j], d[i]); } }

        // A query is "geometrically right" if the answer is its category OR a co-cluster member (proves the
        // cluster is geometrically coherent), regardless of which exact in-cluster concept is nearest.
        (int correct, int viaGeo) Probe()
        {
            int c = 0, g = 0;
            foreach (var (m, set) in inCluster)
            {
                var res = inference.Generate(new GenerationRequest(m, 4));
                if (set.Contains(res.Output.Trim())) c++;
                if (res.DecisionPath.Contains("geometric", StringComparison.OrdinalIgnoreCase)) g++;
            }
            return (c, g);
        }

        var pool = train.ToList();
        var idx = 0; var steps = 0;
        const int maxSteps = 16000, probeEvery = 800;
        for (var s = 0; s < maxSteps; s++)
        {
            if (idx >= pool.Count) { Shuffle(pool); idx = 0; }
            trainer.TrainStep(pool[idx++]); steps++;
            if (steps % probeEvery == 0)
            {
                var (c, g) = Probe();
                Console.Error.WriteLine($"[geo] step {steps} in-cluster {c}/{inCluster.Count} viaGeometric {g}");
                if (c >= inCluster.Count - 2 && g >= inCluster.Count / 2) break;
            }
        }

        var geo = memory.SummarizePushPullGeometry();
        _out.WriteLine($"separation {geo.Separation:F3} (related {geo.RelatedMean:F3}, unrelated {geo.UnrelatedMean:F3})");
        var (correct, viaGeo) = Probe();
        foreach (var (m, set) in inCluster)
        {
            var r = inference.Generate(new GenerationRequest(m, 4));
            _out.WriteLine($"  '{m,-8}' path={r.DecisionPath} -> '{r.Output.Trim()}'");
        }
        _out.WriteLine($"GEOMETRIC: in-cluster {correct}/{inCluster.Count}, via geometric route {viaGeo}/{inCluster.Count}");

        // The space must have SEPARATED (related closer than unrelated) — the un-collapse precondition.
        Assert.True(geo.Separation > 0.05, $"space must be separated for geometric retrieval; separation={geo.Separation:F3}");
        // Most queries resolve to their own cluster, and a real share go through the geometric route.
        Assert.True(correct >= inCluster.Count - 3, $"geometric retrieval should resolve to the right cluster; {correct}/{inCluster.Count}");
        Assert.True(viaGeo >= inCluster.Count / 2, $"a real share must route via platonic-geometric; {viaGeo}/{inCluster.Count}");
    }
}
