using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class PlatonicArithmeticFaceTests
{
    [Fact]
    public void WhenCreatingUnknownConcept_ThenFaceIsDeterministicAcrossSeeds()
    {
        var a = new PlatonicSpaceMemory(faceDimension: 16, seed: 1);
        var b = new PlatonicSpaceMemory(faceDimension: 16, seed: 999);

        a.ObserveContradiction("nebula", "cluster", 0.25);
        b.ObserveContradiction("nebula", "cluster", 0.25);

        var nodeA = a.ExportSnapshot().Nodes.Single(n => n.Name == "nebula");
        var nodeB = b.ExportSnapshot().Nodes.Single(n => n.Name == "nebula");

        Assert.Equal(nodeA.PositiveFace.Length, nodeB.PositiveFace.Length);
        for (var i = 0; i < nodeA.PositiveFace.Length; i++)
            Assert.True(Math.Abs(nodeA.PositiveFace[i] - nodeB.PositiveFace[i]) < 1e-12);
    }

    [Fact]
    public void WhenFineEditingPositiveExample_ThenOutputMovesTowardInputConcept()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 24, seed: 7);
        memory.ObserveContradiction("input-x", "output-y", 0.2);

        var before = memory.ExportSnapshot();
        var inBefore = before.Nodes.Single(n => n.Name == "input-x");
        var outBefore = before.Nodes.Single(n => n.Name == "output-y");
        var beforeDistance = Distance(inBefore.PositiveFace, outBefore.PositiveFace);

        memory.FineEditFromExample(["input-x"], ["output-y"], isNegativeExample: false);

        var after = memory.ExportSnapshot();
        var inAfter = after.Nodes.Single(n => n.Name == "input-x");
        var outAfter = after.Nodes.Single(n => n.Name == "output-y");
        var afterDistance = Distance(inAfter.PositiveFace, outAfter.PositiveFace);

        Assert.True(afterDistance < beforeDistance);
    }

    [Fact]
    public void WhenCreatingAddConcept_ThenLogFaceStaysNearZero()
    {
        var memory = new PlatonicSpaceMemory(faceDimension: 32, seed: 1);
        memory.ObserveContradiction("add", "1", 0.2);

        var snapshot = memory.ExportSnapshot();
        var addNode = snapshot.Nodes.Single(n => n.Name == "add");
        var numericDims = Math.Min(addNode.PositiveFace.Length / 2, 21);
        var logStart = numericDims;

        var polyEnergy = addNode.PositiveFace.Take(numericDims).Sum(Math.Abs);
        var logEnergy = addNode.PositiveFace.Skip(logStart).Take(numericDims).Sum(Math.Abs);

        Assert.True(polyEnergy > 0.0);
        Assert.True(logEnergy < 1e-9);
    }

    [Fact]
    public void WhenObservingAdditionExample_ThenAddPrefersPolynomialFace()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: 24, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: 32, seed: 9);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        trainer.ObserveInferenceResult("4+5", "9");

        var poly = memory.GetContradiction("add", "face:poly");
        var log = memory.GetContradiction("add", "face:log");
        Assert.True(poly < log);
    }

    private static double Distance(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var sum = 0.0;
        for (var i = 0; i < Math.Min(a.Count, b.Count); i++)
        {
            var d = a[i] - b[i];
            sum += d * d;
        }

        return Math.Sqrt(sum);
    }
}
