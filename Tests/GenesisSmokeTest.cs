using System;
using System.IO;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Tests;

/// <summary>
/// Clean-slate smoke test. Verifies the core training interface wires together and
/// runs end-to-end without throwing: tokenizer -> neural model + platonic space -> trainer.
/// Deliberately the ONLY test in the suite; richer coverage is added back deliberately.
/// </summary>
public sealed class GenesisSmokeTest
{
    [Fact]
    public void Tokenizer_RoundTrips()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();

        var tokens = tokenizer.Encode("Hello   WORLD", addBos: true, addEos: true);
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("hello world", decoded);
        Assert.True(tokenizer.VocabularySize >= 5);
    }

    [Fact]
    public void TrainingInterface_RunsSingleStep_WithoutThrowing()
    {
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05));
        var memory = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        var loss = trainer.TrainStep(new GenesisExample("say one", "one"));

        Assert.True(loss.TokenLoss >= 0.0);
        Assert.True(loss.TotalLoss >= 0.0);
        Assert.False(double.IsNaN(loss.TotalLoss));
    }

    /// <summary>
    /// A trained model must survive a full on-disk save -> load round-trip with NO loss: EVERY
    /// parameter group (embeddings, output head, route head, edit head, and the shared GRU — which
    /// every head reads via hInput) must come back identical, and the loaded model must route
    /// identically. This is the guard against the checkpoint silently dropping the GRU/edit head,
    /// which made loaded models behave like untrained ones (REPL failing despite trained correctness).
    /// </summary>
    [Fact]
    public void Checkpoint_SaveLoad_RoundTrips_WholeModel_NoLoss()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory);

        // Train so the GRU + every head are initialized and non-trivial (else there's nothing to lose).
        trainer.TrainStep(new GenesisExample("say one", "one"));
        trainer.TrainStep(new GenesisExample("count two three", "five"));

        var probe = tokenizer.Encode("say one");
        var routeBefore = model.PredictRoute(probe);
        var before = model.Export();
        Assert.NotNull(before.GruWih);      // GRU actually initialized
        Assert.NotNull(before.EditWeights); // edit head actually initialized
        Assert.NotNull(before.RouteWeights);

        var tmp = Path.Combine(Path.GetTempPath(), $"genesis_ckpt_{Guid.NewGuid():N}.json");
        try
        {
            GenesisCheckpointStore.Save(tmp, config, tokenizer, model);
            var loaded = GenesisCheckpointStore.Load(tmp);
            var after = loaded.Model.Export();

            // Every parameter group survives the disk round-trip (float32 precision).
            AssertMatrixEqual(before.Embeddings, after.Embeddings);
            AssertMatrixEqual(before.OutputWeights, after.OutputWeights);
            AssertVectorEqual(before.OutputBias, after.OutputBias);
            AssertMatrixEqual(before.RouteWeights, after.RouteWeights);
            AssertVectorEqual(before.RouteBias, after.RouteBias);
            AssertMatrixEqual(before.EditWeights, after.EditWeights);
            AssertVectorEqual(before.EditBias, after.EditBias);
            AssertMatrixEqual(before.GruWih, after.GruWih);
            AssertMatrixEqual(before.GruWhh, after.GruWhh);
            AssertVectorEqual(before.GruBih, after.GruBih);
            AssertVectorEqual(before.GruBhh, after.GruBhh);

            // End-to-end: the loaded model routes identically (it IS the same trained model).
            var routeAfter = loaded.Model.PredictRoute(probe);
            Assert.Equal(routeBefore.RouteId, routeAfter.RouteId);
            Assert.Equal(routeBefore.Confidence, routeAfter.Confidence, 5);
        }
        finally
        {
            if (File.Exists(tmp))
                File.Delete(tmp);
        }
    }

    private static void AssertMatrixEqual(double[,]? a, double[,]? b)
    {
        Assert.True(a is not null && b is not null, "matrix parameter missing after round-trip");
        Assert.Equal(a!.GetLength(0), b!.GetLength(0));
        Assert.Equal(a.GetLength(1), b.GetLength(1));
        for (var i = 0; i < a.GetLength(0); i++)
            for (var j = 0; j < a.GetLength(1); j++)
                Assert.Equal(a[i, j], b[i, j], 6);
    }

    private static void AssertVectorEqual(double[]? a, double[]? b)
    {
        Assert.True(a is not null && b is not null, "vector parameter missing after round-trip");
        Assert.Equal(a!.Length, b!.Length);
        for (var i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i], 6);
    }
}
