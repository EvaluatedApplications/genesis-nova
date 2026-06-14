using System;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Fast unit tests for the REAL query-supervision code: <see cref="GenesisTrainer.ResolveQueryLabel"/>
/// derives (face operation, operand token mask) from an example's OWN numeric structure — no surface
/// grammar. No training involved; instant.
/// </summary>
public sealed class QueryLabelTests
{
    private readonly WhitespaceGenesisTokenizer _tokenizer = new();
    private readonly GenesisTrainer _trainer;

    public QueryLabelTests()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        _trainer = new GenesisTrainer(_tokenizer, model, memory, config);
    }

    private GenesisQueryLabel? Label(string input, string output)
        => _trainer.ResolveQueryLabel(_tokenizer.Encode(input), output);

    [Theory]
    [InlineData("1 + 1", "2", 1)]               // add
    [InlineData("what is 2 plus 3", "5", 1)]    // framed add — framing must not matter
    [InlineData("8 - 5", "3", 2)]               // sub
    [InlineData("what is 8 minus 5", "3", 2)]   // framed sub
    [InlineData("3 * 4", "12", 3)]              // mul
    [InlineData("9 / 3", "3", 4)]               // div
    [InlineData("the sum of 2 and 5", "7", 1)]  // fully worded framing
    public void DerivesTheFaceOperation_FromNumericStructureOnly(string input, string output, int expectedOp)
    {
        var label = Label(input, output);
        Assert.NotNull(label);
        Assert.Equal(expectedOp, label!.Value.OperationId);
    }

    [Fact]
    public void OperandMask_MarksDigitsAndOnlyDigits()
    {
        var ids = _tokenizer.Encode("what is 2 plus 13");
        var label = _trainer.ResolveQueryLabel(ids, "15");
        Assert.NotNull(label);

        var mask = label!.Value.OperandMask;
        var tokens = ids.Select(id => _tokenizer.Vocabulary[id]).ToArray();
        Assert.Equal(tokens.Length, mask.Length);
        for (var i = 0; i < tokens.Length; i++)
        {
            var isDigit = tokens[i].Length == 1 && tokens[i][0] is >= '0' and <= '9';
            Assert.Equal(isDigit, mask[i]); // digits are operands; framing tokens are negatives
        }
    }

    [Fact]
    public void NegativeSecondOperand_StaysOnTheSurfaceOperator()
    {
        // "5 + -3" = 2 must label ADD with signed operand −3 — NOT relabel as sub. Without the
        // signed-run rule, ~half the generator's add examples supervised the op head as sub and it
        // learned surface noise (empirically: sub-biased 0.49–0.59 wrong → add 0.77–0.87 right).
        var ids = _tokenizer.Encode("5 + -3");
        var label = _trainer.ResolveQueryLabel(ids, "2");
        Assert.NotNull(label);
        Assert.Equal(1, label!.Value.OperationId);

        // The unary '-' is part of the operand and must be in the mask.
        var tokens = ids.Select(id => _tokenizer.Vocabulary[id]).ToArray();
        var minusIndex = Array.LastIndexOf(tokens, "-");
        Assert.True(label.Value.OperandMask[minusIndex]);
    }

    [Fact]
    public void BinaryMinus_IsTheOperator_NotASign()
    {
        var label = Label("8 - 5", "3");
        Assert.NotNull(label);
        Assert.Equal(2, label!.Value.OperationId); // sub(8,5), not add(8,−5)
    }

    [Theory]
    [InlineData("2 + 2", "4")]    // ambiguous: 2+2 == 2*2 — must NOT guess
    [InlineData("1 + 1", "5")]    // no face op fits
    [InlineData("hello", "hi")]   // no numeric structure
    [InlineData("1 + 2 + 3", "6")] // three operands — outside the two-operand face op
    public void AmbiguousOrNonConforming_IsUnsupervised(string input, string output)
        => Assert.Null(Label(input, output));

    [Fact]
    public void UntrainedQueryHeads_Abstain()
    {
        // PredictQuery must return op 0 (abstain) before any supervision has initialized the heads,
        // so the inference engine's GRU-query path degrades to the other tools gracefully.
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var model = new GenesisNeuralModel(config);
        var (opId, confidence, flags) = model.PredictQuery(_tokenizer.Encode("1 + 1"));
        Assert.Equal(0, opId);
        Assert.Equal(0.0, confidence);
        Assert.Empty(flags);
    }

    [Fact]
    public void QueryHeads_AreTornDownOnImport_AndTrainingRecovers()
    {
        // The autonomous-runtime restart path: stop → checkpoint → start → Import(snapshot). The
        // query heads are NOT persisted, so Import MUST dispose+null them — leaving the previous
        // session's stale parameters registered in the fresh optimizer corrupted every subsequent
        // training run (the real-world "worked, restarted, now all runs fail" regression).
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: ProductionDims.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);

        // A few supervised steps initialize the query heads.
        for (var i = 0; i < 4; i++)
            trainer.TrainStep(new GenesisExample("1 + 2", "3"));
        Assert.NotEqual(0, model.PredictQuery(tokenizer.Encode("1 + 2")).OperationId);

        // Simulate the restart: export + import (what auto-resume does).
        var snapshot = model.Export();
        model.Import(snapshot);

        // Heads must be GONE (clean abstain), not stale.
        var (opId, _, flags) = model.PredictQuery(tokenizer.Encode("1 + 2"));
        Assert.Equal(0, opId);
        Assert.Empty(flags);

        // And training must RECOVER: supervised steps after Import reinitialize fresh heads and run
        // without graph/optimizer corruption. (NB not "2 + 2" — that is the AMBIGUOUS case
        // (2+2 == 2*2) which ResolveQueryLabel correctly refuses to supervise.)
        for (var i = 0; i < 4; i++)
            trainer.TrainStep(new GenesisExample("3 + 2", "5"));
        Assert.NotEqual(0, model.PredictQuery(tokenizer.Encode("3 + 2")).OperationId);
    }
}
