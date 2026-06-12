using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// The number-word↔digit equivalence (and the cleanliness of the relation graph it depends on) must be
/// LEARNED RELATIONALLY, not via a hardcoded table and not corrupted by concept-planner noise. Two
/// first-principles fixes are verified here:
///   1. Malformed tokens ("0+", "5-", "1+1") must NOT be treated as numbers (strict numeric parsing) —
///      otherwise they become frozen "value-0" garbage concepts that pollute numeric relations.
///   2. The academic coupling must relate the GENUINE input→output tokens, so a number-word's strongest
///      numeric relation is its own digit ("one"→"1"), clearly ahead of any planner-hallucinated noise.
/// </summary>
public sealed class RelationalEquivalenceTests
{
    private readonly ITestOutputHelper _out;
    public RelationalEquivalenceTests(ITestOutputHelper output) => _out = output;

    // Strict: a genuine number is leading-sign + digits + optional decimal point. Nothing else.
    private static bool IsPlainNumber(string s) =>
        double.TryParse(s,
            NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint
            | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite,
            CultureInfo.InvariantCulture, out _);

    // ── Task 1: malformed tokens must not masquerade as numbers. On a FRESH space, TryGetConceptFace
    // returns a (homomorphic) face only for genuine numbers; a malformed token has no node and no
    // numeric face, so it returns false. ──
    [Theory]
    [InlineData("0", true)]
    [InlineData("1", true)]
    [InlineData("-3", true)]
    [InlineData("3.5", true)]
    [InlineData("0+", false)]   // trailing sign — was wrongly parsed as 0 under NumberStyles.Any
    [InlineData("5-", false)]
    [InlineData("1+1", false)]  // glued expression — not a number
    [InlineData("one", false)]
    public void MalformedTokens_AreNotTreatedAsNumbers(string token, bool expectedNumeric)
    {
        var m = new PlatonicSpaceMemory(faceDimension: 64, seed: 1);
        var isNumeric = m.TryGetConceptFace(token, out _); // fresh space: true ⟺ parses as a number
        Assert.Equal(expectedNumeric, isNumeric);
    }

    // ── Task 2: a number-word's STRONGEST numeric relation is its own digit, learned from the genuine
    // input→output pairing — not planner noise. An untrained word has no numeric relation at all. ──
    [Fact]
    public void NumberWord_StrongestNumericRelation_IsItsOwnDigit_NotNoise()
    {
        var config = new GenesisNovaConfig(HiddenSize: 64, LearningRate: 0.05);
        var tokenizer = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var memory = new PlatonicSpaceMemory(faceDimension: config.HiddenSize / 2, seed: 7);
        var trainer = new GenesisTrainer(tokenizer, model, memory, config);

        var words = new[] { "zero", "one", "two", "three", "four", "five" };
        var curriculum = new List<GenesisExample>();
        for (var n = 0; n <= 5; n++)
        {
            curriculum.Add(new GenesisExample(words[n], $"{n}"));
            curriculum.Add(new GenesisExample($"{n}", words[n]));
        }
        for (var a = 0; a <= 5; a++)
            for (var b = 0; b <= 5; b++)
                curriculum.Add(new GenesisExample($"{a} + {b}", $"{a + b}"));

        var rng = new Random(7);
        for (var e = 0; e < 25; e++)
        {
            for (var i = curriculum.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (curriculum[i], curriculum[j]) = (curriculum[j], curriculum[i]);
            }
            foreach (var ex in curriculum)
                trainer.TrainStep(ex);
        }

        var numericRel = memory
            .GetNeighbors("one", PlatonicNeighborhoodType.Relational, 16, 0.0)
            .Where(n => IsPlainNumber(n.Concept))
            .OrderByDescending(n => n.ObservationCount)
            .ThenByDescending(n => n.Confidence)
            .ToList();
        _out.WriteLine("one numeric REL : " + string.Join(", ",
            numericRel.Select(n => $"{n.Concept}@obs{n.ObservationCount}")));

        Assert.NotEmpty(numericRel);
        Assert.Equal("1", numericRel[0].Concept); // strongest numeric relation of "one" is its digit

        // Clear margin over any other numeric relation — the genuine pair DOMINATES the noise.
        if (numericRel.Count > 1)
            Assert.True(numericRel[0].ObservationCount >= numericRel[1].ObservationCount * 2,
                $"'one'→'1' should dominate: {string.Join(", ", numericRel.Select(n => $"{n.Concept}@{n.ObservationCount}"))}");

        // Not hardcoded: an untrained word has NO numeric relation to resolve through.
        var zarpNumeric = memory
            .GetNeighbors("zarp", PlatonicNeighborhoodType.Relational, 16, 0.0)
            .Where(n => IsPlainNumber(n.Concept))
            .ToList();
        Assert.Empty(zarpNumeric);
    }
}
