using GenesisNova.Data.Creators;

namespace GenesisNova.Tests;

public sealed class PublicTextCorpusCreatorTests
{
    [Fact]
    public void WhenGeneratingFromFallbackCorpus_ThenCreatorProducesNextTokenExamples()
    {
        var sut = new PublicTextCorpusCreator(
            name: "public:test-corpus",
            estimatedComplexity: 10,
            datasetName: "local/test",
            config: "default",
            split: "train",
            textField: "text",
            fallbackSnippets:
            [
                "The quick brown fox jumps over the lazy dog in a deterministic training sentence used for testing.",
                "Machine learning systems improve when data quality, evaluation coverage, and reproducibility are handled deliberately."
            ],
            allowRemoteFetch: false);

        var examples = sut.Generate(count: 16, difficulty: 1, forTraining: true);

        Assert.Equal(16, examples.Length);
        Assert.All(examples, ex =>
        {
            Assert.True(ex.Input.Length > 0);
            Assert.True(ex.Output.Length > 0);
            Assert.DoesNotContain(' ', ex.Output);
        });
    }

    [Fact]
    public void WhenInputsAreSame_ThenGenerationIsDeterministic()
    {
        var sut = new PublicTextCorpusCreator(
            name: "public:test-corpus",
            estimatedComplexity: 10,
            datasetName: "local/test",
            config: "default",
            split: "train",
            textField: "text",
            fallbackSnippets:
            [
                "Deterministic creators should emit the same sequence when name and difficulty are unchanged across calls.",
                "Consistency makes autonomous planning and loss comparisons stable between rounds."
            ],
            allowRemoteFetch: false);

        var first = sut.Generate(count: 20, difficulty: 2, forTraining: true);
        var second = sut.Generate(count: 20, difficulty: 2, forTraining: true);

        Assert.Equal(first, second);
    }
}
