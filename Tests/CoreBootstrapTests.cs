using System;
using System.Linq;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

// Fast tests of the REAL bootstrap code (creators + suite wiring) — no in-test training. The
// training-based findings that shaped these lessons (equivalence carrier 10/10, retrieval 16/16,
// held-out computation 10/10, bare-beats-diverse) were one-off experiments; their conclusions live
// in the lesson docs. The single retained end-to-end training demonstration is
// GruQueryConstructionTests. Each fixture is its own class so xUnit runs them in parallel.

public sealed class NumberWordCreatorTests
{
    [Fact]
    public void GeneratesBidirectionalBarePairs_Deterministically()
    {
        var creator = new NumberWordCreator();
        var a = creator.Generate(20, difficulty: 0, forTraining: true);
        var b = creator.Generate(20, difficulty: 0, forTraining: true);

        Assert.Equal(20, a.Length);
        Assert.Equal(a, b); // deterministic per (creator, difficulty) — IExampleCreator rule 3

        // Bidirectional: word→digit AND digit→word both present, bare (single token each side).
        Assert.Contains(a, p => p.Input == "one" && p.Output == "1");
        Assert.Contains(a, p => p.Input == "1" && p.Output == "one");
        Assert.All(a, p =>
        {
            Assert.DoesNotContain(' ', p.Input);
            Assert.DoesNotContain(' ', p.Output);
        });
    }

    [Fact]
    public void DifficultyWidensTheRange()
    {
        var creator = new NumberWordCreator();
        var d0 = creator.Generate(200, 0, true);
        var d2 = creator.Generate(200, 2, true);

        // d0 covers exactly 0–9; the tens only appear at higher difficulty.
        Assert.DoesNotContain(d0, p => p.Input == "twenty" || p.Output == "twenty");
        Assert.Contains(d2, p => p.Input == "twenty" && p.Output == "20");
    }
}

public sealed class CategoryRetrievalCreatorTests
{
    [Fact]
    public void GeneratesBareItemToCategoryPairs()
    {
        var creator = new CategoryRetrievalCreator();
        var examples = creator.Generate(16, difficulty: 0, forTraining: true);

        Assert.Equal(16, examples.Length);
        Assert.Contains(examples, p => p.Input == "apple" && p.Output == "fruit");
        // BARE is the load-bearing property: filler prompts ("apple is a") empirically destroyed the
        // relation carrier (0/16) while bare pairs formed it perfectly (16/16).
        Assert.All(examples, p =>
        {
            Assert.DoesNotContain(' ', p.Input);
            Assert.DoesNotContain(' ', p.Output);
        });
    }

    [Fact]
    public void IsDeterministicAndCyclesItsTable()
    {
        var creator = new CategoryRetrievalCreator();
        var a = creator.Generate(40, 0, true);
        var b = creator.Generate(40, 0, true);
        Assert.Equal(a, b);
        // count > unique table slice → sampled with replacement (IExampleCreator rule 2).
        Assert.Equal(a[0], a[16]);
    }
}

public sealed class CoreBootstrapSuiteTests
{
    [Fact]
    public void LessonsAreProvenOrderedAndGenerable()
    {
        var lessons = CoreBootstrapSuite.Lessons;
        Assert.NotEmpty(lessons);

        // Ordered simplest → compositional (non-decreasing complexity).
        var complexities = lessons.Select(l => l.Creator.EstimatedComplexity).ToArray();
        Assert.Equal(complexities.OrderBy(c => c), complexities);

        // Every lesson actually generates learnable examples at its declared difficulty (rule 1),
        // and documents what it demonstrates.
        foreach (var lesson in lessons)
        {
            var examples = lesson.Creator.Generate(8, lesson.Difficulty, forTraining: true);
            Assert.NotEmpty(examples);
            Assert.All(examples, p =>
            {
                Assert.False(string.IsNullOrWhiteSpace(p.Input));
                Assert.False(string.IsNullOrWhiteSpace(p.Output));
            });
            Assert.False(string.IsNullOrWhiteSpace(lesson.Demonstrates));
        }

        // The foundational lessons the demonstrations proved are present.
        Assert.Contains(lessons, l => l.Creator is NumberWordCreator);
        Assert.Contains(lessons, l => l.Creator is CategoryRetrievalCreator);
    }

    [Fact]
    public void BootstrapLessons_AreWiredIntoTheRegistry()
    {
        // The UI, autonomous planner, and orchestrator all consume ExampleCreatorRegistry.All —
        // every bootstrap lesson's creator must be registered there (by Name) or it is invisible to
        // real training runs. The bootstrap creators' low EstimatedComplexity also makes
        // complexity-ordered consumers train them FIRST (the validated bootstrap-first ordering).
        var registryNames = ExampleCreatorRegistry.All.Select(c => c.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var lesson in CoreBootstrapSuite.Lessons)
            Assert.Contains(lesson.Creator.Name, registryNames);

        var minRegistryComplexity = ExampleCreatorRegistry.All.Min(c => c.EstimatedComplexity);
        Assert.Equal(minRegistryComplexity, CoreBootstrapSuite.Lessons.Min(l => l.Creator.EstimatedComplexity));
    }
}
