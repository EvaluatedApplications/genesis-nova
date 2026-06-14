using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Train;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Fast unit tests for the autonomous planner's curriculum STRATEGY (no training; pure plan logic).
/// Default = FOCUSED CURRICULUM: train one creator to convergence at a time, complexity-ordered
/// (corenova primitives first), replaying mastered creators — focused converges where composite
/// oscillates. Legacy composite / bootstrap-first gate remain available behind flags.
/// </summary>
public sealed class FocusedCurriculumPlannerTests
{
    private static GenesisAutonomousTrainingPlanner Planner(params IExampleCreator[] creators) =>
        new(creators.ToList());

    private static string[] PlannedCreators(GenesisAutonomousCompositePlan plan) =>
        plan.CreatorPlans.Select(p => p.CreatorName).ToArray();

    private static GenesisAutonomousCompositePlan Plan(
        GenesisAutonomousTrainingPlanner planner, GenesisAutonomousTrainingRequest request) =>
        planner.SuggestComposite(request, history: Array.Empty<GenesisAutonomousTrainingRound>(), roundIndex: 0);

    [Fact]
    public void Focused_TrainsOnlyTheFirstUnmasteredCreator_InComplexityOrder()
    {
        // corenova:number-word-equiv (8) < corenova:retrieval-category (10) < arithmetic:add (20).
        // With nothing mastered, the focus is the single lowest-complexity creator; the rest wait.
        var planner = Planner(
            new ArithmeticCreator("add"), new NumberWordCreator(), new CategoryRetrievalCreator());

        var names = PlannedCreators(Plan(planner, new GenesisAutonomousTrainingRequest()));

        Assert.Equal(new[] { "corenova:number-word-equiv" }, names);
    }

    [Fact]
    public void Focused_SingleCreator_IsNoOp()
    {
        var planner = Planner(new NumberWordCreator());
        Assert.Contains("corenova:number-word-equiv",
            PlannedCreators(Plan(planner, new GenesisAutonomousTrainingRequest())));
    }

    [Fact]
    public void Composite_Fallback_TrainsEverything_WhenFocusedDisabled()
    {
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        var request = new GenesisAutonomousTrainingRequest(FocusedCurriculum: false, BootstrapFirst: false);

        var names = PlannedCreators(Plan(planner, request));

        Assert.Contains("corenova:number-word-equiv", names);
        Assert.Contains("arithmetic:add", names);
    }

    [Fact]
    public void LegacyBootstrapGate_TrainsOnlyCorenova_WhenFocusedDisabled()
    {
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        // Focused off, bootstrap-first on → legacy gate: corenova-only until mastered.
        var request = new GenesisAutonomousTrainingRequest(FocusedCurriculum: false, BootstrapFirst: true);

        var names = PlannedCreators(Plan(planner, request));

        Assert.Contains("corenova:number-word-equiv", names);
        Assert.DoesNotContain("arithmetic:add", names);
    }

    // A prompt-answer round: low loss (passes the loss gate) but a chosen platonic success rate.
    // Difficulty defaults to the request's default MasteryDifficulty so a high-success history can
    // actually reach the drive-to-depth mastery gate (mastery requires maxDifficulty >= MasteryDifficulty).
    private static GenesisAutonomousTrainingRound Round(string creator, double success, int difficulty = 3) =>
        new(Round: 1, CreatorName: creator, SampleCount: 8, Difficulty: difficulty, Epochs: 1,
            Report: new GenesisTrainingReport(
                Epochs: 1, ExampleCount: 8,
                AverageLoss: new GenesisStepLoss(0.2, 0, 0, 0, 0, 0.2),
                ContradictionRate: 0, ConservationDrift: 0, MemoryOverwriteRate: 0,
                IntrospectionCycles: 0, PendingQueueDepth: 0, ExampleSuccessRate: success),
            CreatorProgress: new GenesisCreatorProgress(
                creator, GenesisTrainingExampleKind.PromptAnswer,
                SeenCount: 8, SuccessCount: (int)(8 * success), SuccessRate: success,
                LastTokenLoss: 0.2, AverageTokenLoss: 0.2, BestTokenLoss: 0.2));

    [Fact]
    public void NeuralAnsweringCreator_DoesNotMaster_AndStaysTheFocus()
    {
        // PUNISH NEURAL: low loss alone used to mark a creator mastered. Now a prompt-answer creator
        // answering neurally (low platonic success) is NOT mastered, so the focus does NOT advance.
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        var history = new[]
        {
            Round("corenova:number-word-equiv", success: 0.30),
            Round("corenova:number-word-equiv", success: 0.30),
            Round("corenova:number-word-equiv", success: 0.30),
        };

        var names = PlannedCreators(planner.SuggestComposite(
            new GenesisAutonomousTrainingRequest(), history, roundIndex: 3));

        Assert.Equal(new[] { "corenova:number-word-equiv" }, names); // unmastered → still the focus
    }

    [Fact]
    public void PlatonicSuccessCreator_Masters_AndFocusAdvances()
    {
        // Same low loss, but HIGH platonic success → mastered → focus advances; the mastered creator
        // rides along as replay.
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        var history = new[]
        {
            Round("corenova:number-word-equiv", success: 0.95),
            Round("corenova:number-word-equiv", success: 0.95),
            Round("corenova:number-word-equiv", success: 0.95),
        };

        var names = PlannedCreators(planner.SuggestComposite(
            new GenesisAutonomousTrainingRequest(), history, roundIndex: 3));

        Assert.Contains("arithmetic:add", names);                  // focus advanced to the next creator
        Assert.Contains("corenova:number-word-equiv", names);      // mastered → replayed for retention
    }

    [Fact]
    public void HighSuccessButShallowDifficulty_DoesNotMaster_FocusStaysToClimbDepth()
    {
        // DRIVE-TO-DEPTH: high platonic success but only at difficulty 1 (below the default
        // MasteryDifficulty of 3) must NOT count as mastered — the focus stays on the creator so it
        // keeps climbing its difficulties under focus, instead of advancing and leaving the harder
        // difficulties to the oscillation-prone composite maintenance phase.
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        var history = new[]
        {
            Round("corenova:number-word-equiv", success: 0.95, difficulty: 1),
            Round("corenova:number-word-equiv", success: 0.95, difficulty: 1),
            Round("corenova:number-word-equiv", success: 0.95, difficulty: 1),
        };

        var names = PlannedCreators(planner.SuggestComposite(
            new GenesisAutonomousTrainingRequest(), history, roundIndex: 3));

        Assert.Equal(new[] { "corenova:number-word-equiv" }, names); // shallow → not mastered → still focus
    }

    [Fact]
    public void ExhaustedUnmasteredCreator_RidesAlongAsReplay_NotForgotten()
    {
        // FORGETTING FIX: a creator that burns its entire FocusBudget without mastering must not be
        // dropped from the curriculum — the focus advances past it, but it keeps being replayed so its
        // partial competence is retained (and can keep climbing).
        var planner = Planner(new NumberWordCreator(), new ArithmeticCreator("add"));
        var request = new GenesisAutonomousTrainingRequest();
        var history = Enumerable
            .Range(0, request.FocusBudget) // exactly FocusBudget attempts, none mastering (low success)
            .Select(_ => Round("corenova:number-word-equiv", success: 0.30, difficulty: 1))
            .ToArray();

        var names = PlannedCreators(planner.SuggestComposite(request, history, roundIndex: history.Length));

        Assert.Contains("arithmetic:add", names);             // focus advanced (number-word exhausted)
        Assert.Contains("corenova:number-word-equiv", names); // exhausted but RETAINED as replay
    }
}
