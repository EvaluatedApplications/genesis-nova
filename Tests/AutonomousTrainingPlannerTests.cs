using System.Collections.Immutable;
using GenesisNova.Data;
using GenesisNova.Train;

namespace GenesisNova.Tests;

public sealed class AutonomousTrainingPlannerTests
{
    [Fact]
    public void WhenPlanningInitialRound_ThenUsesLowestComplexityCreatorAndLowSettings()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("hard", 3),
            new StubCreator("easy", 1),
            new StubCreator("medium", 2),
        });

        var request = new GenesisAutonomousTrainingRequest(
            MaxRounds: 8,
            InitialSampleCount: 24,
            InitialDifficulty: 0,
            InitialEpochs: 1);

        var plan = planner.Suggest(request, history: []);

        Assert.Equal("easy", plan.CreatorName);
        Assert.Equal(24, plan.SampleCount);
        Assert.Equal(1, plan.TrainCount);
        Assert.Equal(0, plan.Difficulty);
        Assert.Equal(1, plan.Epochs);
        Assert.Contains("start tiny", plan.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WhenPreviousRoundWasStable_ThenPlannerInterleavesToAnotherCreator()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("easy", 1),
            new StubCreator("medium", 2),
            new StubCreator("hard", 3),
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(
                Round: 1,
                CreatorName: "easy",
                SampleCount: 24,
                Difficulty: 0,
                Epochs: 1,
                Report: CreateReport(loss: 0.08, noise: 0.05))
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 8,
                InitialSampleCount: 24,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 0.10),
            history);

        Assert.Equal("medium", plan.CreatorName);
        Assert.Equal(24, plan.SampleCount);
        Assert.Equal(1, plan.TrainCount);
        Assert.Equal(0, plan.Difficulty);
        Assert.Equal(1, plan.Epochs);
    }

    [Fact]
    public void WhenCreatorIsStronglyLearned_ThenPlannerSkipsItAndTargetsWeakSkill()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("easy", 1),
            new StubCreator("medium", 2),
            new StubCreator("hard", 3),
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(1, "easy", 24, 1, 1, CreateReport(loss: 0.05, noise: 0.03)),
            new GenesisAutonomousTrainingRound(2, "easy", 24, 1, 1, CreateReport(loss: 0.04, noise: 0.03)),
            new GenesisAutonomousTrainingRound(3, "easy", 24, 1, 1, CreateReport(loss: 0.05, noise: 0.02)),
            new GenesisAutonomousTrainingRound(4, "hard", 24, 0, 1, CreateReport(loss: 0.24, noise: 0.10)),
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 12,
                InitialSampleCount: 24,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 0.10),
            history);

        Assert.NotEqual("easy", plan.CreatorName);
        Assert.True(
            plan.CreatorName is "hard" or "medium",
            $"Expected weak or under-trained creator, got {plan.CreatorName}");
    }

    [Fact]
    public void WhenSameWeakCreatorRepeats_ThenPlannerForcesInterleavingToAnotherCreator()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("language:words", 1),
            new StubCreator("language:greet", 2),
            new StubCreator("arithmetic:add", 3),
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(1, "language:words", 24, 0, 1, CreateReport(loss: 1.40, noise: 0.05)),
            new GenesisAutonomousTrainingRound(2, "language:words", 24, 0, 1, CreateReport(loss: 1.70, noise: 0.05)),
            new GenesisAutonomousTrainingRound(3, "language:words", 24, 0, 1, CreateReport(loss: 1.85, noise: 0.05)),
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 12,
                InitialSampleCount: 24,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 0.10),
            history);

        Assert.NotEqual("language:words", plan.CreatorName);
    }

    [Fact]
    public void WhenLossIsBelowThreshold_ThenDifficultyIncreases()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("numeric:compare", 1)
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(
                Round: 1,
                CreatorName: "numeric:compare",
                SampleCount: 12,
                Difficulty: 0,
                Epochs: 1,
                Report: CreateReport(loss: 0.57, noise: 0.01))
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 8,
                InitialSampleCount: 12,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 1.2),
            history);

        Assert.Equal(1, plan.Difficulty);
    }

    [Fact]
    public void WhenLossIsBelowThreshold_ThenHorizonGrowsOrganically()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("public:test", 1)
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(
                Round: 1,
                CreatorName: "public:test",
                SampleCount: 1,
                Difficulty: 0,
                Epochs: 1,
                Report: CreateReport(loss: 0.05, noise: 0.01))
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 8,
                InitialSampleCount: 1,
                InitialTrainCount: 1,
                MinSampleCount: 1,
                MinTrainCount: 1,
                MaxSampleCount: 16,
                MaxTrainCount: 16,
                LossThreshold: 0.10),
            history);

        Assert.Equal(2, plan.SampleCount);
        Assert.Equal(2, plan.TrainCount);
        Assert.Equal(1, plan.Difficulty);
    }

    [Fact]
    public void WhenRoundBudgetExceedsSampleCaps_ThenCompositePlanningClampsWithoutStalling()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("public:a", 1),
            new StubCreator("public:b", 2),
            new StubCreator("public:c", 3),
            new StubCreator("public:d", 4)
        });

        var plan = planner.SuggestComposite(
            new GenesisAutonomousTrainingRequest(
                InitialSampleCount: 1,
                InitialTrainCount: 1,
                MinSampleCount: 1,
                MaxSampleCount: 1,
                MinTrainCount: 1,
                MaxTrainCount: 24,
                RoundTrainBudget: 16),
            history: [],
            roundIndex: 0);

        Assert.Equal(4, plan.CreatorPlans.Count);
        Assert.All(plan.CreatorPlans, creatorPlan => Assert.Equal(1, creatorPlan.TrainCount));
        Assert.Equal(4, plan.CreatorPlans.Sum(p => p.TrainCount));
    }

    [Fact]
    public void WhenLossIsLowButNoiseIsHigh_ThenDifficultyStillIncreases()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("numeric:compare", 1)
        });

        var history = new[]
        {
            new GenesisAutonomousTrainingRound(
                Round: 1,
                CreatorName: "numeric:compare",
                SampleCount: 12,
                Difficulty: 0,
                Epochs: 1,
                Report: CreateReport(loss: 0.57, noise: 0.80))
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 8,
                InitialSampleCount: 12,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 1.2),
            history);

        Assert.Equal(1, plan.Difficulty);
    }

    [Fact]
    public void WhenHistoryContainsPriorRunsWithSameRoundNumbers_ThenPlannerUsesMostRecentCreatorRound()
    {
        var planner = new GenesisAutonomousTrainingPlanner(new IExampleCreator[]
        {
            new StubCreator("numeric:compare", 1)
        });

        var history = new[]
        {
            // Older run snapshot (high round but stale).
            new GenesisAutonomousTrainingRound(200, "numeric:compare", 12, 0, 1, CreateReport(loss: 1.8, noise: 0.01)),
            // Newer run entries appended later with reset round numbering.
            new GenesisAutonomousTrainingRound(1, "numeric:compare", 12, 0, 1, CreateReport(loss: 0.4, noise: 0.01)),
            new GenesisAutonomousTrainingRound(2, "numeric:compare", 12, 1, 1, CreateReport(loss: 0.3, noise: 0.01))
        };

        var plan = planner.Suggest(
            new GenesisAutonomousTrainingRequest(
                MaxRounds: 8,
                InitialSampleCount: 12,
                InitialDifficulty: 0,
                InitialEpochs: 1,
                LossThreshold: 1.2),
            history);

        Assert.Equal(2, plan.Difficulty);
    }

    private static GenesisTrainingReport CreateReport(double loss, double noise)
        => new(
            Epochs: 1,
            ExampleCount: 24,
            AverageLoss: new GenesisStepLoss(loss, 0.0, 0.0, 0.0, 0.0, loss),
            ContradictionRate: 0.05,
            ConservationDrift: 0.0,
            MemoryOverwriteRate: 0.0,
            IntrospectionCycles: 0,
            PendingQueueDepth: 0,
            SpaceManagementCycles: 0,
            NodesPruned: 0,
            RelationsPruned: 0,
            FinalNodeCount: 0,
            FinalRelationCount: 0,
            SpaceNoiseRatio: noise);

    private sealed class StubCreator : IExampleCreator
    {
        public StubCreator(string name, int complexity)
        {
            Name = name;
            EstimatedComplexity = complexity;
        }

        public string Name { get; }
        public int EstimatedComplexity { get; }
        public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.WindowedText;

        public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
            => ImmutableArray.CreateRange(Enumerable.Range(0, Math.Max(0, count)).Select(i => ($"{Name}:{difficulty}:{i}", $"{Name}:{difficulty}:{i}")));
    }
}
