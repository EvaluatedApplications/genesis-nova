using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class RunAutonomousTrainingRoundStep
{
    private readonly GenesisTrainingOrchestrator _orchestrator;

    public RunAutonomousTrainingRoundStep(GenesisTrainingOrchestrator orchestrator)
    {
        _orchestrator = orchestrator;
    }

    public GenesisAutonomousTrainTaskData Execute(GenesisAutonomousTrainTaskData data)
    {
        var plan = data.Plan ?? throw new InvalidOperationException("Autonomous plan is required.");
        var examples = data.TrainingExamples ?? throw new InvalidOperationException("Training examples are required.");
        var creatorMap = data.ExampleCreatorMap;
        data.CancellationToken.ThrowIfCancellationRequested();
        data.UiLogger?.Invoke($"[auto] round {plan.Round}: starting train step (examples={examples.Count}, epochs={plan.Epochs})...");
        var report = _orchestrator.Train(examples, plan.Epochs, data.UiLogger, data.CancellationToken, creatorMap);
        var creatorRounds = (data.CreatorRounds ?? [])
            .Select(r => r with { Report = report })
            .ToArray();

        return data with
        {
            CreatorRounds = creatorRounds,
            Report = report
        };
    }
}
