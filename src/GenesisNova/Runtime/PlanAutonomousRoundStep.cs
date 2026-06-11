using EvalApp.Consumer;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class PlanAutonomousRoundStep : IStep<GenesisAutonomousTrainTaskData>
{
    private readonly GenesisAutonomousTrainingPlanner _planner;

    public PlanAutonomousRoundStep(GenesisAutonomousTrainingPlanner planner)
    {
        _planner = planner;
    }

    public ValueTask<GenesisAutonomousTrainTaskData> ExecuteAsync(GenesisAutonomousTrainTaskData data, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        data.UiLogger?.Invoke($"[auto] round {data.RoundIndex + 1}: planner entering...");
        var plan = _planner.SuggestComposite(data.Request, data.History, data.RoundIndex);
        data.UiLogger?.Invoke($"[auto] round {plan.Round}: planner ready.");
        data.UiLogger?.Invoke($"[auto] round {plan.Round}: {plan.Reason}");
        foreach (var creator in plan.CreatorPlans)
        {
            data.UiLogger?.Invoke(
                $"[auto]   dataset={creator.CreatorName} pool={creator.SampleCount} train={creator.TrainCount} difficulty={creator.Difficulty} priority={creator.Priority:F3}");
        }

        return ValueTask.FromResult(data with { Plan = plan });
    }
}
