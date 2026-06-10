using GenesisNova.Core;
using GenesisNova.Persistence;

namespace GenesisNova.Runtime;

internal sealed class LoadStep
{
    private readonly GenesisRuntimeState _state;
    private readonly AutonomousHistoryStore _historyStore;
    private readonly GenesisCheckpointPersister _persister;
    private readonly GenesisNovaConfig _runtimeConfig;

    public LoadStep(
        GenesisRuntimeState state,
        AutonomousHistoryStore historyStore,
        GenesisCheckpointPersister persister,
        GenesisNovaConfig runtimeConfig)
    {
        _state = state;
        _historyStore = historyStore;
        _persister = persister;
        _runtimeConfig = runtimeConfig;
    }

    public GenesisLoadTaskData Execute(GenesisLoadTaskData data)
    {
        var loaded = GenesisCheckpointStore.LoadForRuntime(data.Path, _runtimeConfig);
        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation);
        _historyStore.Restore(loaded.AutonomousTraining);
        _persister.Persist(reason: "load", detail: data.Path);
        return data with { Loaded = true };
    }
}
