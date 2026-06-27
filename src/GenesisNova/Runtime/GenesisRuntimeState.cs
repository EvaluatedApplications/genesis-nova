using System.Linq;
using System.Text.Json;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed class GenesisRuntimeState
{
    public GenesisRuntimeState(GenesisNovaConfig config)
    {
        ConfigureCpuThreadPool(config);
        Config = config;
        Tokenizer = new WhitespaceGenesisTokenizer();
        Model = new GenesisNeuralModel(config);
        Memory = CreateMemory(config);
        Conversation = new GenesisConversationMemory();
        Trainer = new GenesisTrainer(Tokenizer, Model, Memory, config);
        Orchestrator = new GenesisTrainingOrchestrator(Trainer, config);
        Inference = new GenesisInferenceEngine(
            Tokenizer,
            Model,
            Memory,
            ResolvePlatonicCheckpointPath,
            transformAccumulator: Trainer.TransformAccumulator,
            foldPathDiscovery: Trainer.FoldPathDiscovery);
        NovaConfig.FromLegacy(config).ApplyTo(Model, Memory, Inference, Trainer); // ONE place for every mechanism toggle
        Trainer.SetInferencePolicy(Inference);
    }

    public GenesisNovaConfig Config { get; private set; }
    public WhitespaceGenesisTokenizer Tokenizer { get; private set; }
    public GenesisNeuralModel Model { get; private set; }
    public IPlatonicSpace Memory { get; private set; }
    public GenesisConversationMemory Conversation { get; private set; }
    public GenesisTrainer Trainer { get; private set; }
    public GenesisTrainingOrchestrator Orchestrator { get; private set; }
    public GenesisInferenceEngine Inference { get; private set; }

    public void Replace(
        GenesisNovaConfig config,
        WhitespaceGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicMemorySnapshot? platonicSpaceSnapshot = null,
        GenesisConversationSnapshot? conversationSnapshot = null,
        string? trainerLearningStateJson = null,
        GrammarRoleSnapshot[]? grammarRoles = null)
    {
        ConfigureCpuThreadPool(config);
        Config = config;
        Tokenizer = tokenizer;
        Model = model;
        Memory = CreateMemory(config);
        if (platonicSpaceSnapshot is not null)
            Memory.ImportSnapshot(platonicSpaceSnapshot);
        Conversation = new GenesisConversationMemory();
        Trainer = new GenesisTrainer(Tokenizer, Model, Memory, config);
        ImportTrainerLearningState(trainerLearningStateJson);
        if (conversationSnapshot is not null)
            Conversation.ImportSnapshot(conversationSnapshot);
        Orchestrator = new GenesisTrainingOrchestrator(Trainer, config);
        Inference = new GenesisInferenceEngine(
            Tokenizer,
            Model,
            Memory,
            ResolvePlatonicCheckpointPath,
            transformAccumulator: Trainer.TransformAccumulator,
            foldPathDiscovery: Trainer.FoldPathDiscovery);
        NovaConfig.FromLegacy(config).ApplyTo(Model, Memory, Inference, Trainer); // ONE place for every mechanism toggle
        Trainer.SetInferencePolicy(Inference);
        // Restore the learned grammar tallies into the freshly-built engine — the role head's TRAINING-LABEL source.
        // Without this the head reloads with empty supervision and desyncs (grammar regresses, loss sticks).
        if (grammarRoles is { Length: > 0 })
            Inference.ImportGrammarRoles(grammarRoles.Select(r => (r.Token, r.Present, r.Absent, r.AsAnswer, r.AsCopula)));
    }

    private void ImportTrainerLearningState(string? trainerLearningStateJson)
    {
        if (string.IsNullOrWhiteSpace(trainerLearningStateJson))
            return;

        try
        {
            var state = JsonSerializer.Deserialize<GenesisTrainerLearningState>(trainerLearningStateJson);
            Trainer.ImportLearningState(state);
        }
        catch
        {
            Trainer.ImportLearningState(null);
        }
    }

    private static void ConfigureCpuThreadPool(GenesisNovaConfig config)
    {
        if (config.Deterministic)
        {
            ThreadPool.GetMinThreads(out _, out var deterministicCompletionThreads);
            ThreadPool.SetMinThreads(1, deterministicCompletionThreads);
            return;
        }

        var target = config.MaxDegreeOfParallelism <= 0
            ? Environment.ProcessorCount
            : config.MaxDegreeOfParallelism;

        ThreadPool.GetMinThreads(out var workerThreads, out var completionThreads);
        if (workerThreads < target)
            ThreadPool.SetMinThreads(target, completionThreads);
    }

    /// <summary>Select the substrate implementation behind the IPlatonicSpace contract (PLATONIC_THEORY.md §11
    /// rebuild). Default = legacy PlatonicSpaceMemory; <c>UseDialecticalCore</c> = the ground-up DialecticalSpace.</summary>
    private static IPlatonicSpace CreateMemory(GenesisNovaConfig config)
        => config.UseDialecticalCore
            ? new Cognition.Platonic.DialecticalSpace(config.FaceDimension, config.Seed)
            : new PlatonicSpaceMemory(
                faceDimension: config.FaceDimension,
                seed: config.Seed,
                maxNodes: config.MaxPlatonicNodes,
                maxRelations: config.MaxPlatonicRelations);

    private string? ResolvePlatonicCheckpointPath()
    {
        var path = GenesisLocalStateStore.ResolveCheckpointPath(Config);
        return File.Exists(path) ? path : null;
    }

}
