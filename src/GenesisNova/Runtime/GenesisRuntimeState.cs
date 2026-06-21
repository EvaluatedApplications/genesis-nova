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
        Memory = new PlatonicSpaceMemory(
            faceDimension: config.FaceDimension,
            seed: config.Seed,
            maxNodes: config.MaxPlatonicNodes,
            maxRelations: config.MaxPlatonicRelations);
        Memory.UseInfoNceRepulsion = false; // C4/C2: MANUAL constant-step repulsion is live — InfoNCE's proportional
                                            // push is unbounded off the unit sphere (normalization is now clamp-only)
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
        Trainer.SetInferencePolicy(Inference);
    }

    public GenesisNovaConfig Config { get; private set; }
    public WhitespaceGenesisTokenizer Tokenizer { get; private set; }
    public GenesisNeuralModel Model { get; private set; }
    public PlatonicSpaceMemory Memory { get; private set; }
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
        string? trainerLearningStateJson = null)
    {
        ConfigureCpuThreadPool(config);
        Config = config;
        Tokenizer = tokenizer;
        Model = model;
        Memory = new PlatonicSpaceMemory(
            faceDimension: config.FaceDimension,
            seed: config.Seed,
            maxNodes: config.MaxPlatonicNodes,
            maxRelations: config.MaxPlatonicRelations);
        Memory.UseInfoNceRepulsion = false; // C4/C2: MANUAL constant-step repulsion is live — InfoNCE's proportional
                                            // push is unbounded off the unit sphere (normalization is now clamp-only)
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
        Trainer.SetInferencePolicy(Inference);
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

    private string? ResolvePlatonicCheckpointPath()
    {
        var path = GenesisLocalStateStore.ResolveCheckpointPath(Config);
        return File.Exists(path) ? path : null;
    }

}
