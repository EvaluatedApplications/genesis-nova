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
        GliderInterpreter = new PlatonicGliderInterpreter(Memory);
        Conversation = new GenesisConversationMemory();
        Trainer = new GenesisTrainer(Tokenizer, Model, Memory, config);
        Orchestrator = new GenesisTrainingOrchestrator(Trainer, config);
        Inference = new GenesisInferenceEngine(
            Tokenizer,
            Model,
            Memory,
            ResolvePlatonicCheckpointPath,
            Trainer.FoldPathDiscovery,
            Trainer.TransformAccumulator,
            // Production enables the exact direct face-arithmetic route: when the router selects
            // platonic-direct for a compact arithmetic query, it is answered exactly via the face
            // homomorphism instead of silently falling through to neural.
            enableDiagnosticFaceArithmeticShortcut: true);
        Trainer.SetInferencePolicy(Inference);
    }

    public GenesisNovaConfig Config { get; private set; }
    public WhitespaceGenesisTokenizer Tokenizer { get; private set; }
    public GenesisNeuralModel Model { get; private set; }
    public PlatonicSpaceMemory Memory { get; private set; }
    // Reusable glider blocks (Operand/Literal/Hop/Compute/Seq) made first-class in the runtime: the
    // deterministic interpreter that executes a composition of blocks on the platonic physics. Available
    // for hand-built or (future) GRU-constructed gliders; see PROJECT_GLIDER.md.
    public PlatonicGliderInterpreter GliderInterpreter { get; private set; }
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
        if (platonicSpaceSnapshot is not null)
            Memory.ImportSnapshot(platonicSpaceSnapshot);
        GliderInterpreter = new PlatonicGliderInterpreter(Memory);
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
            Trainer.FoldPathDiscovery,
            Trainer.TransformAccumulator,
            // Production enables the exact direct face-arithmetic route (see ctor above).
            enableDiagnosticFaceArithmeticShortcut: true);
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
