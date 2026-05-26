using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Persistence;
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
        Memory = new PlatonicSpaceMemory(faceDimension: Math.Max(4, config.HiddenSize / 2), seed: config.Seed);
        Cognition = new PlatonicIntrospectionEngine(Memory);
        Conversation = new GenesisConversationMemory();
        Trainer = new GenesisTrainer(Tokenizer, Model, Cognition);
        Orchestrator = new GenesisTrainingOrchestrator(Trainer);
        Inference = new GenesisInferenceEngine(Tokenizer, Model, Cognition);
    }

    public GenesisNovaConfig Config { get; private set; }
    public WhitespaceGenesisTokenizer Tokenizer { get; private set; }
    public GenesisNeuralModel Model { get; private set; }
    public PlatonicSpaceMemory Memory { get; private set; }
    public PlatonicIntrospectionEngine Cognition { get; private set; }
    public GenesisConversationMemory Conversation { get; private set; }
    public GenesisTrainer Trainer { get; private set; }
    public GenesisTrainingOrchestrator Orchestrator { get; private set; }
    public GenesisInferenceEngine Inference { get; private set; }

    public void Replace(
        GenesisNovaConfig config,
        WhitespaceGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicCognitionSnapshot? cognitionSnapshot,
        GenesisConversationSnapshot? conversationSnapshot = null)
    {
        ConfigureCpuThreadPool(config);
        Config = config;
        Tokenizer = tokenizer;
        Model = model;
        Memory = new PlatonicSpaceMemory(faceDimension: Math.Max(4, config.HiddenSize / 2), seed: config.Seed);
        Cognition = new PlatonicIntrospectionEngine(Memory);
        Conversation = new GenesisConversationMemory();
        Trainer = new GenesisTrainer(Tokenizer, Model, Cognition);
        if (cognitionSnapshot is not null)
            Trainer.ImportCognitionSnapshot(cognitionSnapshot);
        if (conversationSnapshot is not null)
            Conversation.ImportSnapshot(conversationSnapshot);
        Orchestrator = new GenesisTrainingOrchestrator(Trainer);
        Inference = new GenesisInferenceEngine(Tokenizer, Model, Cognition);
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

}
