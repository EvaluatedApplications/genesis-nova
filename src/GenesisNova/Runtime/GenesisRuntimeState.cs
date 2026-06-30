using System;
using System.Linq;
using System.Text.Json;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using TorchSharp;
using static TorchSharp.torch;

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
        Navigator = CreateNavigator(config);
        WireNavigatorDisambiguator();
    }

    public GenesisNovaConfig Config { get; private set; }
    public WhitespaceGenesisTokenizer Tokenizer { get; private set; }
    public GenesisNeuralModel Model { get; private set; }
    public IPlatonicSpace Memory { get; private set; }
    public GenesisConversationMemory Conversation { get; private set; }
    public GenesisTrainer Trainer { get; private set; }
    public GenesisTrainingOrchestrator Orchestrator { get; private set; }
    public GenesisInferenceEngine Inference { get; private set; }

    /// <summary>THE PLATONIC NAVIGATOR — one shared query-conditioned policy net (gym trains it; REPL/inspect read it).
    /// Built at <c>config.FaceDimension</c> (production 1024 ⇒ self length 608), hidden 2048, the {GENUS,DOMAIN,ROOT}
    /// cue table. Its weights + the engine's persistent self are checkpointed together so an overnight-trained navigator
    /// survives an app restart (a load restores them via <see cref="Replace"/>; an old checkpoint keeps this fresh net).</summary>
    public NavQueryPolicyNet Navigator { get; private set; }

    private static NavQueryPolicyNet CreateNavigator(GenesisNovaConfig config)
        => new(config.FaceDimension, NavQueryFeatures.CueCount, hidden: 2048);

    public void Replace(
        GenesisNovaConfig config,
        WhitespaceGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicMemorySnapshot? platonicSpaceSnapshot = null,
        GenesisConversationSnapshot? conversationSnapshot = null,
        string? trainerLearningStateJson = null,
        GrammarRoleSnapshot[]? grammarRoles = null,
        NavigatorSnapshot? navigator = null,
        double[]? navigatorSelfField = null)
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

        // Rebuild the shared navigator fresh, then restore its trained weights (no-op/fresh if the checkpoint had none —
        // an old checkpoint, or a navigator built at different dims, both leave a freshly-initialised net). Dispose the
        // previous net (it is a TorchSharp module) so a reload doesn't leak its tensors. Restore the persistent self too
        // (null → the engine resumes self-less), so the mind reasons from WHO IT HAD BECOME across the restart.
        Navigator?.Dispose();
        Navigator = CreateNavigator(config);
        Navigator.ImportWeights(navigator);
        Inference.RestoreSelfField(navigatorSelfField);
        WireNavigatorDisambiguator();
    }

    /// <summary>M1 (PLATONIC_NAVIGATOR.md): attach the trained shared navigator to the inference engine's
    /// AMBIGUOUS-BRANCH disambiguator hook, when <see cref="GenesisNovaConfig.NavigatorDisambiguation"/> is on and the
    /// substrate is the dialectical core (the only one the navigator walks). The delegate runs <see cref="Navigator"/>
    /// as a <see cref="QueryNavPolicy"/> walk from the anchor under the query's cue, seeded by the engine's persistent
    /// self, and returns the decoded landing + whether the walk CONFIDENTLY halted (so the engine can fall through to
    /// the one-shot reason on a non-resolve). The net rests on whatever device its weights are on (CPU between gym
    /// cycles); the walk is serialized with predict/train by the runtime's model gate, so torch is single-threaded here.
    /// When the flag is off (or the space is legacy) the hook is left null ⇒ the ambiguous branch is byte-identical.</summary>
    private void WireNavigatorDisambiguator()
    {
        if (!Config.NavigatorDisambiguation || Memory is not DialecticalSpace ds)
        {
            Inference.NavigatorDisambiguator = null;
            return;
        }
        var net = Navigator;
        Inference.NavigatorDisambiguator = (anchor, cue, self, kindFace) =>
        {
            if (string.IsNullOrWhiteSpace(anchor) || !ds.TryGetConceptFace(anchor, out var anchorFace))
                return (string.Empty, 0.0, false);
            var device = net.parameters().FirstOrDefault()?.device ?? CPU;
            try
            {
                using var policy = new QueryNavPolicy(net, ds, anchorFace, (int)cue, device,
                    NavQueryDaggerTrainer.DefaultK, minConfidence: 0.0, haltThreshold: 0.5, selfVec: self, kindFace: kindFace);
                var res = new NavigatorWalk().Walk(ds, anchor, anchorFace, goalSymbol: null, policy,
                    new NavWalkOptions(MaxSteps: 8));
                // A CONFIDENT halt (the learned halt head fired) on a concept OTHER than the start = the query relaxed
                // to an answer. A budget-exhausted stop, or halting back on the anchor, is not a resolution → abstain.
                var ok = policy.LastHalt && !string.Equals(res.FinalSymbol, anchor, StringComparison.Ordinal);
                return (res.FinalSymbol, ok ? 0.9 : 0.0, ok);
            }
            catch
            {
                return (string.Empty, 0.0, false); // a single bad walk must never break a predict
            }
        };
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
