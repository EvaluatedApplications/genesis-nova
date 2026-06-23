using EvalApp.Consumer;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Licensing;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed partial class GenesisEvalAppRuntime
{
    public int[] EncodeTokens(string text) => WithModelGate(() => _state.Tokenizer.Encode(text));
    public string DecodeTokens(IReadOnlyList<int> tokens) => WithModelGate(() => _state.Tokenizer.Decode(tokens));
    public string TokenText(int tokenId)
    {
        return WithModelGate(() =>
        {
            var vocab = _state.Tokenizer.Vocabulary;
            return tokenId >= 0 && tokenId < vocab.Count
                ? vocab[tokenId]
                : $"<unk:{tokenId}>";
        });
    }

    /// <summary>Decode many token ids in ONE model-gate acquisition (instead of N separate <see cref="TokenText"/>
    /// calls each taking the gate). Each returned string is identical to what <see cref="TokenText"/> would yield
    /// for that id, in input order.</summary>
    public IReadOnlyList<string> TokenTexts(IReadOnlyList<int> tokens)
    {
        return WithModelGate(() =>
        {
            var vocab = _state.Tokenizer.Vocabulary;
            var texts = new string[tokens.Count];
            for (var i = 0; i < tokens.Count; i++)
            {
                var tokenId = tokens[i];
                texts[i] = tokenId >= 0 && tokenId < vocab.Count
                    ? vocab[tokenId]
                    : $"<unk:{tokenId}>";
            }
            return (IReadOnlyList<string>)texts;
        });
    }

    /// <summary>READ-ONLY push/pull geometry summary of the loaded platonic space (related vs unrelated
    /// concept distances — the magnitude the contrastive dynamics achieved).</summary>
    public Cognition.PlatonicSpaceMemory.GeometrySummary GeometrySummary()
        => WithModelGate(() => _state.Memory.SummarizePushPullGeometry());

    public PlatonicActivationView AnalyzePlatonicActivation(string input, int maxNodes = 24, int maxEdges = 40)
    {
        return WithModelGate(() =>
        {
            var safeInput = input ?? string.Empty;
            var tokenIds = _state.Tokenizer.Encode(safeInput);
            var tokenTexts = tokenIds.Select(id =>
            {
                var vocab = _state.Tokenizer.Vocabulary;
                return id >= 0 && id < vocab.Count ? vocab[id] : $"<unk:{id}>";
            }).ToArray();
            var lexicalParts = System.Text.RegularExpressions.Regex
                .Matches(safeInput.ToLowerInvariant(), @"-?\d+(?:\.\d+)?|[a-z]+|[+\-*/x]")
                .Select(m => m.Value)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            var anchorSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var token in tokenTexts.Concat(lexicalParts))
            {
                if (_state.Memory.ContainsConcept(token))
                    anchorSet.Add(token);
            }

            var anchors = anchorSet.OrderBy(a => a, StringComparer.OrdinalIgnoreCase).ToArray();
            var snapshot = _state.Memory.ExportSnapshot();
            var nodes = new List<PlatonicActivatedNode>(snapshot.Nodes.Length);
            var anchorHash = new HashSet<string>(anchors, StringComparer.OrdinalIgnoreCase);

            foreach (var node in snapshot.Nodes)
            {
                var isAnchor = anchorHash.Contains(node.Name);
                var baseScore = isAnchor ? 1.0 : 0.0;
                if (!isAnchor && anchors.Length > 0)
                {
                    baseScore = anchors
                        .Select(anchor => 1.0 - _state.Memory.GetContradiction(anchor, node.Name))
                        .DefaultIfEmpty(0.0)
                        .Max();
                }

                var obsBoost = Math.Min(0.20, Math.Log10(Math.Max(1, node.ObservationCount)) / 10.0);
                var score = Math.Max(0.0, Math.Min(1.0, baseScore + obsBoost));
                nodes.Add(new PlatonicActivatedNode(node.Name, score, node.ObservationCount, isAnchor));
            }

            var selectedNodes = nodes
                .Where(n => !GenesisNova.Cognition.PlatonicSpaceMemory.IsReservedConcept(n.Name)) // hide internal face: routing markers
                .OrderByDescending(n => n.IsAnchor)
                .ThenByDescending(n => n.Score)
                .ThenByDescending(n => n.ObservationCount)
                .Take(Math.Max(4, maxNodes))
                .ToArray();

            var selectedSet = new HashSet<string>(selectedNodes.Select(n => n.Name), StringComparer.OrdinalIgnoreCase);
            var edges = snapshot.Relations
                .Where(r =>
                    (selectedSet.Contains(r.Left) && selectedSet.Contains(r.Right)) ||
                    anchorHash.Contains(r.Left) || anchorHash.Contains(r.Right))
                .Select(r =>
                {
                    var confidence = 1.0 - r.SynthesisContradiction;
                    var obsBoost = Math.Min(0.15, Math.Log10(Math.Max(1, r.ObservationCount)) / 12.0);
                    var score = Math.Max(0.0, Math.Min(1.0, confidence + obsBoost));
                    return new PlatonicActivatedEdge(
                        Left: r.Left,
                        Right: r.Right,
                        Score: score,
                        Contradiction: r.SynthesisContradiction,
                        ObservationCount: r.ObservationCount);
                })
                .OrderByDescending(e => e.Score)
                .ThenByDescending(e => e.ObservationCount)
                .Take(Math.Max(8, maxEdges))
                .ToArray();

            return new PlatonicActivationView(
                Input: safeInput,
                InputTokens: tokenTexts,
                Anchors: anchors,
                Nodes: selectedNodes,
                Edges: edges);
        });
    }

    /// <summary>
    /// Structured introspection of the live model + platonic substrate (for the diagnostic CLI). Reads the
    /// SAME runtime state inference uses, under the model gate, so it never diverges from what the model does.
    /// </summary>
    public GenesisRuntimeDiagnostics Diagnose(int topRelations = 12, int topFunctions = 16, int topChunks = 12)
    {
        return WithModelGate(() =>
        {
            var model = _state.Model;
            var mem = _state.Memory;
            var trainer = _state.Trainer;

            var transforms = trainer.TransformAccumulator.ExportSnapshot().Transforms;
            var folds = trainer.FoldPathDiscovery.ExportSnapshot();
            var chunks = mem.ExportSnapshot().Chunks ?? Array.Empty<PlatonicChunkSnapshot>();

            var funcs = mem.FunctionElements;
            var funcById = funcs.ToDictionary(f => f.Id, f => f.Symbol);
            var functionSummaries = funcs
                .Take(topFunctions)
                .Select(f => new FunctionElementSummary(
                    f.Symbol,
                    f.RelatedTo.Select(id => funcById.TryGetValue(id, out var s) ? s : $"#{id}").ToArray()))
                .ToArray();

            var topRel = mem.GetAllRelations()
                .OrderByDescending(r => r.ObservationCount)
                .ThenBy(r => r.Left, StringComparer.OrdinalIgnoreCase)
                .Take(topRelations)
                .Select(r => new RelationSummary(r.Left, r.Right, r.ObservationCount))
                .ToArray();

            var path = GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);

            return new GenesisRuntimeDiagnostics(
                CheckpointPath: path,
                CheckpointExists: File.Exists(path),
                CheckpointWriteUtc: File.Exists(path) ? File.GetLastWriteTimeUtc(path) : null,
                Backend: _runtimeConfig.Backend.ToString(),
                HiddenSize: model.HiddenSize,
                FaceDimension: mem.FaceDimension,
                VocabularySize: _state.Tokenizer.VocabularySize,
                ParameterCount: model.ParameterCount(),
                PlanKindCount: GenesisNeuralModel.PlanKindCount,
                NodeCount: mem.NodeCount,
                RelationCount: mem.RelationCount,
                FunctionElementCount: funcs.Count,
                LearnedTransformCount: transforms.Count,
                FoldPathCount: folds.FoldPaths.Count,
                LogLinearFitCount: folds.LogLinearFits.Count,
                ChunkTagCount: chunks.Select(c => c.Tag).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
                ChunkCount: chunks.Length,
                AutonomousRounds: _historyStore.History.Count,
                MaxNodes: Math.Max(256, _runtimeConfig.MaxPlatonicNodes),
                MaxRelations: Math.Max(1024, _runtimeConfig.MaxPlatonicRelations),
                SpaceManagerEnabled: _runtimeConfig.AutoManagePlatonicSpace,
                // Soft relation budget the SpaceManager prunes toward: nodes×TargetRelationsPerNode(6)+NodeBuffer(128),
                // clamped to [MinRelations(1024), MaxRelations]. Relation-pressure = RelationCount / this budget.
                RelationBudget: Math.Clamp(mem.NodeCount * 6 + 128, 1024, Math.Max(1024, _runtimeConfig.MaxPlatonicRelations)),
                TopRelations: topRel,
                FunctionElements: functionSummaries,
                LearnedTransforms: transforms
                    .Take(topFunctions)
                    .Select(t => new TransformSummary(t.FunctionName, t.ObservationCount, t.Confidence, t.State.ToString()))
                    .ToArray(),
                Chunks: chunks
                    .OrderByDescending(c => c.Count)
                    .Take(topChunks)
                    .Select(c => new ChunkSummary(c.Tag, c.Chunk, c.Count))
                    .ToArray());
        });
    }
}
