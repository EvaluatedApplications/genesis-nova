using GenesisNova.Cognition;

namespace GenesisNova.Core;

public enum SpaceToolKind
{
    Observe = 1,
    Stabilize = 2,
    Expand = 3,
    Rebalance = 4,
    Reinforce = 5,
    DefaultAlgorithm = 6,
    CreateConcept = 7,
    EditConceptFace = 8,
    EditRelationContradiction = 9,
    CreateOrStrengthenRelation = 10,
    WeakenOrDecayRelation = 11,
    TriadConsistencyEdit = 12,
    NeighborhoodRetype = 13,
    CentroidPullPush = 14,
    MergeConceptHint = 15,
    PruneHint = 16,
    AnchorBindingEdit = 17,
    AttentionScopeSelect = 18,
    CommitLevelSet = 19,
    RewardTagEmit = 20,
    DiscoverAbstractions = 21
}

public sealed record SpaceManagerSettings(
    bool Enabled = true,
    int MinNodes = 256,
    int MaxNodes = 12_000,
    int MinRelations = 1_024,
    int MaxRelations = 48_000,
    int TargetRelationsPerNode = 6,
    int NodeBuffer = 128,
    double NoiseThreshold = 0.65,
    double MinUtilityToKeep = 0.15,
    int MaxRelationPrunesPerCycle = 48,
    int MaxNodePrunesPerCycle = 6,
    int MaxNodeMergesPerCycle = 3,
    int MaxRebalancePrunesPerCycle = 48);

public sealed record SpaceManagementResult(
    bool Compacted,
    int NodesBefore,
    int NodesAfter,
    int RelationsBefore,
    int RelationsAfter,
    int NodesPruned,
    int RelationsPruned,
    double NoiseRatio,
    int RelationBudget,
    SpaceToolKind RecommendedTool = SpaceToolKind.Observe);

public sealed class SpaceManager
{
    private const int AnchorObservationProtectionThreshold = 8;
    private const int AnchorDegreeProtectionThreshold = 16;

    private readonly PlatonicSpaceMemory _memory;
    private readonly SpaceManagerSettings _settings;

    public SpaceManager(PlatonicSpaceMemory memory, SpaceManagerSettings? settings = null)
    {
        _memory = memory;
        _settings = settings ?? new SpaceManagerSettings();
    }

    public SpaceManagementResult Manage()
    {
        var snapshot = _memory.ExportSnapshot();
        var nodes = snapshot.Nodes.Length;
        var relations = snapshot.Relations.Length;
        var relationTuples = snapshot.Relations
            .Select(r => (r.Left, r.Right, ObservationCount: (long)r.ObservationCount))
            .ToArray();
        var quality = TransformQualityMetrics.GenerateReport(relationTuples);
        var relationBudget = ComputeTargetRelationBudget(nodes);

        return new SpaceManagementResult(
            Compacted: false,
            NodesBefore: nodes,
            NodesAfter: nodes,
            RelationsBefore: relations,
            RelationsAfter: relations,
            NodesPruned: 0,
            RelationsPruned: 0,
            NoiseRatio: quality.NoiseRatio,
            RelationBudget: relationBudget,
            RecommendedTool: SpaceToolKind.Observe);
    }

    public SpaceManagementResult ExecuteTool(
        SpaceToolKind tool,
        IReadOnlyCollection<string>? focusConcepts = null)
    {
        var assessmentBefore = Manage();
        if (!_settings.Enabled || tool == SpaceToolKind.Observe)
            return assessmentBefore with { RecommendedTool = tool };

        var protectedConcepts = BuildProtectedConcepts(focusConcepts);
        var relationsPruned = 0;
        var nodesPruned = 0;
        var nodesMerged = 0;
        var linksAdded = 0;
        var editsApplied = 0;

        switch (tool)
        {
            case SpaceToolKind.DefaultAlgorithm:
                ExecuteDefaultAlgorithm(
                    assessmentBefore,
                    protectedConcepts,
                    ref relationsPruned,
                    ref nodesPruned,
                    ref nodesMerged,
                    ref linksAdded);
                break;
            case SpaceToolKind.Stabilize:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: _settings.MaxRelationPrunesPerCycle,
                        MaxNodePrunes: _settings.MaxNodePrunesPerCycle,
                        MaxNodeMerges: _settings.MaxNodeMergesPerCycle,
                        MaxRebalancePrunes: Math.Max(1, _settings.MaxRebalancePrunesPerCycle / 2),
                        MaxRelationsPerNode: Math.Max(2, _settings.TargetRelationsPerNode + 2),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.70,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned = maintenance.RelationsPruned;
                    nodesPruned = maintenance.NodesPruned;
                    nodesMerged = maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.Rebalance:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: Math.Max(1, _settings.MaxRelationPrunesPerCycle / 2),
                        MaxNodePrunes: Math.Max(1, _settings.MaxNodePrunesPerCycle / 3),
                        MaxNodeMerges: Math.Max(1, _settings.MaxNodeMergesPerCycle / 2),
                        MaxRebalancePrunes: _settings.MaxRebalancePrunesPerCycle,
                        MaxRelationsPerNode: Math.Max(1, _settings.TargetRelationsPerNode),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.78,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned = maintenance.RelationsPruned;
                    nodesPruned = maintenance.NodesPruned;
                    nodesMerged = maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.Expand:
                linksAdded = SeedExplorationLinks(protectedConcepts, Math.Max(8, _settings.MaxNodeMergesPerCycle * 2));
                break;
            case SpaceToolKind.Reinforce:
                linksAdded = ReinforceFocusNeighborhood(protectedConcepts, Math.Max(8, _settings.MaxRelationPrunesPerCycle / 4));
                break;
            case SpaceToolKind.CreateConcept:
                editsApplied = CreateConcepts(protectedConcepts, Math.Max(1, _settings.MaxNodeMergesPerCycle));
                break;
            case SpaceToolKind.EditConceptFace:
                editsApplied = EditConceptFaces(protectedConcepts, Math.Max(1, _settings.MaxNodeMergesPerCycle));
                break;
            case SpaceToolKind.EditRelationContradiction:
                editsApplied = EditRelationContradictions(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 8));
                break;
            case SpaceToolKind.CreateOrStrengthenRelation:
                linksAdded = CreateOrStrengthenRelations(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 6));
                break;
            case SpaceToolKind.WeakenOrDecayRelation:
                editsApplied = WeakenOrDecayRelations(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 6));
                break;
            case SpaceToolKind.TriadConsistencyEdit:
                editsApplied = TriadConsistencyEdit(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 8));
                break;
            case SpaceToolKind.NeighborhoodRetype:
                editsApplied = NeighborhoodRetype(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 8));
                break;
            case SpaceToolKind.CentroidPullPush:
                editsApplied = CentroidPullPush(protectedConcepts, Math.Max(1, _settings.MaxNodeMergesPerCycle / 2));
                break;
            case SpaceToolKind.MergeConceptHint:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: 0,
                        MaxNodePrunes: 0,
                        MaxNodeMerges: Math.Max(1, _settings.MaxNodeMergesPerCycle),
                        MaxRebalancePrunes: 0,
                        MaxRelationsPerNode: 0,
                        TargetRelationCount: 0,
                        MinRelationsToKeep: _settings.MinRelations,
                        MinNodesToKeep: _settings.MinNodes,
                        ProtectedConcepts: protectedConcepts));
                    nodesMerged = maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.PruneHint:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: Math.Max(1, _settings.MaxRelationPrunesPerCycle / 2),
                        MaxNodePrunes: Math.Max(1, _settings.MaxNodePrunesPerCycle / 2),
                        MaxNodeMerges: 0,
                        MaxRebalancePrunes: Math.Max(1, _settings.MaxRebalancePrunesPerCycle / 3),
                        MaxRelationsPerNode: Math.Max(2, _settings.TargetRelationsPerNode + 1),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.74,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned = maintenance.RelationsPruned;
                    nodesPruned = maintenance.NodesPruned;
                    break;
                }
            case SpaceToolKind.AnchorBindingEdit:
                linksAdded = AnchorBindingEdit(protectedConcepts, Math.Max(2, _settings.MaxNodeMergesPerCycle));
                break;
            case SpaceToolKind.AttentionScopeSelect:
                editsApplied = AttentionScopeSelect(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 10));
                break;
            case SpaceToolKind.CommitLevelSet:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: Math.Max(1, _settings.MaxRelationPrunesPerCycle / 3),
                        MaxNodePrunes: Math.Max(1, _settings.MaxNodePrunesPerCycle / 3),
                        MaxNodeMerges: Math.Max(1, _settings.MaxNodeMergesPerCycle / 3),
                        MaxRebalancePrunes: Math.Max(1, _settings.MaxRebalancePrunesPerCycle / 2),
                        MaxRelationsPerNode: Math.Max(2, _settings.TargetRelationsPerNode),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.72,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned = maintenance.RelationsPruned;
                    nodesPruned = maintenance.NodesPruned;
                    nodesMerged = maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.RewardTagEmit:
                editsApplied = RewardTagEmit(protectedConcepts, Math.Max(2, _settings.MaxRelationPrunesPerCycle / 8));
                break;
            case SpaceToolKind.DiscoverAbstractions:
                linksAdded = DiscoverAbstractions(protectedConcepts);
                break;
        }

        var assessmentAfter = Manage();
        var changed = relationsPruned > 0 || nodesPruned > 0 || nodesMerged > 0 || linksAdded > 0 || editsApplied > 0;
        return new SpaceManagementResult(
            Compacted: changed,
            NodesBefore: assessmentBefore.NodesAfter,
            NodesAfter: assessmentAfter.NodesAfter,
            RelationsBefore: assessmentBefore.RelationsAfter,
            RelationsAfter: assessmentAfter.RelationsAfter,
            NodesPruned: nodesPruned,
            RelationsPruned: relationsPruned,
            NoiseRatio: assessmentAfter.NoiseRatio,
            RelationBudget: assessmentAfter.RelationBudget,
            RecommendedTool: tool);
    }

    private void ExecuteDefaultAlgorithm(
        SpaceManagementResult assessmentBefore,
        IReadOnlyCollection<string> protectedConcepts,
        ref int relationsPruned,
        ref int nodesPruned,
        ref int nodesMerged,
        ref int linksAdded)
    {
        var baselineTool = SelectBaselineTool(assessmentBefore);
        switch (baselineTool)
        {
            case SpaceToolKind.Stabilize:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: _settings.MaxRelationPrunesPerCycle,
                        MaxNodePrunes: _settings.MaxNodePrunesPerCycle,
                        MaxNodeMerges: _settings.MaxNodeMergesPerCycle,
                        MaxRebalancePrunes: Math.Max(1, _settings.MaxRebalancePrunesPerCycle / 2),
                        MaxRelationsPerNode: Math.Max(2, _settings.TargetRelationsPerNode + 2),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.70,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned += maintenance.RelationsPruned;
                    nodesPruned += maintenance.NodesPruned;
                    nodesMerged += maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.Rebalance:
                {
                    var maintenance = _memory.ApplyMaintenance(new PlatonicSpaceMemory.SpaceMaintenanceRequest(
                        MaxRelationPrunes: Math.Max(1, _settings.MaxRelationPrunesPerCycle / 2),
                        MaxNodePrunes: Math.Max(1, _settings.MaxNodePrunesPerCycle / 3),
                        MaxNodeMerges: Math.Max(1, _settings.MaxNodeMergesPerCycle / 2),
                        MaxRebalancePrunes: _settings.MaxRebalancePrunesPerCycle,
                        MaxRelationsPerNode: Math.Max(1, _settings.TargetRelationsPerNode),
                        TargetRelationCount: assessmentBefore.RelationBudget,
                        MinRelationsToKeep: Math.Max(_settings.MinRelations, assessmentBefore.RelationBudget / 2),
                        MinNodesToKeep: _settings.MinNodes,
                        MaxSynthesisContradictionToKeep: 0.78,
                        ProtectedConcepts: protectedConcepts));
                    relationsPruned += maintenance.RelationsPruned;
                    nodesPruned += maintenance.NodesPruned;
                    nodesMerged += maintenance.NodesMerged;
                    break;
                }
            case SpaceToolKind.Expand:
                linksAdded += SeedExplorationLinks(protectedConcepts, Math.Max(8, _settings.MaxNodeMergesPerCycle * 2));
                break;
            case SpaceToolKind.Reinforce:
                linksAdded += ReinforceFocusNeighborhood(protectedConcepts, Math.Max(8, _settings.MaxRelationPrunesPerCycle / 4));
                break;
            case SpaceToolKind.Observe:
            default:
                break;
        }
    }

    private SpaceToolKind SelectBaselineTool(SpaceManagementResult assessment)
    {
        var relationPressure = assessment.RelationBudget > 0
            ? (double)assessment.RelationsAfter / assessment.RelationBudget
            : 0.0;

        if (assessment.NoiseRatio >= Math.Max(0.55, _settings.NoiseThreshold * 0.9))
            return SpaceToolKind.Stabilize;

        if (relationPressure >= 1.12)
            return SpaceToolKind.Rebalance;

        if (relationPressure <= 0.70 && assessment.NoiseRatio <= 0.42)
            return SpaceToolKind.Expand;

        if (assessment.NoiseRatio <= 0.50)
            return SpaceToolKind.Reinforce;

        return SpaceToolKind.Observe;
    }

    private int ComputeTargetRelationBudget(int nodes)
    {
        var dynamicBudget = (nodes * Math.Max(1, _settings.TargetRelationsPerNode)) + Math.Max(0, _settings.NodeBuffer);
        return Math.Clamp(dynamicBudget, _settings.MinRelations, _settings.MaxRelations);
    }

    private IReadOnlyCollection<string> BuildProtectedConcepts(IReadOnlyCollection<string>? focusConcepts)
    {
        var protectedConcepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (focusConcepts is not null)
        {
            foreach (var concept in focusConcepts)
            {
                if (!string.IsNullOrWhiteSpace(concept))
                    protectedConcepts.Add(concept.Trim());
            }
        }

        var snapshot = _memory.ExportSnapshot();
        var degreeByConcept = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var relation in snapshot.Relations)
        {
            degreeByConcept[relation.Left] = degreeByConcept.TryGetValue(relation.Left, out var leftDegree) ? leftDegree + 1 : 1;
            degreeByConcept[relation.Right] = degreeByConcept.TryGetValue(relation.Right, out var rightDegree) ? rightDegree + 1 : 1;

            if (relation.ObservationCount >= AnchorObservationProtectionThreshold)
            {
                protectedConcepts.Add(relation.Left);
                protectedConcepts.Add(relation.Right);
            }
        }

        foreach (var node in snapshot.Nodes)
        {
            var degree = degreeByConcept.TryGetValue(node.Name, out var value) ? value : 0;
            if (node.ObservationCount >= AnchorObservationProtectionThreshold ||
                degree >= AnchorDegreeProtectionThreshold)
            {
                protectedConcepts.Add(node.Name);
            }
        }

        return protectedConcepts.ToArray();
    }

    private int SeedExplorationLinks(IReadOnlyCollection<string> concepts, int maxLinks)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var budget = Math.Max(1, maxLinks);
        var links = 0;
        for (var i = 0; i < conceptList.Length - 1 && links < budget; i++)
        {
            if (TryObserveLink(conceptList[i], conceptList[i + 1], 0.38))
                links++;
        }

        if (links < budget && conceptList.Length >= 3)
        {
            if (TryObserveLink(conceptList[^1], conceptList[0], 0.42))
                links++;
        }

        return links;
    }

    private int ReinforceFocusNeighborhood(IReadOnlyCollection<string> concepts, int maxUpdates)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length == 0)
            return 0;

        var budget = Math.Max(1, maxUpdates);
        var updates = 0;
        foreach (var concept in conceptList)
        {
            if (updates >= budget)
                break;

            var neighbors = _memory.GetNeighbors(
                concept,
                PlatonicNeighborhoodType.Any,
                maxNeighbors: Math.Max(2, Math.Min(6, budget - updates)),
                minConfidence: 0.10);
            foreach (var neighbor in neighbors)
            {
                if (updates >= budget)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                var reinforced = Math.Max(0.0, contradiction * 0.92);
                if (TryObserveLink(concept, neighbor.Concept, reinforced))
                    updates++;
            }
        }

        if (updates == 0 && conceptList.Length >= 2)
            updates += SeedExplorationLinks(conceptList, Math.Min(2, budget));

        return updates;
    }

    private int CreateConcepts(IReadOnlyCollection<string> concepts, int maxCreates)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(Math.Max(1, maxCreates))
            .ToArray();
        var created = 0;
        foreach (var concept in conceptList)
        {
            _memory.FineEditFromExample(new[] { concept }, new[] { concept }, isNegativeExample: false);
            created++;
        }

        return created;
    }

    private int EditConceptFaces(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length == 0)
            return 0;

        var edits = 0;
        foreach (var concept in conceptList)
        {
            if (edits >= maxEdits)
                break;

            var neighbors = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Any, maxNeighbors: 4, minConfidence: 0.05);
            if (neighbors.Count == 0)
            {
                _memory.FineEditFromExample(new[] { concept }, new[] { concept }, isNegativeExample: false);
                edits++;
                continue;
            }

            foreach (var neighbor in neighbors)
            {
                if (edits >= maxEdits)
                    break;
                _memory.FineEditFromExample(new[] { concept }, new[] { neighbor.Concept }, isNegativeExample: false);
                edits++;
            }
        }

        return edits;
    }

    private int EditRelationContradictions(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var edits = 0;
        foreach (var concept in concepts)
        {
            if (edits >= maxEdits)
                break;
            var neighbors = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Any, maxNeighbors: 4, minConfidence: 0.0);
            foreach (var neighbor in neighbors)
            {
                if (edits >= maxEdits)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                var adjusted = Math.Clamp((contradiction * 0.85) + 0.075, 0.0, 1.0);
                if (TryObserveLink(concept, neighbor.Concept, adjusted))
                    edits++;
            }
        }

        return edits;
    }

    private int CreateOrStrengthenRelations(IReadOnlyCollection<string> concepts, int maxLinks)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var links = SeedExplorationLinks(conceptList, maxLinks);
        if (links >= maxLinks)
            return links;

        return links + ReinforceFocusNeighborhood(conceptList, maxLinks - links);
    }

    private int WeakenOrDecayRelations(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var edits = 0;
        foreach (var concept in concepts)
        {
            if (edits >= maxEdits)
                break;
            var neighbors = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Any, maxNeighbors: 4, minConfidence: 0.0);
            foreach (var neighbor in neighbors)
            {
                if (edits >= maxEdits)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                var decayed = Math.Clamp((contradiction * 0.92) + 0.08, 0.0, 1.0);
                if (TryObserveLink(concept, neighbor.Concept, decayed))
                    edits++;
            }
        }

        return edits;
    }

    private int TriadConsistencyEdit(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var edits = 0;
        for (var i = 0; i < conceptList.Length && edits < maxEdits; i++)
        {
            var left = conceptList[i];
            var neighbors = _memory.GetNeighbors(left, PlatonicNeighborhoodType.Any, maxNeighbors: 3, minConfidence: 0.0);
            foreach (var a in neighbors)
            {
                if (edits >= maxEdits)
                    break;
                foreach (var b in neighbors)
                {
                    if (edits >= maxEdits)
                        break;
                    if (a.Concept.Equals(b.Concept, StringComparison.OrdinalIgnoreCase))
                        continue;
                    var leftA = _memory.GetContradiction(left, a.Concept);
                    var leftB = _memory.GetContradiction(left, b.Concept);
                    var predicted = Math.Clamp(0.5 + (0.5 * Math.Abs(leftA - leftB)), 0.0, 1.0);
                    var current = _memory.GetContradiction(a.Concept, b.Concept);
                    var target = Math.Clamp((current * 0.7) + (predicted * 0.3), 0.0, 1.0);
                    if (TryObserveLink(a.Concept, b.Concept, target))
                        edits++;
                }
            }
        }

        return edits;
    }

    private int NeighborhoodRetype(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var edits = 0;
        foreach (var concept in concepts)
        {
            if (edits >= maxEdits)
                break;

            var semantic = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Semantic, maxNeighbors: 3, minConfidence: 0.0);
            foreach (var neighbor in semantic)
            {
                if (edits >= maxEdits)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                if (TryObserveLink(concept, neighbor.Concept, Math.Clamp(contradiction * 0.9, 0.0, 1.0)))
                    edits++;
            }

            var numeric = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Numeric, maxNeighbors: 2, minConfidence: 0.0);
            foreach (var neighbor in numeric)
            {
                if (edits >= maxEdits)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                if (TryObserveLink(concept, neighbor.Concept, Math.Clamp((contradiction * 0.9) + 0.07, 0.0, 1.0)))
                    edits++;
            }
        }

        return edits;
    }

    private int CentroidPullPush(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var anchor = conceptList[0];
        var edits = 0;
        foreach (var concept in conceptList.Skip(1))
        {
            if (edits >= maxEdits)
                break;
            var contradiction = _memory.GetContradiction(anchor, concept);
            _memory.FineEditFromExample(
                new[] { anchor },
                new[] { concept },
                isNegativeExample: contradiction >= 0.55);
            edits++;
        }

        return edits;
    }

    private int AnchorBindingEdit(IReadOnlyCollection<string> concepts, int maxLinks)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var anchor = conceptList[0];
        var links = 0;
        foreach (var concept in conceptList.Skip(1))
        {
            if (links >= maxLinks)
                break;
            if (TryObserveLink(anchor, concept, 0.28))
                links++;
        }

        return links;
    }

    private int AttentionScopeSelect(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var edits = 0;
        foreach (var concept in concepts)
        {
            if (edits >= maxEdits)
                break;

            var neighbors = _memory.GetNeighbors(concept, PlatonicNeighborhoodType.Any, maxNeighbors: 3, minConfidence: 0.08);
            foreach (var neighbor in neighbors)
            {
                if (edits >= maxEdits)
                    break;
                var contradiction = _memory.GetContradiction(concept, neighbor.Concept);
                var focused = Math.Clamp(contradiction * 0.9, 0.0, 1.0);
                if (TryObserveLink(concept, neighbor.Concept, focused))
                    edits++;
            }
        }

        return edits;
    }

    private int RewardTagEmit(IReadOnlyCollection<string> concepts, int maxEdits)
    {
        var conceptList = concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (conceptList.Length < 2)
            return 0;

        var edits = 0;
        for (var i = 0; i < conceptList.Length - 1 && edits < maxEdits; i++)
        {
            var contradiction = _memory.GetContradiction(conceptList[i], conceptList[i + 1]);
            var rewarded = Math.Clamp(contradiction * 0.85, 0.0, 1.0);
            if (TryObserveLink(conceptList[i], conceptList[i + 1], rewarded))
                edits++;
        }

        return edits;
    }

    private int DiscoverAbstractions(IReadOnlyCollection<string> focusConcepts)
    {
        var snapshot = _memory.ExportSnapshot();
        if (snapshot.Nodes.Length < 3)
            return 0;

        var nodesByName = snapshot.Nodes.ToDictionary(n => n.Name, StringComparer.OrdinalIgnoreCase);
        var existingNames = new HashSet<string>(nodesByName.Keys, StringComparer.OrdinalIgnoreCase);
        var relationKeys = snapshot.Relations
            .Select(r => RelationKey(r.Left, r.Right))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var focus = focusConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var candidates = snapshot.Relations
            .Where(r => r.SynthesisContradiction <= 0.32 && r.ObservationCount >= 2)
            .Where(r => nodesByName.ContainsKey(r.Left) && nodesByName.ContainsKey(r.Right))
            .Select(r => new
            {
                Relation = r,
                Distance = FaceDistance(nodesByName[r.Left].PositiveFace, nodesByName[r.Right].PositiveFace),
                FocusBoost = focus.Contains(r.Left) || focus.Contains(r.Right) ? 1.0 : 0.0
            })
            .Where(x => x.Distance <= 0.45)
            .OrderByDescending(x => x.FocusBoost)
            .ThenByDescending(x => x.Relation.ObservationCount)
            .ThenBy(x => x.Relation.SynthesisContradiction)
            .ThenBy(x => x.Distance)
            .Take(24)
            .ToArray();
        if (candidates.Length == 0)
            return 0;

        var newNodes = new List<PlatonicNodeSnapshot>();
        var newRelations = new List<PlatonicRelationSnapshot>();
        var consumed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var candidate in candidates)
        {
            if (newNodes.Count >= 2)
                break;
            if (consumed.Contains(candidate.Relation.Left) || consumed.Contains(candidate.Relation.Right))
                continue;

            var members = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                candidate.Relation.Left,
                candidate.Relation.Right
            };
            foreach (var relation in snapshot.Relations
                .Where(r => r.SynthesisContradiction <= 0.36 && r.ObservationCount >= 1)
                .OrderByDescending(r => r.ObservationCount))
            {
                if (members.Count >= 5)
                    break;
                var connectsLeft = members.Contains(relation.Left) && nodesByName.ContainsKey(relation.Right);
                var connectsRight = members.Contains(relation.Right) && nodesByName.ContainsKey(relation.Left);
                var next = connectsLeft ? relation.Right : connectsRight ? relation.Left : null;
                if (next is null || members.Contains(next))
                    continue;
                if (members.All(m => _memory.GetContradiction(m, next) <= 0.42))
                    members.Add(next);
            }

            if (members.Count < 3)
                continue;

            var seed = members
                .OrderByDescending(m => nodesByName[m].ObservationCount)
                .ThenBy(m => m, StringComparer.OrdinalIgnoreCase)
                .First();
            var abstractName = $"abstract:{seed}";
            if (existingNames.Contains(abstractName))
                continue;

            var centroid = ComputeCentroid(members.Select(m => nodesByName[m].PositiveFace).ToArray(), snapshot.FaceDimension);
            var negative = centroid.Select(v => -v).ToArray();
            newNodes.Add(new PlatonicNodeSnapshot(
                abstractName,
                centroid,
                negative,
                ObservationCount: 1,
                UseCount: 1,
                SuccessCount: 1,
                FailureCount: 0,
                LastUsedStep: 0));
            existingNames.Add(abstractName);

            foreach (var member in members.OrderBy(m => m, StringComparer.OrdinalIgnoreCase))
            {
                var key = RelationKey(abstractName, member);
                if (relationKeys.Add(key))
                {
                    newRelations.Add(new PlatonicRelationSnapshot(
                        abstractName,
                        member,
                        ThesisContradiction: 0.24,
                        LastObservedContradiction: 0.24,
                        SynthesisContradiction: 0.24,
                        ObservationCount: 1,
                        UseCount: 1,
                        SuccessCount: 1,
                        FailureCount: 0,
                        LastUsedStep: 0));
                }
                consumed.Add(member);
            }
        }

        if (newNodes.Count == 0)
            return 0;

        _memory.ImportSnapshot(new PlatonicMemorySnapshot(
            snapshot.FaceDimension,
            snapshot.Nodes.Concat(newNodes).ToArray(),
            snapshot.Relations.Concat(newRelations).ToArray()));
        return newRelations.Count;
    }

    private bool TryObserveLink(string left, string right, double contradiction)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            return false;
        if (left.Equals(right, StringComparison.OrdinalIgnoreCase))
            return false;
        _memory.ObserveContradiction(left, right, contradiction);
        return true;
    }

    private static double[] ComputeCentroid(IReadOnlyList<double[]> faces, int dimension)
    {
        var centroid = new double[dimension];
        if (faces.Count == 0)
            return centroid;
        foreach (var face in faces)
        {
            for (var i = 0; i < Math.Min(dimension, face.Length); i++)
                centroid[i] += face[i];
        }
        for (var i = 0; i < centroid.Length; i++)
            centroid[i] /= faces.Count;
        return centroid;
    }

    private static double FaceDistance(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var length = Math.Min(left.Count, right.Count);
        if (length == 0)
            return double.MaxValue;
        var sum = 0.0;
        for (var i = 0; i < length; i++)
        {
            var d = left[i] - right[i];
            sum += d * d;
        }
        return Math.Sqrt(sum / length);
    }

    private static string RelationKey(string left, string right)
        => string.Compare(left, right, StringComparison.OrdinalIgnoreCase) <= 0
            ? $"{left}|{right}"
            : $"{right}|{left}";

}
