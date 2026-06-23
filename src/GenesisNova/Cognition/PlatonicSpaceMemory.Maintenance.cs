namespace GenesisNova.Cognition;

public sealed partial class PlatonicSpaceMemory
{
    public SpaceMaintenanceResult ApplyMaintenance(SpaceMaintenanceRequest request)
    {
        var protectedConcepts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (request.ProtectedConcepts != null)
        {
            foreach (var concept in request.ProtectedConcepts)
            {
                if (!string.IsNullOrWhiteSpace(concept))
                    protectedConcepts.Add(Normalize(concept));
            }
        }

        var nodes = _nodes.Values
            .Select(n => new MutableNode(
                n.Name,
                n.PositiveFace.ToArray(),
                n.NegativeFace.ToArray(),
                n.ObservationCount,
                n.UseCount,
                n.SuccessCount,
                n.FailureCount,
                n.LastUsedStep))
            .ToDictionary(n => n.Name, StringComparer.OrdinalIgnoreCase);
        var relations = _relationIndex.Values
            .Select(r => new MutableRelation(
                r.Left,
                r.Right,
                r.ThesisContradiction,
                r.LastObservedContradiction,
                r.SynthesisContradiction,
                r.ObservationCount,
                r.UseCount,
                r.SuccessCount,
                r.FailureCount,
                r.LastUsedStep))
            .ToDictionary(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase);

        var minRelationsToKeep = Math.Max(0, request.MinRelationsToKeep);
        var minNodesToKeep = Math.Max(0, request.MinNodesToKeep);
        var relationPruneBudget = Math.Max(0, request.MaxRelationPrunes);
        var rebalanceBudget = Math.Max(0, request.MaxRebalancePrunes);
        var nodePruneBudget = Math.Max(0, request.MaxNodePrunes);
        var nodeMergeBudget = Math.Max(0, request.MaxNodeMerges);
        var minRelObs = Math.Max(0, request.MinRelationObservationToKeep);
        var minNodeObs = Math.Max(0, request.MinNodeObservationToKeep);
        var maxSynthesis = Clamp01(request.MaxSynthesisContradictionToKeep);

        var relationsPruned = 0;
        var nodesPruned = 0;
        var nodesMerged = 0;

        var staleRelations = relations.Values
            .Where(r =>
            {
                if (protectedConcepts.Contains(r.Left) || protectedConcepts.Contains(r.Right))
                    return false;
                if (RelationUtilityScore(r) > 0.45)
                    return false;
                if (r.ObservationCount > minRelObs && RelationUtilityScore(r) >= 0.15)
                    return false;
                if (r.SynthesisContradiction < maxSynthesis && RelationUtilityScore(r) >= 0.05)
                    return false;
                var leftObs = nodes.TryGetValue(r.Left, out var leftNode) ? leftNode.ObservationCount : 0;
                var rightObs = nodes.TryGetValue(r.Right, out var rightNode) ? rightNode.ObservationCount : 0;
                return leftObs <= (minNodeObs + 1) && rightObs <= (minNodeObs + 1);
            })
            .OrderBy(r => RelationUtilityScore(r))
            .ThenBy(r => r.ObservationCount)
            .ThenByDescending(r => r.SynthesisContradiction)
            .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var relation in staleRelations)
        {
            if (relationPruneBudget <= 0 || relations.Count <= minRelationsToKeep)
                break;
            if (relations.Remove(RelationKey(relation.Left, relation.Right)))
            {
                relationPruneBudget--;
                relationsPruned++;
            }
        }

        if (nodeMergeBudget > 0 && nodes.Count > minNodesToKeep)
        {
            var mergeCandidates = nodes.Values
                .Where(n => !protectedConcepts.Contains(n.Name) && NodeUtilityScore(n) < 0.55)
                .OrderBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            for (var i = 0; i < mergeCandidates.Length && nodeMergeBudget > 0; i++)
            {
                var sourceName = mergeCandidates[i].Name;
                if (!nodes.ContainsKey(sourceName))
                    continue;

                for (var j = i + 1; j < mergeCandidates.Length && nodeMergeBudget > 0; j++)
                {
                    if (!nodes.TryGetValue(sourceName, out var source))
                        break;
                    if (!nodes.TryGetValue(mergeCandidates[j].Name, out var target))
                        continue;
                    if (source.Name.Equals(target.Name, StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (protectedConcepts.Contains(source.Name) || protectedConcepts.Contains(target.Name))
                        continue;

                    var sourceDegree = CountRelationsForNode(relations.Values, source.Name);
                    var targetDegree = CountRelationsForNode(relations.Values, target.Name);
                    if (sourceDegree > 1 || targetDegree > 1)
                        continue;

                    var distance = FaceDistance(source.PositiveFace, target.PositiveFace);
                    if (distance > request.MergeDistanceThreshold)
                        continue;

                    var sourceValue = source.ObservationCount + NodeUtilityScore(source);
                    var targetValue = target.ObservationCount + NodeUtilityScore(target);
                    var canonical = sourceValue >= targetValue ? source : target;
                    var duplicate = ReferenceEquals(canonical, source) ? target : source;
                    MergeNodeInto(canonical, duplicate, nodes, relations);
                    nodesMerged++;
                    nodeMergeBudget--;
                }
            }
        }

        var maxRelationsPerNode = Math.Max(0, request.MaxRelationsPerNode);
        var targetRelationCount = Math.Max(0, request.TargetRelationCount);
        if (maxRelationsPerNode > 0 && rebalanceBudget > 0 && relations.Count > targetRelationCount)
        {
            var nodeOrder = nodes.Keys
                .OrderBy(k => k, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            foreach (var nodeName in nodeOrder)
            {
                if (rebalanceBudget <= 0 || relations.Count <= Math.Max(minRelationsToKeep, targetRelationCount))
                    break;
                if (protectedConcepts.Contains(nodeName))
                    continue;

                var incident = relations.Values
                    .Where(r => r.Left.Equals(nodeName, StringComparison.OrdinalIgnoreCase) ||
                                r.Right.Equals(nodeName, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(r => RelationUtilityScore(r))
                    .ThenBy(r => r.ObservationCount)
                    .ThenByDescending(r => r.SynthesisContradiction)
                    .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                var over = incident.Length - maxRelationsPerNode;
                if (over <= 0)
                    continue;

                foreach (var relation in incident)
                {
                    if (over <= 0 || rebalanceBudget <= 0 || relations.Count <= Math.Max(minRelationsToKeep, targetRelationCount))
                        break;
                    if (protectedConcepts.Contains(relation.Left) || protectedConcepts.Contains(relation.Right))
                        continue;
                    if (RelationUtilityScore(relation) > 0.45)
                        continue;
                    if (relation.ObservationCount > (minRelObs + 1) && relation.SynthesisContradiction < (maxSynthesis + 0.1) && RelationUtilityScore(relation) >= 0.15)
                        continue;

                    if (relations.Remove(RelationKey(relation.Left, relation.Right)))
                    {
                        over--;
                        rebalanceBudget--;
                        relationsPruned++;
                    }
                }
            }
        }

        if (nodePruneBudget > 0 && nodes.Count > minNodesToKeep)
        {
            var relationEndpoints = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var relation in relations.Values)
            {
                relationEndpoints.Add(relation.Left);
                relationEndpoints.Add(relation.Right);
            }

            var staleNodes = nodes.Values
                .Where(n =>
                    !protectedConcepts.Contains(n.Name) &&
                    n.ObservationCount <= minNodeObs &&
                    NodeUtilityScore(n) < 0.35 &&
                    !relationEndpoints.Contains(n.Name))
                .OrderBy(n => NodeUtilityScore(n))
                .ThenBy(n => n.ObservationCount)
                .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            foreach (var node in staleNodes)
            {
                if (nodePruneBudget <= 0 || nodes.Count <= minNodesToKeep)
                    break;
                if (nodes.Remove(node.Name))
                {
                    nodePruneBudget--;
                    nodesPruned++;
                }
            }
        }

        if (relationsPruned == 0 && nodesPruned == 0 && nodesMerged == 0)
            return new SpaceMaintenanceResult(0, 0, 0);

        ImportSnapshot(new PlatonicMemorySnapshot(
            FaceDimension: _faceDimension,
            Nodes: nodes.Values
                .OrderBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .Select(n => new PlatonicNodeSnapshot(
                    n.Name,
                    n.PositiveFace,
                    n.NegativeFace,
                    Math.Max(0, n.ObservationCount),
                    Math.Max(0, n.UseCount),
                    Math.Max(0, n.SuccessCount),
                    Math.Max(0, n.FailureCount),
                    Math.Max(0, n.LastUsedStep)))
                .ToArray(),
            Relations: relations.Values
                .OrderBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
                .Select(r => new PlatonicRelationSnapshot(
                    r.Left,
                    r.Right,
                    Clamp01(r.ThesisContradiction),
                    Clamp01(r.LastObservedContradiction),
                    Clamp01(r.SynthesisContradiction),
                    Math.Max(0, r.ObservationCount),
                    Math.Max(0, r.UseCount),
                    Math.Max(0, r.SuccessCount),
                    Math.Max(0, r.FailureCount),
                    Math.Max(0, r.LastUsedStep)))
                .ToArray()));

        return new SpaceMaintenanceResult(
            RelationsPruned: relationsPruned,
            NodesPruned: nodesPruned,
            NodesMerged: nodesMerged);
    }

    private ConceptNode GetOrCreate(string concept, IReadOnlyCollection<string>? protectedConcepts = null)
    {
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
            return node;

        // G6 (Irreversibility): a re-observed concept that was archived is REACTIVATED with its learned face intact
        // — the distinction was only made dormant, never unmade — instead of being recreated from a fresh seed.
        if (_archivedNodes.TryGetValue(key, out var archived))
        {
            _archivedNodes.Remove(key);
            _nodes[key] = archived;
            _lattice.RegisterNode(key);
            return archived;
        }

        EnsureNodeCapacity(protectedConcepts is null
            ? new[] { key }
            : protectedConcepts.Append(key).ToArray());
        var positiveFace = TryCreateSeededFace(key, out var seeded)
            ? seeded
            : CreateFace(key);
        // SPAWN SPREAD: non-numeric elements are born on the UNIT SPHERE of their FREE region (a random
        // direction from the seed noise), so in ~800 free dims any two fresh elements are near-orthogonal and
        // kNN can separate them from birth — we do NOT want everything clumped at the word-face origin waiting
        // for attraction to drag it out. Attraction then pulls RELATED elements in (it just needs enough
        // observations to converge from the spread start); repulsion keeps unrelated apart. Identity dims (the
        // numeric homomorphism, the frozen char spelling) are left exact.
        if (!IsFrozenConcept(key))
            NormaliseFreeRegion(positiveFace, key);
        node = new ConceptNode(
            name: key,
            positiveFace: positiveFace,
            negativeFace: positiveFace.Select(x => -x).ToArray(),
            observationCount: 0,
            useCount: 0,
            successCount: 0,
            failureCount: 0,
            lastUsedStep: 0);
        _nodes[key] = node;
        _lattice.RegisterNode(key);
        return node;
    }

    private double NodeUtilityScore(ConceptNode node)
        => UtilityScore(node.UseCount, node.SuccessCount, node.FailureCount, node.LastUsedStep);

    private double NodeUtilityScore(MutableNode node)
        => UtilityScore(node.UseCount, node.SuccessCount, node.FailureCount, node.LastUsedStep);

    private double RelationUtilityScore(RelationElementNode relation)
        => UtilityScore(relation.UseCount, relation.SuccessCount, relation.FailureCount, relation.LastUsedStep);

    private double RelationUtilityScore(MutableRelation relation)
        => UtilityScore(relation.UseCount, relation.SuccessCount, relation.FailureCount, relation.LastUsedStep);

    private double UtilityScore(int useCount, int successCount, int failureCount, long lastUsedStep)
    {
        if (useCount <= 0)
            return 0.0;
        var successRatio = (successCount + 0.5) / Math.Max(1.0, successCount + failureCount + 1.0);
        var useSignal = Math.Min(1.0, Math.Log(1.0 + useCount) / Math.Log(17.0));
        var recency = lastUsedStep <= 0 ? 0.0 : 1.0 / (1.0 + Math.Max(0, _utilityStep - lastUsedStep) / 512.0);
        var failurePenalty = failureCount > successCount ? Math.Min(0.4, (failureCount - successCount) / Math.Max(1.0, useCount) * 0.5) : 0.0;
        return Math.Clamp((0.55 * successRatio) + (0.25 * useSignal) + (0.20 * recency) - failurePenalty, 0.0, 1.0);
    }

    private void EnsureNodeCapacity(IReadOnlyCollection<string> protectedConcepts)
    {
        if (_nodes.Count < _maxPlatonicNodes)
            return;

        var protectedSet = protectedConcepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(Normalize)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var candidate = _nodes.Values
            .Where(n => !protectedSet.Contains(n.Name))
            .OrderBy(n => NodeUtilityScore(n))
            .ThenBy(n => n.ObservationCount)
            .ThenBy(n => GetRelationDegree(n.Name))
            .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault()
            ?? _nodes.Values
                .OrderBy(n => NodeUtilityScore(n))
                .ThenBy(n => n.ObservationCount)
                .ThenBy(n => GetRelationDegree(n.Name))
                .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();
        if (candidate is not null)
            RemoveNode(candidate.Name);
    }

    private void EnsureRelationCapacity()
    {
        if (_relationIndex.Count < _maxPlatonicRelations)
            return;

        var candidate = _relationIndex.Values
            .OrderBy(r => RelationUtilityScore(r))
            .ThenBy(r => r.ObservationCount)
            .ThenByDescending(r => r.SynthesisContradiction)
            .ThenBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
        if (candidate is not null)
            RemoveRelation(candidate);
    }

    // G6 (Irreversibility): the space only EXPANDS — a distinction is never destroyed, only made DORMANT. The node
    // leaves the active lattice/geometry (so the active space stays bounded) but is ARCHIVED with its learned face
    // intact, ready to be reactivated by GetOrCreate if re-observed. (Was: _nodes.Remove → permanent forgetting.)
    private void RemoveNode(string concept)
    {
        var key = Normalize(concept);
        var incident = _relationIndex.Values
            .Where(r => r.Left.Equals(key, StringComparison.OrdinalIgnoreCase) ||
                        r.Right.Equals(key, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        foreach (var relation in incident)
            RemoveRelation(relation);

        if (_nodes.TryGetValue(key, out var node))
            _archivedNodes[key] = node;        // archive (dormant), never destroy
        _nodes.Remove(key);
        _lattice.UnregisterNode(key);
    }

    // G6: archive the relation rather than destroying it; ObserveContradiction reactivates it (with its learned
    // synthesis-contradiction intact) if the pair is observed again.
    private void RemoveRelation(RelationElementNode relation)
    {
        var key = RelationKey(relation.Left, relation.Right);
        _archivedRelations[key] = relation;    // archive (dormant), never destroy
        _relationIndex.Remove(key);
        _lattice.RemoveEdge(relation.Left, relation.Right);
    }

    private static int CountRelationsForNode(IEnumerable<MutableRelation> relations, string nodeName)
        => relations.Count(r =>
            r.Left.Equals(nodeName, StringComparison.OrdinalIgnoreCase) ||
            r.Right.Equals(nodeName, StringComparison.OrdinalIgnoreCase));

    private static void MergeNodeInto(
        MutableNode canonical,
        MutableNode duplicate,
        IDictionary<string, MutableNode> nodes,
        IDictionary<string, MutableRelation> relations)
    {
        var totalObs = Math.Max(1, canonical.ObservationCount + duplicate.ObservationCount);
        var canonicalWeight = canonical.ObservationCount / (double)totalObs;
        var duplicateWeight = duplicate.ObservationCount / (double)totalObs;
        for (var i = 0; i < canonical.PositiveFace.Length; i++)
        {
            canonical.PositiveFace[i] = (canonical.PositiveFace[i] * canonicalWeight) + (duplicate.PositiveFace[i] * duplicateWeight);
            canonical.NegativeFace[i] = (canonical.NegativeFace[i] * canonicalWeight) + (duplicate.NegativeFace[i] * duplicateWeight);
        }
        canonical.ObservationCount += duplicate.ObservationCount;
        canonical.UseCount += duplicate.UseCount;
        canonical.SuccessCount += duplicate.SuccessCount;
        canonical.FailureCount += duplicate.FailureCount;
        canonical.LastUsedStep = Math.Max(canonical.LastUsedStep, duplicate.LastUsedStep);

        var incidentRelations = relations.Values
            .Where(r => r.Left.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase) ||
                        r.Right.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase))
            .OrderBy(r => RelationKey(r.Left, r.Right), StringComparer.OrdinalIgnoreCase)
            .ToArray();

        foreach (var relation in incidentRelations)
        {
            relations.Remove(RelationKey(relation.Left, relation.Right));
            var remappedLeft = relation.Left.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase)
                ? canonical.Name
                : relation.Left;
            var remappedRight = relation.Right.Equals(duplicate.Name, StringComparison.OrdinalIgnoreCase)
                ? canonical.Name
                : relation.Right;
            if (remappedLeft.Equals(remappedRight, StringComparison.OrdinalIgnoreCase))
                continue;

            var remappedKey = RelationKey(remappedLeft, remappedRight);
            if (relations.TryGetValue(remappedKey, out var existing))
            {
                var combinedObs = Math.Max(1, existing.ObservationCount + relation.ObservationCount);
                existing.ThesisContradiction = Clamp01(
                    ((existing.ThesisContradiction * existing.ObservationCount) + (relation.ThesisContradiction * relation.ObservationCount)) / combinedObs);
                existing.LastObservedContradiction = Clamp01((existing.LastObservedContradiction + relation.LastObservedContradiction) * 0.5);
                existing.SynthesisContradiction = Clamp01(
                    ((existing.SynthesisContradiction * existing.ObservationCount) + (relation.SynthesisContradiction * relation.ObservationCount)) / combinedObs);
                existing.ObservationCount = combinedObs;
                existing.UseCount += relation.UseCount;
                existing.SuccessCount += relation.SuccessCount;
                existing.FailureCount += relation.FailureCount;
                existing.LastUsedStep = Math.Max(existing.LastUsedStep, relation.LastUsedStep);
            }
            else
            {
                relation.Left = remappedLeft;
                relation.Right = remappedRight;
                relations[remappedKey] = relation;
            }
        }

        nodes.Remove(duplicate.Name);
    }
}
