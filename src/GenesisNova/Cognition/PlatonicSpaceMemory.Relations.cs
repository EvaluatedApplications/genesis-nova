namespace GenesisNova.Cognition;

public sealed partial class PlatonicSpaceMemory
{
    public void ObserveContradiction(string left, string right, double observedContradiction)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
            return;
        if (left.Equals(right, StringComparison.OrdinalIgnoreCase))
            return;

        // OP-TOKEN COUPLING GUARD — the single choke point for relation formation. An operation/framing token
        // ("of"/"plus"/"add"/"is"…) may couple ONLY to a reserved face marker (its op→face routing affinity,
        // e.g. add↔face:poly); it must NEVER couple to a CONTENT concept. Guarding HERE blocks framing-word
        // mega-hubs (of↔pear, plus↔+, is↔greater) from EVERY caller — the per-caller mirror filter missed the
        // learning-enabled preview/concept-plan generations, which leaked these edges. See
        // [[nova-find-hub-collapse-fix]]. (Returning before GetOrCreate also avoids minting the framing-word node.)
        if ((IsOperationToken(left) && !IsReservedConcept(right)) ||
            (IsOperationToken(right) && !IsReservedConcept(left)))
            return;

        var a = GetOrCreate(left, new[] { right });
        var b = GetOrCreate(right, new[] { left });
        a.ObservationCount++;
        b.ObservationCount++;

        var key = RelationKey(left, right);
        if (!_relationIndex.TryGetValue(key, out var relation))
        {
            if (_archivedRelations.TryGetValue(key, out var dormant))
            {
                // G6 (Irreversibility): re-observing an archived pair REACTIVATES the dormant relation with its
                // learned synthesis-contradiction intact, rather than relearning from scratch.
                _archivedRelations.Remove(key);
                relation = dormant;
                _relationIndex[key] = relation;
                IndexRelation(relation);
            }
            else
            {
                EnsureRelationCapacity();
                relation = new RelationElementNode(
                    left: Normalize(left),
                    right: Normalize(right),
                    thesisContradiction: observedContradiction,
                    lastObservedContradiction: observedContradiction,
                    synthesisContradiction: observedContradiction,
                    observationCount: 0,
                    useCount: 0,
                    successCount: 0,
                    failureCount: 0,
                    lastUsedStep: 0);
                _relationIndex[key] = relation;
                IndexRelation(relation);
            }
        }

        relation.LastObservedContradiction = Clamp01(observedContradiction);
        relation.SynthesisContradiction = relation.ObservationCount == 0
            ? relation.LastObservedContradiction
            : (0.85 * relation.SynthesisContradiction) + (0.15 * relation.LastObservedContradiction);
        relation.ObservationCount++;

        // Academic update: mutate ONLY the directly-observed pair. No triadic synthesis — pulling
        // unrelated third concepts toward this pair is cross-contamination, not signal.
        UpdateConceptGeometry(a, b, relation.SynthesisContradiction, 1.0);

        // Periodic contrastive repulsion keeps unrelated concepts discriminable (genesis parity).
        if (++_observationsSinceRepulsion >= RepulsionInterval)
        {
            _observationsSinceRepulsion = 0;
            ApplyContrastiveRepulsionPass();
        }
    }

    public double GetContradiction(string left, string right)
    {
        var key = RelationKey(left, right);
        return _relationIndex.TryGetValue(key, out var relation)
            ? relation.SynthesisContradiction
            : 0.5;
    }

    /// <summary>
    /// Projects the relation index into its canonical positioned relation-ELEMENTS (see
    /// <see cref="RelationElement"/>): each relation is an element located at the CENTROID of its endpoints'
    /// faces, strength = 1 − synthesis contradiction. This is the substrate-native form — relations are
    /// positioned objects that can themselves be related/composed (higher-order); the keyed index simply
    /// provides O(1) access to them (the same role <c>_nodes</c> plays for concept-elements).
    /// </summary>
    public IReadOnlyList<RelationElement> GetRelationElements()
    {
        var result = new List<RelationElement>(_relationIndex.Count);
        foreach (var rel in _relationIndex.Values)
        {
            var embedding = Array.Empty<double>();
            if (TryGetConceptFace(rel.Left, out var lf) && TryGetConceptFace(rel.Right, out var rf)
                && lf.Length == rf.Length && lf.Length > 0)
            {
                embedding = new double[lf.Length];
                for (var i = 0; i < lf.Length; i++)
                    embedding[i] = 0.5 * (lf[i] + rf[i]); // centroid: the relation sits between its endpoints
            }
            result.Add(new RelationElement(rel.Left, rel.Right, rel.Strength, embedding));
        }
        return result;
    }

    /// <summary>
    /// Strongest related neighbour of a concept by TRAVERSING relation-elements — the canonical
    /// element-native retrieval. Returns the other endpoint of the highest-strength relation-element
    /// referencing the concept.
    /// </summary>
    public bool TryRelationElementNeighbour(string concept, out string neighbour, out double strength)
    {
        neighbour = string.Empty;
        strength = 0.0;
        if (string.IsNullOrWhiteSpace(concept))
            return false;
        var key = Normalize(concept);
        var found = false;
        foreach (var re in GetRelationElements())
        {
            string? other = null;
            if (string.Equals(re.Left, key, StringComparison.OrdinalIgnoreCase)) other = re.Right;
            else if (string.Equals(re.Right, key, StringComparison.OrdinalIgnoreCase)) other = re.Left;
            if (other is null) continue;
            if (!found || re.Strength > strength)
            {
                neighbour = other;
                strength = re.Strength;
                found = true;
            }
        }
        return found;
    }

    public PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops = 2,
        int beamWidth = 2)
        => QueryConceptChain(anchorConcepts, maxHops, beamWidth, out _);

    public PlatonicQueryResult QueryConceptChain(
        IReadOnlyList<string> anchorConcepts,
        int maxHops,
        int beamWidth,
        out IReadOnlyList<PlatonicEvidence> evidence)
    {
        evidence = Array.Empty<PlatonicEvidence>();
        var anchors = NormalizeConcepts(anchorConcepts)
            .Where(ContainsConcept)
            .ToArray();
        if (anchors.Length == 0)
            return new PlatonicQueryResult(string.Empty, 0.0, 0, 0);

        var hops = Math.Clamp(maxHops, 1, 6);
        var beam = Math.Clamp(beamWidth, 1, 4);
        var seen = new HashSet<string>(anchors, StringComparer.OrdinalIgnoreCase);
        var frontier = anchors;
        var decoded = new List<string>();
        var confidences = new List<double>();
        var evidenceTrail = new List<PlatonicEvidence>();

        for (var hop = 0; hop < hops; hop++)
        {
            var candidateScores = new Dictionary<string, List<double>>(StringComparer.OrdinalIgnoreCase);
            var candidateEvidence = new Dictionary<string, List<PlatonicEvidence>>(StringComparer.OrdinalIgnoreCase);
            foreach (var source in frontier)
            {
                // Relational-first retrieval: LEARNED relations are observed facts ("3" relates to
                // "three"); numeric value-proximity ("3" is near "4") and face KNN are geometry, not
                // relatedness — and for numbers they always win on full-vector distance (poly/log face
                // clusters values), returning the adjacent number instead of the related concept. So
                // prefer the relational tier; fall back to Any (semantic/numeric) only when the concept
                // has NO learned relations. General: 3→three, him→paul, apple→fruit.
                var neighbors = GetNeighbors(
                    source,
                    type: PlatonicNeighborhoodType.Relational,
                    maxNeighbors: Math.Max(8, beam * 8),
                    minConfidence: 0.35);
                if (neighbors.Count == 0)
                    neighbors = GetNeighbors(
                        source,
                        type: PlatonicNeighborhoodType.Any,
                        maxNeighbors: Math.Max(8, beam * 8),
                        minConfidence: 0.35);

                foreach (var neighbor in neighbors)
                {
                    var target = neighbor.Concept;
                    if (seen.Contains(target))
                        continue;
                    var confidence = neighbor.Confidence;

                    if (!candidateScores.TryGetValue(target, out var list))
                    {
                        list = new List<double>();
                        candidateScores[target] = list;
                    }
                    list.Add(confidence);
                    if (!candidateEvidence.TryGetValue(target, out var evList))
                    {
                       evList = new List<PlatonicEvidence>();
                       candidateEvidence[target] = evList;
                    }
                    evList.Add(new PlatonicEvidence(target, source, confidence / (hop + 1.0), hop + 1));
                }
            }

            if (candidateScores.Count == 0)
                break;

            var selected = candidateScores
                .Select(kvp => new
                {
                    Concept = kvp.Key,
                    Score = kvp.Value.Average()
                })
                .OrderByDescending(x => x.Score)
                .Take(beam)
                .ToArray();

            if (selected.Length == 0)
                break;

            foreach (var item in selected)
            {
                seen.Add(item.Concept);
                decoded.Add(item.Concept);
                confidences.Add(item.Score);
                if (candidateEvidence.TryGetValue(item.Concept, out var evList) && evList.Count > 0)
                {
                    evidenceTrail.AddRange(evList
                        .OrderByDescending(e => Math.Abs(e.Contribution))
                        .Take(2));
                }
            }

            frontier = selected.Select(x => x.Concept).ToArray();
        }

        if (decoded.Count == 0)
        {
            var first = anchors[0];
            evidence = new[] { new PlatonicEvidence(first, null, 0.42, 1) };
            return new PlatonicQueryResult(
                Text: first,
                Confidence: 0.42,
                Hops: 1,
                ConceptCount: 1);
        }

        evidence = evidenceTrail
            .GroupBy(e => (Concept: Normalize(e.Concept), Related: e.RelatedConcept is null ? null : Normalize(e.RelatedConcept), e.Hop))
            .Select(g => new PlatonicEvidence(g.Key.Concept, g.Key.Related, g.Sum(e => e.Contribution), g.Key.Hop))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(32)
            .ToArray();
        return new PlatonicQueryResult(
            Text: string.Join(' ', decoded),
            Confidence: Clamp01(confidences.DefaultIfEmpty(0.0).Average()),
            Hops: Math.Min(hops, Math.Max(1, decoded.Count)),
            ConceptCount: decoded.Count);
    }

    public IReadOnlyList<PlatonicNeighbor> GetNeighbors(
        string concept,
        PlatonicNeighborhoodType type = PlatonicNeighborhoodType.Any,
        int maxNeighbors = 16,
        double minConfidence = 0.0)
    {
        if (string.IsNullOrWhiteSpace(concept))
            return Array.Empty<PlatonicNeighbor>();

        var key = Normalize(concept);
        var capped = Math.Clamp(maxNeighbors, 1, 256);
        var threshold = Clamp01(minConfidence);
        var collected = new Dictionary<string, PlatonicNeighbor>(StringComparer.OrdinalIgnoreCase);

        void Consider(PlatonicNeighbor candidate)
        {
            if (candidate.Confidence < threshold)
                return;
            if (string.Equals(candidate.Concept, key, StringComparison.OrdinalIgnoreCase))
                return;
            if (IsReservedConcept(candidate.Concept)) // internal op→face markers (face:poly/log) are never neighbours/answers
                return;
            if (!collected.TryGetValue(candidate.Concept, out var existing) || candidate.Confidence > existing.Confidence)
                collected[candidate.Concept] = candidate;
        }

        var wantRelational = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Relational;
        var wantNumeric = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Numeric;
        var wantSemantic = type is PlatonicNeighborhoodType.Any or PlatonicNeighborhoodType.Semantic;

        // Relational tier: traverse the concept's RELATION-ELEMENTS — the adjacency index gives O(1) access
        // to them, and each element's Strength (1 − synthesis contradiction) is the confidence. This is the
        // element-native retrieval; the index is just the access path, not a separate graph.
        if (wantRelational)
        {
            foreach (var neighbor in _lattice.GetRelationalNeighbors(key))
            {
                if (!_relationIndex.TryGetValue(RelationKey(key, neighbor), out var relation))
                    continue;
                Consider(new PlatonicNeighbor(
                    Concept: neighbor,
                    // RETRIEVAL is weighted by the edge's ANSWER TRACK RECORD: an edge that has produced WRONG
                    // answers loses confidence (so a framing-word hub stops winning), proven edges keep full
                    // strength, untested edges are neutral (no cold-start penalty). Credit flows in via
                    // ReinforceEvidence on each graded example.
                    Confidence: relation.Strength * RelationTrackRecordWeight(relation),
                    Type: ClassifyConceptType(neighbor),
                    ObservationCount: relation.ObservationCount));
            }
        }

        // Numeric tier: value-proximity neighbours ("position IS address"); confidence from distance.
        if (wantNumeric && TryParseNumber(key, out var numericValue))
        {
            foreach (var (name, distance) in _lattice.GetNumericNeighbors(numericValue, range: capped, k: capped))
                Consider(new PlatonicNeighbor(
                    Concept: name,
                    Confidence: DistanceConfidence(distance),
                    Type: PlatonicNeighborhoodType.Numeric,
                    ObservationCount: NodeObservation(name)));
        }

        // Semantic tier: embedding KNN via VP-Tree; confidence from face distance.
        if (wantSemantic && TryGetConceptFace(key, out var face) && face.Length > 0)
        {
            foreach (var (name, distance) in _lattice.GetSemanticNeighbors(face, capped, key))
                Consider(new PlatonicNeighbor(
                    Concept: name,
                    Confidence: DistanceConfidence(distance),
                    Type: PlatonicNeighborhoodType.Semantic,
                    ObservationCount: NodeObservation(name)));
        }

        if (collected.Count == 0)
            return Array.Empty<PlatonicNeighbor>();

        return collected.Values
            .OrderByDescending(n => n.Confidence)
            .ThenBy(n => n.Concept, StringComparer.OrdinalIgnoreCase)
            .Take(capped)
            .ToArray();
    }

    // Map a non-negative face/value distance to a confidence in (0, 1]; nearer ⇒ more confident.
    private static double DistanceConfidence(double distance) => 1.0 / (1.0 + Math.Max(0.0, distance));

    private int NodeObservation(string concept)
        => _nodes.TryGetValue(Normalize(concept), out var node) ? node.ObservationCount : 0;

    public IReadOnlyList<(string Left, string Right, long ObservationCount)> GetAllRelations()
    {
        var result = new List<(string Left, string Right, long ObservationCount)>();
        foreach (var rel in _relationIndex.Values)
        {
            result.Add((rel.Left, rel.Right, rel.ObservationCount));
        }
        return result;
    }

    public void ReinforceEvidence(IReadOnlyList<PlatonicEvidence> evidence, bool success)
    {
        if (evidence.Count == 0)
            return;

        foreach (var item in evidence
            .Where(e => !string.IsNullOrWhiteSpace(e.Concept))
            .OrderByDescending(e => Math.Abs(e.Contribution))
            .Take(64))
        {
            var concept = Normalize(item.Concept);
            var related = string.IsNullOrWhiteSpace(item.RelatedConcept) ? null : Normalize(item.RelatedConcept);
            var step = ++_utilityStep;
            var magnitude = Math.Clamp(Math.Abs(item.Contribution), 0.05, 1.0);
            var hopScale = 1.0 / Math.Max(1, item.Hop);

            TouchNode(concept, success, step);
            if (related is not null)
                TouchNode(related, success, step);

            if (related is null || !_relationIndex.TryGetValue(RelationKey(concept, related), out var relation))
                continue;

            TouchRelation(relation, success, step);
            var degreeAttenuation = 1.0 / (1.0 + Math.Log(1.0 + GetRelationDegree(concept) + GetRelationDegree(related)));
            var delta = 0.025 * magnitude * hopScale * degreeAttenuation;
            relation.SynthesisContradiction = success
                ? Clamp01(relation.SynthesisContradiction - delta)
                : Clamp01(relation.SynthesisContradiction + delta);
            relation.LastObservedContradiction = relation.SynthesisContradiction;
            if (_nodes.TryGetValue(concept, out var left) && _nodes.TryGetValue(related, out var right))
                UpdateConceptGeometry(left, right, relation.SynthesisContradiction, magnitude * hopScale);
        }
    }

    public int GetRelationDegree(string concept)
        => _lattice.Degree(Normalize(concept));

    // Adjacency is owned by the lattice (Layer 1 topology). The relation payload
    // (contradiction, observation counts) stays in _relationIndex, keyed by RelationKey.
    private void IndexRelation(RelationElementNode relation)
        => _lattice.AddEdge(relation.Left, relation.Right);

    private void TouchNode(string concept, bool success, long step)
    {
        if (!_nodes.TryGetValue(Normalize(concept), out var node))
            return;
        node.UseCount++;
        if (success)
            node.SuccessCount++;
        else
            node.FailureCount++;
        node.LastUsedStep = step;
    }

    private static void TouchRelation(RelationElementNode relation, bool success, long step)
    {
        relation.UseCount++;
        if (success)
            relation.SuccessCount++;
        else
            relation.FailureCount++;
        relation.LastUsedStep = step;
    }

    // RETRIEVAL track-record weight for an edge: a poor success rate → strong penalty (the edge is avoided), an
    // UNTESTED edge (no graded uses yet) → neutral 1.0 (don't punish what we haven't tried — cold start), a proven
    // edge → ~full strength. Distinct from UtilityScore (which mixes use/recency for PRUNING); this is the pure
    // right/wrong reliability that re-ranks retrieval so wrong-answer hubs stop winning.
    private static double RelationTrackRecordWeight(RelationElementNode r)
    {
        if (r.UseCount <= 0) return 1.0;
        var ratio = (r.SuccessCount + 0.5) / (r.SuccessCount + r.FailureCount + 1.0);
        return 0.25 + 0.75 * ratio;
    }

    private static IReadOnlyList<string> NormalizeConcepts(IReadOnlyList<string> concepts)
        => concepts
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(Normalize)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(32)
            .ToArray();

    private static string RelationKey(string left, string right)
    {
        var a = Normalize(left);
        var b = Normalize(right);
        return string.Compare(a, b, StringComparison.OrdinalIgnoreCase) <= 0
            ? $"{a}|{b}"
            : $"{b}|{a}";
    }

    private static PlatonicNeighborhoodType ClassifyConceptType(string token)
        => TryParseNumber(token, out _)
            ? PlatonicNeighborhoodType.Numeric
            : PlatonicNeighborhoodType.Semantic;
}
