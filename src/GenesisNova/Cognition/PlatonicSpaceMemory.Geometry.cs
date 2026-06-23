namespace GenesisNova.Cognition;

public sealed partial class PlatonicSpaceMemory
{
    /// <summary>
    /// Returns the positive face of a concept without side effects.
    /// For numeric concepts not in the space, returns their seeded face (homomorphic structure preserved).
    /// Returns false only for non-numeric unseen concepts.
    /// </summary>
    public bool TryGetConceptFace(string concept, out double[] positiveFace)
    {
        // Numbers always use the mathematical (homomorphic) face so that face arithmetic stays
        // exact regardless of whether a node was created by training side-effects.
        if (TryParseNumber(concept, out var numeric))
        {
            positiveFace = CreateNumericFace(numeric);
            return true;
        }
        var key = Normalize(concept);
        if (_nodes.TryGetValue(key, out var node))
        {
            positiveFace = node.PositiveFace;
            return true;
        }
        positiveFace = Array.Empty<double>();
        return false;
    }

    /// <summary>
    /// Returns nearest known concepts to <paramref name="concept"/> in face space.
    /// Optionally restricts to a candidate set to keep per-turn work bounded.
    /// </summary>
    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConcepts(
        string concept,
        IReadOnlyCollection<string>? candidates = null,
        int maxNeighbors = 8,
        int maxCandidates = 96)
    {
        var limit = Math.Clamp(maxNeighbors, 1, 32);
        if (!TryGetConceptFace(concept, out var conceptFace) || conceptFace.Length == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        var conceptKey = Normalize(concept);

        // Bounded candidate set: score those directly to keep per-turn work bounded.
        if (candidates is { Count: > 0 })
        {
            var budget = Math.Clamp(maxCandidates, 8, 512);
            var scored = new List<(string Symbol, double Distance)>(Math.Min(budget, limit * 4));
            var seen = 0;
            foreach (var candidate in candidates)
            {
                if (seen >= budget)
                    break;
                var normalized = Normalize(candidate);
                if (normalized.Equals(conceptKey, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (!_nodes.TryGetValue(normalized, out var node))
                    continue;
                scored.Add((normalized, FaceAwareDistance(conceptFace, node.PositiveFace)));
                seen++;
            }
            return scored.Count == 0
                ? Array.Empty<(string Symbol, double Distance)>()
                : scored.OrderBy(x => x.Distance).Take(limit).ToArray();
        }

        // Global nearest: O(log N) VP-Tree over the whole space (replaces the brute-force O(N) scan).
        return _lattice.GetSemanticNeighbors(conceptFace, limit, conceptKey);
    }

    /// <summary>
    /// PUSH/PULL geometry summary (READ-ONLY): over a sample of MUTABLE (non-frozen) concepts, the semantic-face
    /// distance to relation-edged neighbours (the PULL — should be SMALL) vs. to random UNRELATED concepts (the
    /// PUSH — should be LARGE). Faces are unit-normalised so distances live in [0, 2]; <c>Separation</c> =
    /// unrelatedMean − relatedMean is the magnitude the message-passing + contrastive repulsion actually moved.
    /// </summary>
    public GeometrySummary SummarizePushPullGeometry(int maxConcepts = 600, int unrelatedPerNode = 8, int seed = 1234)
    {
        var rng = new Random(seed);
        var mutable = _nodes.Values.Where(n => !IsFrozenConcept(n.Name) && n.PositiveFace is { Length: > 0 }).ToArray();
        var sample = mutable.Length <= maxConcepts ? mutable : mutable.OrderBy(_ => rng.Next()).Take(maxConcepts).ToArray();

        var related = new List<double>();
        var unrelated = new List<double>();
        foreach (var node in sample)
        {
            // PULL: distance to each relation-edged neighbour.
            foreach (var rn in _lattice.GetRelationalNeighbors(node.Name))
                if (_nodes.TryGetValue(Normalize(rn), out var other) && !IsFrozenConcept(other.Name) && other.PositiveFace.Length > 0)
                    related.Add(FaceAwareDistance(node.PositiveFace, other.PositiveFace));
            // PUSH: distance to random UNRELATED concepts (no relation edge).
            for (var s = 0; s < unrelatedPerNode && mutable.Length > 1; s++)
            {
                var other = mutable[rng.Next(mutable.Length)];
                if (ReferenceEquals(other, node))
                    continue;
                if (_relationIndex.ContainsKey(RelationKey(node.Name, other.Name)))
                    continue;
                unrelated.Add(FaceAwareDistance(node.PositiveFace, other.PositiveFace));
            }
        }

        static (double Mean, double Min, double Max) St(List<double> xs)
            => xs.Count == 0 ? (0.0, 0.0, 0.0) : (xs.Average(), xs.Min(), xs.Max());
        var r = St(related);
        var u = St(unrelated);
        return new GeometrySummary(_nodes.Count, mutable.Length, related.Count, unrelated.Count,
            r.Mean, r.Min, r.Max, u.Mean, u.Min, u.Max);
    }

    /// <summary>
    /// LIVE-FACE nearest concepts for WITHIN-STEP edit verification / perception. The global
    /// <see cref="GetNearestConcepts"/> reads the throttled VP-Tree, so a face that just moved this step is not
    /// reflected until the next rebuild — which staled the edit-head's pre/post retrievability DELTA and the
    /// route/edit perceptions (the controller couldn't "see" that a correct edit landed). This instead assembles
    /// a BOUNDED candidate set — the caller's <paramref name="seeds"/> (e.g. the example's target concepts), the
    /// anchor's relational neighbours (Layer-1, always current), and the anchor's current VP-Tree neighbourhood
    /// used ONLY as a candidate-NAME source — then re-scores every candidate against LIVE faces. So everything
    /// that moved is measured at its current position, in the same step, at O(candidates) cost (no rebuild). The
    /// periodic full rebuild remains the authority for discovering brand-new neighbourhoods across the space.
    /// </summary>
    public IReadOnlyList<(string Symbol, double Distance)> GetNearestConceptsFresh(
        string concept, IReadOnlyCollection<string>? seeds = null, int maxNeighbors = 8)
    {
        if (!TryGetConceptFace(concept, out var conceptFace) || conceptFace.Length == 0)
            return Array.Empty<(string Symbol, double Distance)>();

        var conceptKey = Normalize(concept);
        var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (seeds is not null)
            foreach (var s in seeds)
            {
                var n = Normalize(s);
                if (!n.Equals(conceptKey, StringComparison.OrdinalIgnoreCase))
                    candidates.Add(n);
            }

        // Relational neighbours (Layer-1 adjacency is maintained incrementally → always current names).
        foreach (var n in _lattice.GetRelationalNeighbors(conceptKey))
            candidates.Add(n);

        // VP-Tree neighbourhood of the anchor's CURRENT face — used only to harvest candidate NAMES (their
        // stored positions may be slightly stale); they are RE-SCORED against live faces by GetNearestConcepts.
        var pool = Math.Clamp(maxNeighbors * 3, 8, 64);
        foreach (var (n, _) in _lattice.GetSemanticNeighbors(conceptFace, pool, conceptKey))
            candidates.Add(n);

        candidates.Remove(conceptKey);
        return candidates.Count == 0
            ? Array.Empty<(string Symbol, double Distance)>()
            : GetNearestConcepts(concept, candidates: candidates, maxNeighbors: maxNeighbors);
    }

    /// <summary>TARGET-AGNOSTIC route-perception vector (SPACE_AWARE_GRU.md §I): "can the space answer a query
    /// anchored here?" — usable at INFERENCE where the target is unknown. [has-neighbour, nearest-confidence,
    /// degree-norm, mean-top-confidence, transform-reliability, bias]; dims match
    /// <c>GenesisNeuralModel.EditPerceptionDim</c>. <paramref name="transformReliability"/> is the EARNED UCB
    /// reliability of the model's best applicable learned transform (0 when none / when the feature is off) —
    /// it bubbles proven transform capability up to the route head so it learns which functions to trust.</summary>
    public double[] ComputeRoutePerception(string anchor, double transformReliability = 0.0)
    {
        // LIVE faces (not the throttled tree) so the perception reflects edits made this step — see
        // GetNearestConceptsFresh. Target-agnostic here (no seeds): relational + current-neighbourhood candidates.
        var near = GetNearestConceptsFresh(anchor, seeds: null, maxNeighbors: 4);
        var hasNeighbour = near.Count > 0 ? 1.0 : 0.0;
        var nearestConf = near.Count > 0 ? 1.0 / (1.0 + Math.Max(0.0, near[0].Distance)) : 0.0;
        var meanTopConf = near.Count > 0 ? near.Average(n => 1.0 / (1.0 + Math.Max(0.0, n.Distance))) : 0.0;
        var degreeNorm = Math.Clamp(GetRelationDegree(Normalize(anchor)) / 8.0, 0.0, 1.0);
        return new[] { hasNeighbour, nearestConf, degreeNorm, meanTopConf, Math.Clamp(transformReliability, 0.0, 1.0), 1.0 };
    }

    public void FineEditFromExample(
        IReadOnlyList<string> inputConcepts,
        IReadOnlyList<string> outputConcepts,
        bool isNegativeExample)
    {
        var inputs = NormalizeConcepts(inputConcepts);
        var outputs = NormalizeConcepts(outputConcepts);
        if (inputs.Count == 0 && outputs.Count == 0)
            return;

        var editScope = inputs.Concat(outputs).ToArray();
        var inputNodes = inputs.Select(c => GetOrCreate(c, editScope)).ToArray();
        var outputNodes = outputs.Select(c => GetOrCreate(c, editScope)).ToArray();
        if (inputNodes.Length == 0 || outputNodes.Length == 0)
            return;

        var inputCentroid = ComputeCentroid(inputNodes);
        var outputCentroid = ComputeCentroid(outputNodes);
        var rate = isNegativeExample ? 0.03 : 0.06;
        var outputSign = isNegativeExample ? -1.0 : 1.0;
        var inputSign = isNegativeExample ? -0.3 : 0.3;

        foreach (var node in outputNodes)
            ApplyCentroidNudge(node, inputCentroid, outputSign, rate);

        foreach (var node in inputNodes)
            ApplyCentroidNudge(node, outputCentroid, inputSign, rate * 0.5);
    }

    /// <summary>
    /// TASK-OUTCOME DISRUPTION (PLATONIC_BACKPROP.md Rung 1 — gradient-free function signal). A VALUE-WRONG answer
    /// means the geometry placed <paramref name="anchor"/> close enough to <paramref name="answer"/> to retrieve it
    /// — so actively REPEL the answer's face away from the anchor (a hard negative anchored on the REAL failure,
    /// not a random draw — the §3 skeleton, but driven by the end-task outcome instead of graph structure). If a
    /// now-wrong relation edge produced it, weaken that edge too. This is the missing path: the answer-loss reaching
    /// the space to disrupt the element that created the wrong answer. Numbers are NEVER scattered (frozen) — a
    /// numeric mis-retrieval (apple→29) is a routing/filter problem, not a geometry one; reserved markers/op-tokens
    /// are not answers. No-op unless both are real, mutable, distinct concepts that exist in the space.
    /// </summary>
    public void DisruptAssociation(string anchor, string answer)
    {
        if (string.IsNullOrWhiteSpace(anchor) || string.IsNullOrWhiteSpace(answer))
            return;
        var a = Normalize(anchor);
        var b = Normalize(answer);
        if (a == b || IsFrozenConcept(a) || IsFrozenConcept(b) || IsReservedConcept(a) || IsReservedConcept(b))
            return;
        if (!_nodes.TryGetValue(a, out var left) || !_nodes.TryGetValue(b, out var right))
            return;

        // If a relation edge produced this wrong answer, mark the failure on it (weaken — raise contradiction).
        if (_relationIndex.TryGetValue(RelationKey(a, b), out var rel))
        {
            var step = ++_utilityStep;
            TouchRelation(rel, success: false, step);
            rel.SynthesisContradiction = Clamp01(rel.SynthesisContradiction + DisruptionContradictionStep);
            rel.LastObservedContradiction = rel.SynthesisContradiction;
        }

        // Repel the two faces — targetContradiction 1.0 ⇒ affinity −1 ⇒ MessagePassUpdate pushes them apart.
        UpdateConceptGeometry(left, right, targetContradiction: 1.0, rateScale: 1.0);
        _lattice.MarkEmbeddingsDirty();
    }

    /// <summary>
    /// RUNG 2 (PLATONIC_BACKPROP.md §4–§6) — the FUNCTION GRADIENT, computed analytically in face space. Treat the
    /// anchor's face as the query <c>q</c>, the CORRECT answer as the positive, and the confusers (incl. whatever
    /// was wrongly retrieved) as negatives; descend softmax cross-entropy over the dot-product scores so the
    /// anchor's nearest neighbour BECOMES the target. The gradient is exactly "pull the positive toward q, push
    /// each negative away, self-scaled by the probability it stole": <c>dL/dc_i = (p_i − 1{target})·q</c> — the
    /// InfoNCE force with NO hand constant (the force rate emerges from how wrong each pair is). Unlike the §4
    /// GEOMETRY gradient (graph edges as positives), the positive here is the TASK target — so the space is shaped
    /// by whether it produces the RIGHT answer, which is the missing function path (§2/§5). Only FREE dims move
    /// (identity frozen); inference still uses the hard read (STE spirit — train soft, ship hard). No-op unless the
    /// anchor + target are real, mutable, distinct concepts and at least one valid negative exists.
    /// </summary>
    public void FunctionGradientStep(string anchor, string target, IReadOnlyList<string> distractors, double rate = FunctionGradientRate)
    {
        var qName = Normalize(anchor);
        var tName = Normalize(target);
        if (qName == tName || IsFrozenConcept(qName) || IsFrozenConcept(tName) || IsReservedConcept(tName))
            return;
        if (!TryGetConceptFace(qName, out var qLive) || !_nodes.TryGetValue(tName, out var tNode))
            return;
        var q = (double[])qLive.Clone(); // fixed query reference for this step

        var cands = new List<ConceptNode> { tNode };                                   // index 0 = the positive
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { qName, tName };
        foreach (var d in distractors ?? Array.Empty<string>())
        {
            var dn = Normalize(d);
            if (!seen.Add(dn) || IsFrozenConcept(dn) || IsReservedConcept(dn))
                continue;
            if (_nodes.TryGetValue(dn, out var dNode))
                cands.Add(dNode);
        }
        if (cands.Count < 2)
            return; // need at least one negative to contrast against

        var dim = _faceDimension;
        var p = new double[cands.Count];
        var maxS = double.NegativeInfinity;
        for (var i = 0; i < cands.Count; i++)
        {
            var c = cands[i].PositiveFace;
            var s = 0.0;
            for (var d = 0; d < dim && d < c.Length && d < q.Length; d++) s += q[d] * c[d];
            p[i] = s; if (s > maxS) maxS = s;
        }
        var sum = 0.0;
        for (var i = 0; i < cands.Count; i++) { p[i] = Math.Exp(p[i] - maxS); sum += p[i]; }
        for (var i = 0; i < cands.Count; i++) p[i] /= sum;                              // softmax probabilities

        for (var i = 0; i < cands.Count; i++)
        {
            var coeff = rate * (p[i] - (i == 0 ? 1.0 : 0.0));   // (p_i − 1{target}); <0 for target (pull), >0 (push)
            var node = cands[i];
            var original = (double[])node.PositiveFace.Clone();
            var face = node.PositiveFace;
            for (var d = 0; d < dim && d < q.Length; d++)
                face[d] -= coeff * q[d];
            RestoreFrozenIdentity(face, original, node.Name);   // identity dims never move
            NormaliseFreeRegion(face, node.Name);
            for (var d = 0; d < dim; d++) node.NegativeFace[d] = -face[d];
        }
        _lattice.MarkEmbeddingsDirty();
    }

    /// <summary>
    /// Canonical embedding message-passing update (conforms to GraphAligner / EmbeddingSpace.UpdateEmbedding
    /// in the source of truth). Replaces the non-canonical contradiction→spring-distance mutator.
    ///
    /// For each concept we (1) pull its positive face toward its neighbour (excluding the complement),
    /// (2) repel from the complement (the negative face) to maintain dual-space separation,
    /// (3) restore frozen identity dims (arithmetic poly/log seeds, operator faces) so the homomorphism
    /// never drifts, (4) re-normalize to UNIT norm (||e|| = 1), and (5) enforce the hard G4 conservation
    /// law embed(¬x) = −embed(x).
    ///
    /// The contradiction signal selects pull (low contradiction = agree → pull together) vs. push
    /// (high contradiction = disagree → push apart), preserving the external behavior callers rely on
    /// (ObserveContradiction / ReinforceEvidence), while routing all value updates through the canonical rule.
    /// </summary>
    private void UpdateConceptGeometry(ConceptNode a, ConceptNode b, double targetContradiction, double rateScale)
    {
        var baseRate = GeometryLearningRate * Math.Clamp(rateScale, 0.0, 1.0);
        // Canonical alpha = max(0.05, 2/(n+1)); n here is the relational degree (neighbour count proxy).
        var nA = GetRelationDegree(a.Name);
        var nB = GetRelationDegree(b.Name);
        var alphaA = baseRate * Math.Max(0.05, 2.0 / (nA + 1)) * NodePlasticity(a);
        var alphaB = baseRate * Math.Max(0.05, 2.0 / (nB + 1)) * NodePlasticity(b);

        // Low contradiction → concepts agree → pull together (+1).
        // High contradiction → concepts disagree → push apart (−1).
        var affinity = 1.0 - (2.0 * Clamp01(targetContradiction)); // [+1 .. -1]
        // Floor the PULL so an observed relation clusters even under a weak edit-head magnitude (balance the
        // push). Only when attracting — a repel (affinity<0) keeps its full strength.
        if (affinity > 0.0)
            affinity = Math.Max(affinity, MinAttractAffinity);

        MessagePassUpdate(a, b.PositiveFace, alphaA, affinity);
        MessagePassUpdate(b, a.PositiveFace, alphaB, affinity);
    }

    /// <summary>
    /// One canonical message-passing step on a single node's positive face:
    /// pull toward (or push from) a neighbour face, repel from the node's own complement (negative face),
    /// restore frozen identity dims, unit-normalize, then enforce the hard complement (NegativeFace = −PositiveFace).
    /// </summary>
    private void MessagePassUpdate(ConceptNode node, double[] neighbourFace, double alpha, double affinity)
    {
        // Numbers move their NON-identity (char/word) region like any concept — pulled toward their word
        // partners (7<->"seven") and pushed from unrelated — while RestoreFrozenIdentity below keeps the
        // poly/log VALUE exact (IdentityRange("7") = [0,42), so [42,dim) is its free region). The old
        // whole-node freeze left every number's word face at the origin (word=0.000), so all numbers were
        // indistinguishable there and number<->word retrieval collapsed to a single constant.
        var dim = _faceDimension;
        var original = (double[])node.PositiveFace.ToArray();
        var updated = node.PositiveFace;

        // (1) Neighbour pull / push — canonical: updated += lr * (neighbour - updated), signed by affinity.
        // DIALECTIC (Phase 1): when DimensionalContradiction is on, gate the step PER DIMENSION. Dims where the
        // pair currently AGREES (face product ≥ 0) get the full step — reinforce a shared aspect (pull) or break a
        // false agreement (push); dims where it CONTRADICTS (product < 0) move only at DialecticPreserveFraction, so
        // the distinguishing aspects survive and position settles as the synthesis of agreements and contradictions
        // (Law D), instead of the whole-vector collapse toward identical (Law M violation). Off ⇒ legacy scalar.
        for (var i = 0; i < dim; i++)
        {
            var gate = DimensionalContradiction && (updated[i] * neighbourFace[i]) < 0.0
                ? DialecticPreserveFraction
                : 1.0;
            updated[i] += alpha * affinity * gate * (neighbourFace[i] - updated[i]);
        }

        // (2) Complement repulsion — push AWAY from the negative face to maintain dual-space separation
        // (canonical EmbeddingSpace.UpdateEmbedding repels from the complement when too close).
        var dist = EuclideanDistance(updated, node.NegativeFace);
        if (dist < 1.35)
        {
            var repulsion = alpha * 0.5;
            for (var i = 0; i < dim; i++)
                updated[i] += repulsion * (updated[i] - node.NegativeFace[i]);
        }

        // (3) Restore frozen identity dims (arithmetic/lexical identity never drifts), then
        // (4) unit-normalize ONLY the free region so learned wiggle magnitude survives.
        RestoreFrozenIdentity(updated, original, node.Name);
        NormaliseFreeRegion(updated, node.Name);

        // (5) Hard G4 conservation: embed(¬x) = −embed(x).
        for (var i = 0; i < dim; i++)
            node.NegativeFace[i] = -node.PositiveFace[i];

        _lattice.MarkEmbeddingsDirty();
    }

    /// <summary>
    /// Contrastive repulsion pass — the genesis <c>NudgeGraphAlignment</c> repulsion half, ported.
    /// For each mutable concept, push its face AWAY from a few sampled UNRELATED concepts (those with
    /// no relation edge), so unrelated/fringe things stay far apart and proximity is earned only by
    /// confirmed attraction. Frozen identity dims are restored and the free region re-normalised, so
    /// the homomorphism and lexical fingerprints never drift. Deterministic (seeded per node+pass).
    /// </summary>
    private void ApplyContrastiveRepulsionPass()
    {
        var n = _nodes.Count;
        if (n < 3)
            return;

        var snapshot = _nodes.Values.ToArray();
        // Repel ONLY from MUTABLE (non-number) targets, and move only those. Numbers reach their place in the
        // word face by ATTRACTION to their word partner (7<->"seven"), NOT by repulsion: actively scattering the
        // ~60% of nodes that are numbers INTO the semantic word-space turns them into distractors that break
        // text-vs-text retrieval (measured: synonym 100%->19%, category 25%->0%). Attraction-only settles each
        // number ONTO its (already-separated) word, so digit<->word works without numbers invading unrelated
        // text retrieval. RestoreFrozenIdentity still keeps every number's poly/log value exact.
        var targets = snapshot.Where(x => !IsFrozenConcept(x.Name) && x.PositiveFace is { Length: > 0 }).ToArray();
        if (targets.Length < 2)
            return;
        var dim = _faceDimension;
        _repulsionPassCount++;

        foreach (var node in snapshot)
        {
            if (IsFrozenConcept(node.Name)) // numbers don't get pushed — they ride attraction to their word
                continue;

            var original = (double[])node.PositiveFace.Clone();
            var updated = node.PositiveFace;
            var rng = new Random(unchecked(StableHash(node.Name) + _repulsionPassCount));

            // SEMI-HARD NEGATIVES (PLATONIC_BACKPROP.md §3): sample a candidate pool, score by LIVE distance,
            // and push from the NEAREST truly-unrelated ones — the confusers actually colliding NOW — rather
            // than uniform-random draws (which mostly hit already-far concepts and under-cover at scale). Uses
            // LIVE faces (not the throttled VP-Tree, whose stale targets let attraction collapse the space).
            // EXEMPT both direct edges (1-hop) AND shared-neighbour pairs (2-hop): co-members of a hub are
            // related VIA the hub even without a direct edge, so pushing them apart would shred clusters. This
            // mirrors the pull (anchored on the node's real neighbourhood) — the InfoNCE skeleton.
            var nodeNbrs = new HashSet<string>(_lattice.GetRelationalNeighbors(node.Name), StringComparer.OrdinalIgnoreCase);
            var pool = Math.Min(RepulsionSamples * 3, targets.Length);
            var cands = new List<(ConceptNode Other, double Dist2)>(pool);
            for (var c = 0; c < pool; c++)
            {
                var other = targets[rng.Next(targets.Length)];
                if (ReferenceEquals(other, node)) continue;
                if (_relationIndex.ContainsKey(RelationKey(node.Name, other.Name))) continue; // 1-hop edged = exempt
                if (nodeNbrs.Count > 0 && _lattice.GetRelationalNeighbors(other.Name).Any(nodeNbrs.Contains))
                    continue; // 2-hop (shared neighbour) = related via a hub → exempt
                var of0 = other.PositiveFace;
                var d2 = 0.0;
                for (var d = 0; d < dim && d < of0.Length; d++) { var dx = updated[d] - of0[d]; d2 += dx * dx; }
                if (d2 >= 1e-12) cands.Add((other, d2));
            }
            cands.Sort((x, y) => x.Dist2.CompareTo(y.Dist2)); // nearest first = hardest negatives
            var pushCount = Math.Min(RepulsionSamples, cands.Count);

            if (UseInfoNceRepulsion && pushCount > 0)
            {
                // INFONCE GRADIENT: weight each negative by softmax(-distance/τ) — the hardest (nearest) negative
                // gets the most push, self-scaling (Σw=1). This IS backprop of the contrastive loss's repulsion
                // term (analytic, the faces being double[]). One InfoNceStep replaces the force constants.
                var w = new double[pushCount];
                var maxSim = double.NegativeInfinity;
                for (var s = 0; s < pushCount; s++) { w[s] = -Math.Sqrt(cands[s].Dist2) / InfoNceTau; if (w[s] > maxSim) maxSim = w[s]; }
                var sumExp = 0.0;
                for (var s = 0; s < pushCount; s++) { w[s] = Math.Exp(w[s] - maxSim); sumExp += w[s]; }
                for (var s = 0; s < pushCount; s++)
                {
                    var of = cands[s].Other.PositiveFace;
                    var g = InfoNceStep * (w[s] / sumExp);
                    for (var d = 0; d < dim && d < of.Length; d++)
                        updated[d] += g * (updated[d] - of[d]);
                }
            }
            else
            {
                for (var s = 0; s < pushCount; s++)
                {
                    var of = cands[s].Other.PositiveFace;
                    var dist2 = 0.0;
                    for (var d = 0; d < dim && d < of.Length; d++) { var dx = updated[d] - of[d]; dist2 += dx * dx; }
                    if (dist2 < 1e-12) continue;
                    // Repulsive force inversely proportional to distance (genesis: alpha / max(dist, 0.01)).
                    var force = (RepulsionRate * RepulsionRatio) / Math.Max(Math.Sqrt(dist2), 0.01);
                    for (var d = 0; d < dim && d < of.Length; d++)
                        updated[d] += force * (updated[d] - of[d]);
                }
            }

            // Identity never drifts; free region re-normalised; hard G4 complement re-enforced.
            RestoreFrozenIdentity(updated, original, node.Name);
            NormaliseFreeRegion(updated, node.Name);
            for (var d = 0; d < dim; d++)
                node.NegativeFace[d] = -updated[d];
        }

        _lattice.MarkEmbeddingsDirty();
    }

    private static int StableHash(string value)
    {
        var hash = 2166136261u;
        foreach (var c in value)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return unchecked((int)hash);
    }

    private bool TryCreateSeededFace(string concept, out double[] face)
    {
        if (TryParseNumber(concept, out var numeric))
        {
            face = CreateNumericFace(numeric);
            return true;
        }

        // Operator words are no longer special-cased: the operation is determined by parsing the
        // operator symbol (and by the learned transforms), so "plus"/"times"/etc. are ordinary
        // learnable text concepts composed on the canonical face geometry like any other word.
        face = Array.Empty<double>();
        return false;
    }

    private double[] CreateNumericFace(double value)
    {
        var face = new double[_faceDimension];
        var numericDims = Math.Min(_faceDimension / 2, 21);
        var logStart = numericDims;
        var logDims = Math.Min(numericDims, _faceDimension - logStart);

        for (var i = 0; i < numericDims; i++)
            face[i] = value * Math.Pow(10, -(i + 1));

        if (Math.Abs(value) > 1e-12)
        {
            var logValue = Math.Log(Math.Abs(value));
            for (var i = 0; i < logDims; i++)
                face[logStart + i] = logValue * Math.Pow(10, -(i + 1));
        }

        return face;
    }

    // Text concepts are composed on the canonical face geometry (clean char slots / word face +
    // seeded free dims) so a stored concept face and its input embedding are directly comparable.
    // Numeric concepts never reach here — they are seeded via TryCreateSeededFace.
    private double[] CreateFace(string concept)
        => Core.PlatonicFaceComposer.Compose(concept, _faceDimension);

    private double[] ComputeCentroid(IReadOnlyList<ConceptNode> nodes)
    {
        var centroid = new double[_faceDimension];
        if (nodes.Count == 0)
            return centroid;

        foreach (var node in nodes)
        {
            for (var i = 0; i < _faceDimension; i++)
                centroid[i] += node.PositiveFace[i];
        }

        var scale = 1.0 / nodes.Count;
        for (var i = 0; i < _faceDimension; i++)
            centroid[i] *= scale;
        return centroid;
    }

    private void ApplyCentroidNudge(ConceptNode node, IReadOnlyList<double> centroid, double sign, double rate)
    {
        // Numeric concepts are ground truth — never nudged.
        if (IsFrozenConcept(node.Name))
            return;

        var effectiveRate = rate * NodePlasticity(node);
        var original = (double[])node.PositiveFace.ToArray();
        for (var i = 0; i < _faceDimension; i++)
        {
            var delta = centroid[i] - node.PositiveFace[i];
            node.PositiveFace[i] += sign * effectiveRate * delta;
        }

        // Restore identity dims, then unit-normalize ONLY the free region (preserve wiggle magnitude).
        RestoreFrozenIdentity(node.PositiveFace, original, node.Name);
        NormaliseFreeRegion(node.PositiveFace, node.Name);
        for (var i = 0; i < _faceDimension; i++)
            node.NegativeFace[i] = -node.PositiveFace[i];

        _lattice.MarkEmbeddingsDirty();
    }

    private double NodePlasticity(ConceptNode node)
    {
        var degree = GetRelationDegree(node.Name);
        var usage = Math.Log(1.0 + Math.Max(0, degree)) + Math.Log(1.0 + Math.Max(0, node.ObservationCount));
        return 1.0 / (1.0 + usage);
    }

    // Only numbers are frozen ground truth: their poly/log faces are the homomorphic basis and
    // must never drift. Everything else (including operator words) is learnable.
    private static bool IsFrozenConcept(string concept) => TryParseNumber(concept, out _);

    /// <summary>
    /// The frozen identity region of a concept's face — the dims that carry its exact, canonical
    /// signature and must never drift. Everything OUTSIDE this range is free to "wiggle" (learnable).
    /// <list type="bullet">
    ///   <item>Numbers: the arithmetic face [0 .. 2*NumericDimensions) (poly + log) —
    ///   the homomorphic basis (value*10^-i, ln|value|*10^-i).</item>
    ///   <item>Text concepts: the character-slot face [CharFaceStart .. WordFaceStart) — the
    ///   lexical fingerprint. The word face and all other dims stay learnable. (The spelling-independent
    ///   per-concept IDENTITY lives on a SEPARATE word ELEMENT, not in the concept's face — see
    ///   <c>RegisterWordElement</c> / WordElementRegistry — so it is stable by being in its own index.)</item>
    /// </list>
    /// </summary>
    private (int Start, int End) IdentityRange(string concept)
    {
        if (IsFrozenConcept(concept))
            return (0, Math.Min(2 * NumericDimensions, _faceDimension));

        var charStart = Core.FaceLayout.CharFaceStart(_faceDimension);
        var charEnd = Core.FaceLayout.WordFaceStart(_faceDimension);
        return charEnd > charStart && charEnd <= _faceDimension ? (charStart, charEnd) : (0, 0);
    }

    /// <summary>
    /// Restore the frozen identity dims after an embedding update so the concept's exact signature
    /// (arithmetic homomorphism for numbers, lexical fingerprint for text) never drifts. Conforms to
    /// EmbeddingSpace.RestoreFrozenIdentity, generalised per-region by concept type. Guarantees
    /// embed(a+b)=embed(a)+embed(b) and ln(a*b)=ln a + ln b survive message passing.
    /// </summary>
    private void RestoreFrozenIdentity(double[] updated, double[] original, string concept)
    {
        var (start, end) = IdentityRange(concept);
        for (var i = start; i < end && i < updated.Length; i++)
            updated[i] = original[i];
    }

    /// <summary>
    /// Unit-normalize ONLY the free (non-identity) region of a face, leaving the frozen identity dims
    /// exactly as-is. Fixes the prior whole-vector normalization, which renormalized every dim each
    /// tick and so erased the learned MAGNITUDE of the free "wiggle" dims (only direction survived).
    /// Per-region normalization keeps the free dims at a stable, non-vanishing scale while the identity
    /// dims stay exact.
    /// </summary>
    private void NormaliseFreeRegion(double[] face, string concept)
    {
        var (idStart, idEnd) = IdentityRange(concept);
        var sum = 0.0;
        for (var i = 0; i < face.Length; i++)
        {
            if (i >= idStart && i < idEnd) continue;
            sum += face[i] * face[i];
        }

        var norm = Math.Sqrt(sum);
        if (norm <= MaxFreeNorm)
            return; // C2: free to roam radially below the cap — only clamp the explosive tail

        var inv = MaxFreeNorm / norm;
        for (var i = 0; i < face.Length; i++)
        {
            if (i >= idStart && i < idEnd) continue;
            face[i] *= inv;
        }
    }

    /// <summary>
    /// Total charge of the space (G4 diagnostic): sum of every coordinate across all faces.
    /// With the hard complement embed(¬x) = −embed(x) enforced on every update, positive and
    /// negative faces cancel pairwise so this should stay ≈ 0.
    /// </summary>
    public double TotalCharge()
    {
        var sum = 0.0;
        foreach (var node in _nodes.Values)
        {
            for (var i = 0; i < node.PositiveFace.Length; i++)
                sum += node.PositiveFace[i] + node.NegativeFace[i];
        }
        return sum;
    }

    private static double EuclideanDistance(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        var length = Math.Min(a.Count, b.Count);
        var sum = 0.0;
        for (var i = 0; i < length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    // FACE-AWARE distance (genesis hybrid): relatedness is measured on the SEMANTIC face [WordFaceStart..dim)
    // so a concept's value (numeric face) and spelling (char face) cannot contaminate it. For a NUMERIC
    // query (arithmetic-face signal present) we ALSO measure the arithmetic face and take the MIN, so
    // numbers still surface value-near neighbours while text comparisons stay shielded. This is the real
    // fix the per-concept zero-freezing was bandaging.
    private double FaceAwareDistance(IReadOnlyList<double> query, IReadOnlyList<double> candidate)
    {
        var dim = _faceDimension;
        var arithEnd = Math.Min(2 * NumericDimensions, dim);
        var semStart = Core.FaceLayout.WordFaceStart(dim);
        var semantic = semStart > 0 && semStart < dim
            ? RangeDistance(query, candidate, semStart, dim)
            : RangeDistance(query, candidate, 0, dim);

        var arithNorm = 0.0;
        for (var i = 0; i < arithEnd && i < query.Count; i++)
            arithNorm += query[i] * query[i];
        if (arithNorm <= 0.01) // text query → semantic face only
            return semantic;

        return Math.Min(semantic, RangeDistance(query, candidate, 0, arithEnd)); // numeric → blend value face
    }

    private static double RangeDistance(IReadOnlyList<double> a, IReadOnlyList<double> b, int start, int end)
    {
        var n = Math.Min(a.Count, b.Count);
        var e = Math.Min(end, n);
        var s = Math.Min(Math.Max(0, start), e);
        var sum = 0.0;
        for (var i = s; i < e; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double FaceDistance(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var length = Math.Min(left.Count, right.Count);
        var sum = 0.0;
        for (var i = 0; i < length; i++)
        {
            var d = left[i] - right[i];
            sum += d * d;
        }

        return Math.Sqrt(sum / Math.Max(1, length));
    }
}
