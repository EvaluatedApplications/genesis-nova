namespace GenesisNova.Cognition;

public sealed partial class PlatonicSpaceMemory
{
    /// <summary>The Function-elements registered in the space (shapes-as-elements).</summary>
    public IReadOnlyList<Core.PlatonicElement> FunctionElements => _functions.Elements;

    /// <summary>Register (idempotently) a shape as a positioned <see cref="Core.ElementKind.Function"/> element.</summary>
    public Core.PlatonicElement RegisterFunctionElement(string name, IReadOnlyList<string>? references = null)
        => _functions.Register(name, references);

    public bool TryGetFunctionElement(string name, out Core.PlatonicElement element)
        => _functions.TryGet(name, out element);

    /// <summary>The WORD ELEMENTS registered in the space (whole-string identities in the word face).</summary>
    public IReadOnlyList<Core.PlatonicElement> WordElements => _words.Elements;

    /// <summary>
    /// Register (idempotently) a concept as a first-class WORD ELEMENT — a distinct element whose identity
    /// lives in the word face (spelling-independent), NOT a region of the concept's char face. A MULTI-word
    /// concept auto-registers each constituent word element FIRST and RELATES to them (concat); reading those
    /// related parts is decompose (<see cref="DecomposeWordElement"/>).
    /// </summary>
    public Core.PlatonicElement RegisterWordElement(string concept)
    {
        var key = (concept ?? string.Empty).Trim();
        var tokens = key.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 2)
            return _words.Register(key, null);                 // atomic word element
        foreach (var t in tokens)
            _words.Register(t, null);                          // ensure each part exists first
        return _words.Register(key, tokens);                   // the whole, relating to its parts (concat)
    }

    public bool TryGetWordElement(string concept, out Core.PlatonicElement element)
        => _words.TryGet(concept, out element);

    /// <summary>DECOMPOSE a (multi-word) word element into its constituent word elements — the inverse of the
    /// concat structure, read from the element's <c>RelatedTo</c>. Empty for an atomic / unregistered concept.</summary>
    public IReadOnlyList<Core.PlatonicElement> DecomposeWordElement(string concept)
        => _words.TryGet(concept, out var e) ? _words.Parts(e) : Array.Empty<Core.PlatonicElement>();

    /// <summary>Record one observation of <paramref name="chunk"/> as a scaffold for <paramref name="tag"/>.</summary>
    public void MineChunk(string tag, string chunk) => _chunks.Mine(tag, chunk);

    /// <summary>The most-reinforced chunk mined for <paramref name="tag"/> (false if none yet).</summary>
    public bool TryGetTopChunk(string tag, out string chunk) => _chunks.TryGetTop(tag, out chunk);

    public PlatonicMemorySnapshot ExportSnapshot()
    {
        return new PlatonicMemorySnapshot(
            FaceDimension: _faceDimension,
            Nodes: _nodes.Values
               .Select(n => new PlatonicNodeSnapshot(
                   n.Name,
                   n.PositiveFace.ToArray(),
                   n.NegativeFace.ToArray(),
                   n.ObservationCount,
                   n.UseCount,
                   n.SuccessCount,
                   n.FailureCount,
                   n.LastUsedStep))
               .ToArray(),
            Relations: _relationIndex.Values
               .Select(r => new PlatonicRelationSnapshot(
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
               .ToArray(),
            Chunks: _chunks.Export(),
            OperationTokens: _operationTokens.ToArray());
    }

    public void ImportSnapshot(PlatonicMemorySnapshot snapshot)
    {
        _nodes.Clear();
        _relationIndex.Clear();
        _lattice.Clear();

        // Restore op-tokens from the snapshot so a mid-run reload (e.g. RefreshLatestStateForReplPredict after an
        // autosave) doesn't wipe the registry and re-open the framing-word hub leak. Additive: any tokens already
        // registered for this session are kept, the persisted set is merged in.
        if (snapshot.OperationTokens is not null)
            foreach (var token in snapshot.OperationTokens)
                if (!string.IsNullOrWhiteSpace(token))
                    _operationTokens.Add(Normalize(token));

        foreach (var node in snapshot.Nodes)
        {
            var normalized = Normalize(node.Name);
            var positiveFace = Resize(node.PositiveFace, _faceDimension);
            // Numeric/operator faces are re-seeded so the homomorphic basis is exact on import;
            // all others are re-projected to unit norm (canonical) with frozen identity preserved.
            if (TryCreateSeededFace(normalized, out var seeded))
            {
                positiveFace = seeded;
            }
            else
            {
                // Re-normalize the free region; the char-face fingerprint stays as loaded.
                NormaliseFreeRegion(positiveFace, normalized);
            }
            // Hard G4 conservation: embed(¬x) = −embed(x).
            var negativeFace = new double[_faceDimension];
            for (var i = 0; i < _faceDimension; i++)
                negativeFace[i] = -positiveFace[i];
            _nodes[normalized] = new ConceptNode(
                name: normalized,
                positiveFace: positiveFace,
                negativeFace: negativeFace,
                observationCount: Math.Max(0, node.ObservationCount),
                useCount: Math.Max(0, node.UseCount),
                successCount: Math.Max(0, node.SuccessCount),
                failureCount: Math.Max(0, node.FailureCount),
                lastUsedStep: Math.Max(0, node.LastUsedStep));
            _lattice.RegisterNode(normalized);
        }

        foreach (var relation in snapshot.Relations)
        {
            var key = RelationKey(relation.Left, relation.Right);
            var conceptRelation = new RelationElementNode(
                left: Normalize(relation.Left),
                right: Normalize(relation.Right),
                thesisContradiction: Clamp01(relation.ThesisContradiction),
                lastObservedContradiction: Clamp01(relation.LastObservedContradiction),
                synthesisContradiction: Clamp01(relation.SynthesisContradiction),
                observationCount: Math.Max(0, relation.ObservationCount),
                useCount: Math.Max(0, relation.UseCount),
                successCount: Math.Max(0, relation.SuccessCount),
                failureCount: Math.Max(0, relation.FailureCount),
                lastUsedStep: Math.Max(0, relation.LastUsedStep));
            _relationIndex[key] = conceptRelation;
            IndexRelation(conceptRelation);
        }

        // Chunk store restored ADDITIVELY (counts merged) so a chunk-less maintenance snapshot never wipes
        // mined scaffolds; function-elements aren't snapshotted (re-registered deterministically by the registry).
        _chunks.ImportMerge(snapshot.Chunks);

        _utilityStep = Math.Max(
            _nodes.Values.Select(n => n.LastUsedStep).DefaultIfEmpty(0).Max(),
            _relationIndex.Values.Select(r => r.LastUsedStep).DefaultIfEmpty(0).Max());
    }
}
