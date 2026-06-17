using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace GenesisNova.Cognition;

/// <summary>
/// Keyed store of WORD ELEMENTS — first-class POSITIONED elements whose identity lives in the WORD FACE: a
/// stable, ~orthogonal WHOLE-STRING code (<see cref="Core.PlatonicFaceComposer.AddWordIdentity"/>), DECOUPLED
/// from char spelling and number value. Its own index — exactly as concept-nodes, relation-elements, and the
/// <see cref="FunctionElementRegistry"/> have theirs — so word elements never contaminate concept retrieval.
///
/// A word element is a distinct element (not a region of a concept's face): "who is this concept", spelling-
/// independent. A MULTI-word element's <c>RelatedTo</c> points at its constituent word elements, so CONCAT
/// (compose parts → a whole element) and DECOMPOSE (read the parts) are element-native relations, not string
/// surgery. The same pattern as the Function/shape registry; this is the word's existence in the substrate.
/// </summary>
internal sealed class WordElementRegistry
{
    private readonly int _faceDimension;
    private readonly List<Core.PlatonicElement> _elements = new();
    private readonly Dictionary<string, int> _index = new(StringComparer.OrdinalIgnoreCase);

    public WordElementRegistry(int faceDimension) => _faceDimension = faceDimension;

    public IReadOnlyList<Core.PlatonicElement> Elements => _elements;

    /// <summary>
    /// Register (idempotently) a word element for <paramref name="concept"/>. <paramref name="parts"/> are the
    /// symbols of OTHER already-registered word elements it composes (its <c>RelatedTo</c>); for a multi-word
    /// concept these are its constituent words (the concat structure / decompose target). Returns the existing
    /// element if already registered.
    /// </summary>
    public Core.PlatonicElement Register(string concept, IReadOnlyList<string>? parts)
    {
        var key = concept.Trim();
        var ikey = key.ToLowerInvariant();
        if (_index.TryGetValue(ikey, out var existing))
            return _elements[existing];

        var related = ImmutableArray.CreateBuilder<int>();
        if (parts is not null)
            foreach (var p in parts)
            {
                if (string.IsNullOrWhiteSpace(p))
                    continue;
                if (_index.TryGetValue(p.Trim().ToLowerInvariant(), out var pid))
                    related.Add(pid);
            }

        // The element's POSITION is its whole-string identity in the word face (orthogonal per concept).
        var embedding = new double[_faceDimension];
        Core.PlatonicFaceComposer.AddWordIdentity(embedding, key, _faceDimension);

        var element = new Core.PlatonicElement(
            Id: _elements.Count,
            Kind: Core.ElementKind.Composition, // a word element = a composed identity (multi-word → its parts)
            Embedding: embedding,
            Symbol: key,
            GeneratedAtTick: 0,
            RelatedTo: related.ToImmutable(),
            GenerationPath: "word:element");
        _elements.Add(element);
        _index[ikey] = element.Id;
        return element;
    }

    public bool TryGet(string concept, out Core.PlatonicElement element)
    {
        if (_index.TryGetValue(concept.Trim().ToLowerInvariant(), out var id))
        {
            element = _elements[id];
            return true;
        }
        element = default!;
        return false;
    }

    /// <summary>The constituent word elements a (multi-word) element composes — DECOMPOSE reads this.</summary>
    public IReadOnlyList<Core.PlatonicElement> Parts(Core.PlatonicElement element)
    {
        var parts = new List<Core.PlatonicElement>();
        if (!element.RelatedTo.IsDefaultOrEmpty)
            foreach (var id in element.RelatedTo)
                if (id >= 0 && id < _elements.Count)
                    parts.Add(_elements[id]);
        return parts;
    }
}
