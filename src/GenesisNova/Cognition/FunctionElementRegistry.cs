using System;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace GenesisNova.Cognition;

/// <summary>
/// Keyed store of composer shapes as first-class POSITIONED <see cref="Core.ElementKind.Function"/> elements
/// (realizes the dormant kind) — its own index, exactly as concept-nodes and relation-elements have theirs.
/// Each element carries a composed embedding (positioned) and a RelatedTo pointing at the OTHER shape-elements
/// it references (higher-order). The executable glider lives in PlatonicShapeRegistry; THIS is the element's
/// existence in the substrate. Extracted from <see cref="PlatonicSpaceMemory"/> (single-responsibility); kept
/// out of the concept store so it never contaminates concept retrieval.
/// </summary>
internal sealed class FunctionElementRegistry
{
    private readonly int _faceDimension;
    private readonly List<Core.PlatonicElement> _elements = new();
    private readonly Dictionary<string, int> _index = new(StringComparer.OrdinalIgnoreCase);

    public FunctionElementRegistry(int faceDimension) => _faceDimension = faceDimension;

    public IReadOnlyList<Core.PlatonicElement> Elements => _elements;

    /// <summary>
    /// Register (idempotently) a shape as a positioned Function element. <paramref name="references"/> are the
    /// names of OTHER registered shape-elements it composes (its RelatedTo). Returns the existing element if
    /// already registered.
    /// </summary>
    public Core.PlatonicElement Register(string name, IReadOnlyList<string>? references)
    {
        var key = name.Trim().ToLowerInvariant();
        if (_index.TryGetValue(key, out var existing))
            return _elements[existing];

        var related = ImmutableArray.CreateBuilder<int>();
        if (references is not null)
            foreach (var r in references)
            {
                if (string.IsNullOrWhiteSpace(r))
                    continue;
                if (_index.TryGetValue(r.Trim().ToLowerInvariant(), out var refId))
                    related.Add(refId);
            }

        var element = new Core.PlatonicElement(
            Id: _elements.Count,
            Kind: Core.ElementKind.Function,
            Embedding: Core.PlatonicFaceComposer.Compose(key, _faceDimension),
            Symbol: key,
            GeneratedAtTick: 0,
            RelatedTo: related.ToImmutable(),
            GenerationPath: "shape:function");
        _elements.Add(element);
        _index[key] = element.Id;
        return element;
    }

    public bool TryGet(string name, out Core.PlatonicElement element)
    {
        if (_index.TryGetValue(name.Trim().ToLowerInvariant(), out var id))
        {
            element = _elements[id];
            return true;
        }
        element = default!;
        return false;
    }
}
