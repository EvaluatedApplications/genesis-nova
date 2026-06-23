using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace GenesisNova.Core;

/// <summary>
/// The one platonic generative rule still in use: R2 Compose. (R1, R3–R9 were removed 2026-06-14 — they
/// ran only on a disconnected per-example scratchpad in the trainer whose elements were never written to
/// the live memory nor read by inference; their sole effect was a telemetry counter. The single useful
/// capability — composition — is invoked DIRECTLY by the glider blocks. See PLATONIC_SPACE.md §4/§7.)
/// </summary>
public enum TickKind
{
    /// <summary>R2: sum the embeddings of an element's related elements into a Composition element.</summary>
    Compose = 2
}

/// <summary>A single action on the platonic space.</summary>
public record TickAction(TickKind Kind, int PrimaryElementId);

/// <summary>Result of a single tick execution.</summary>
public record TickResult(PlatonicElement? NewElement, bool Success, string Message);

/// <summary>
/// Executes R2 Compose: builds a first-class <see cref="ElementKind.Composition"/> element from the
/// sum of an element's related elements' embeddings. Used by the glider blocks (PlatonicGlider) for
/// element-native arithmetic (the meta-layer sum is its verification oracle).
/// </summary>
public static class TickExecutor
{
    /// <summary>Execute a single tick action; returns the updated state and any newly created elements.</summary>
    public static (PlatonicState UpdatedState, ImmutableArray<PlatonicElement> NewElements)
        ExecuteTick(TickAction tick, PlatonicState state, Random? rng = null)
    {
        var newElements = ImmutableArray.CreateBuilder<PlatonicElement>();
        var elements = state.Elements;
        // Build an id->element index ONCE (was an O(n) FirstOrDefault scan per lookup, called inside the
        // RelatedTo loop → O(n·k) per compose). First-wins semantics preserved: only the FIRST occurrence of
        // a given id is kept, matching FirstOrDefault(e => e.Id == id).
        var byId = new Dictionary<int, PlatonicElement>(elements.Length);
        foreach (var element in elements)
            byId.TryAdd(element.Id, element);

        var primaryElement = FindElementById(byId, tick.PrimaryElementId);
        if (primaryElement == null)
            return (state, ImmutableArray<PlatonicElement>.Empty);

        var result = tick.Kind == TickKind.Compose
            ? ExecuteR2_Compose(primaryElement, byId, state, newElements)
            : new TickResult(null, false, $"Unsupported tick kind {tick.Kind}");

        var updatedElements = elements.ToBuilder();
        if (result.NewElement != null && newElements.All(e => e.Id != result.NewElement.Id))
            newElements.Add(result.NewElement);
        updatedElements.AddRange(newElements);
        var updatedState = state with
        {
            Elements = updatedElements.ToImmutable(),
            NextId = state.NextId + newElements.Count
        };
        return (updatedState, newElements.ToImmutable());
    }

    /// <summary>R2: compose an element's related elements (sum-of-embeddings) into a Composition element.</summary>
    private static TickResult ExecuteR2_Compose(
        PlatonicElement element,
        IReadOnlyDictionary<int, PlatonicElement> byId,
        PlatonicState state,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (element.RelatedTo.Length < 2)
            return new TickResult(null, false, "Element needs 2+ relations to compose");

        var relatedElements = new List<PlatonicElement>();
        foreach (var relId in element.RelatedTo)
        {
            var rel = FindElementById(byId, relId);
            if (rel != null) relatedElements.Add(rel);
        }
        if (relatedElements.Count < 2)
            return new TickResult(null, false, "Insufficient related elements");

        var compositionEmbedding = SumEmbeddings(relatedElements, state.EmbeddingDimension);
        var bridgeConfidence = ComputeDerivedBridgeConfidence(element, relatedElements, noveltyScore: 0.8, fallback: 0.3);

        var composition = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Composition,
            Embedding: compositionEmbedding,
            Symbol: $"C({string.Join("+", relatedElements.Select(r => r.Symbol))})",
            GeneratedAtTick: state.CurrentTick,
            NoveltyScore: 0.8,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.CreateRange(relatedElements.Select(r => r.Id)),
            GenerationPath: $"R2:compose:{string.Join(",", relatedElements.Select(r => r.Id))}");

        newElements.Add(composition);
        return new TickResult(composition, true, "R2 compose succeeded");
    }

    private static PlatonicElement? FindElementById(IReadOnlyDictionary<int, PlatonicElement> byId, int id)
        => byId.TryGetValue(id, out var element) ? element : null;

    private static double[] SumEmbeddings(List<PlatonicElement> elements, int dim)
    {
        var result = new double[dim];
        foreach (var elem in elements)
            for (int d = 0; d < dim; d++)
                result[d] += elem.Embedding[d];
        return result;
    }

    private static double ComputeDerivedBridgeConfidence(
        PlatonicElement primary, IEnumerable<PlatonicElement> related, double noveltyScore, double fallback)
    {
        var sample = new List<double>();
        if (!double.IsNaN(primary.BridgeConfidence))
            sample.Add(Clamp01(primary.BridgeConfidence));
        var relatedCount = 0;
        foreach (var element in related)
        {
            if (!double.IsNaN(element.BridgeConfidence))
                sample.Add(Clamp01(element.BridgeConfidence));
            relatedCount++;
        }
        var baseConfidence = sample.Count > 0 ? sample.Average() : fallback;
        var relationBoost = Math.Min(0.15, relatedCount * 0.03);
        var noveltyPenalty = Math.Max(0.0, Clamp01(noveltyScore) - 0.8) * 0.1;
        return Clamp01(baseConfidence + relationBoost - noveltyPenalty);
    }

    private static double Clamp01(double value) => Math.Max(0.0, Math.Min(1.0, value));
}
