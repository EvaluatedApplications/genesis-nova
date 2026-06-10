using System.Collections.Immutable;
using GenesisNova.Inference;

namespace GenesisNova.Core;

/// <summary>
/// A single action that mutates the platonic space: R1-R8 generative rules.
/// </summary>
public enum TickKind
{
    /// <summary>R1: Link isolated element to nearest by kind</summary>
    Relate = 1,
    
    /// <summary>R2: Sum embeddings of related elements (composition)</summary>
    Compose = 2,
    
    /// <summary>R3: Probe direction of unexpected position (surprise)</summary>
    Surprise = 3,
    
    /// <summary>R4: Transfer embedding arithmetic pattern (analogy)</summary>
    Analogy = 4,
    
    /// <summary>R5: Close A→M→C by inferring A→C (gap closing)</summary>
    Gap = 5,
    
    /// <summary>R6: Update transform for function (local learning)</summary>
    LocalLearn = 6,
    
    /// <summary>R7: Split agent at confusion point (branch detection)</summary>
    BranchDetect = 7,
    
    /// <summary>R8: Discover composition pattern (fold chain)</summary>
    FoldChain = 8,

    /// <summary>R9: Invoke a space-management tool</summary>
    SpaceTool = 9
}

/// <summary>
/// A single action in the platonic space.
/// </summary>
public record TickAction(
    TickKind Kind,
    int PrimaryElementId,
    int[]? SecondaryIds = null,
    string? Parameter = null,
    double[]? AuxiliaryEmbedding = null
);

/// <summary>
/// Result of a single tick execution.
/// </summary>
public record TickResult(
    PlatonicElement? NewElement,
    bool Success,
    string Message
);

/// <summary>
/// Executes generative rules (R1-R8) on the platonic space.
/// </summary>
public static class TickExecutor
{
    /// <summary>
    /// Execute a single tick action on the platonic state.
    /// Returns updated state and newly created elements.
    /// </summary>
    public static (PlatonicState UpdatedState, ImmutableArray<PlatonicElement> NewElements) 
        ExecuteTick(
            TickAction tick,
            PlatonicState state,
            Random? rng = null)
    {
        rng ??= new Random();
        
        var newElements = ImmutableArray.CreateBuilder<PlatonicElement>();
        var elements = state.Elements;
        var primaryElement = FindElementById(elements, tick.PrimaryElementId);
        
        if (primaryElement == null)
            return (state, ImmutableArray<PlatonicElement>.Empty);
        
        TickResult result = tick.Kind switch
        {
            TickKind.Relate => ExecuteR1_Relate(primaryElement, elements, state, newElements),
            TickKind.Compose => ExecuteR2_Compose(primaryElement, elements, state, tick.SecondaryIds, newElements),
            TickKind.Surprise => ExecuteR3_Surprise(primaryElement, elements, state, rng, newElements),
            TickKind.Analogy => ExecuteR4_Analogy(primaryElement, elements, state, tick.SecondaryIds, newElements),
            TickKind.Gap => ExecuteR5_Gap(primaryElement, elements, state, tick.SecondaryIds, newElements),
            TickKind.LocalLearn => ExecuteR6_LocalLearn(primaryElement, elements, state, tick.Parameter, tick.AuxiliaryEmbedding, newElements),
            TickKind.BranchDetect => ExecuteR7_BranchDetect(primaryElement, elements, state, newElements),
            TickKind.FoldChain => ExecuteR8_FoldChain(primaryElement, elements, state, tick.SecondaryIds, newElements),
            TickKind.SpaceTool => ExecuteR9_SpaceTool(primaryElement, elements, state, tick.Parameter, newElements),
            _ => new TickResult(null, false, "Unknown tick kind")
        };

        var updatedElements = elements.ToBuilder();
        if (result.NewElement != null)
        {
            var existingIndex = -1;
            for (var i = 0; i < updatedElements.Count; i++)
            {
                if (updatedElements[i].Id != result.NewElement.Id)
                {
                    continue;
                }

                existingIndex = i;
                break;
            }

            if (existingIndex >= 0)
            {
                updatedElements[existingIndex] = result.NewElement;
            }
            else if (!newElements.Any(e => e.Id == result.NewElement.Id))
            {
                newElements.Add(result.NewElement);
            }
        }

        updatedElements.AddRange(newElements);
        var updatedState = state with
        {
            Elements = updatedElements.ToImmutable(),
            NextId = state.NextId + newElements.Count
        };
        
        return (updatedState, newElements.ToImmutable());
    }
    
    /// <summary>R1: Relate isolated element to nearest</summary>
    private static TickResult ExecuteR1_Relate(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (!element.RelatedTo.IsEmpty)
            return new TickResult(null, false, "Element already has relations");
        
        var nearest = FindNearestByKind(elements, element);
        if (nearest == null)
            return new TickResult(null, false, "No nearest element found");
        
        var relationEmbedding = InterpolateEmbeddings(
            element.Embedding, nearest.Embedding, state.EmbeddingDimension);
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            new[] { nearest },
            noveltyScore: 1.0,
            fallback: 0.5);
        
        var relation = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Relation,
            Embedding: relationEmbedding,
            Symbol: $"R({element.Symbol}→{nearest.Symbol})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 1.0,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.Create(element.Id, nearest.Id),
            GenerationPath: $"R1:relate:{element.Id},{nearest.Id}"
        );
        
        newElements.Add(relation);
        return new TickResult(relation, true, "R1 relate succeeded");
    }
    
    /// <summary>R2: Compose related elements</summary>
    private static TickResult ExecuteR2_Compose(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        int[]? secondaryIds,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (element.RelatedTo.Length < 2)
            return new TickResult(null, false, "Element needs 2+ relations to compose");
        
        var relatedElements = new List<PlatonicElement>();
        foreach (var relId in element.RelatedTo)
        {
            var rel = FindElementById(elements, relId);
            if (rel != null) relatedElements.Add(rel);
        }
        
        if (relatedElements.Count < 2)
            return new TickResult(null, false, "Insufficient related elements");
        
        var compositionEmbedding = SumEmbeddings(relatedElements, state.EmbeddingDimension);
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            relatedElements,
            noveltyScore: 0.8,
            fallback: 0.3);
        
        var composition = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Composition,
            Embedding: compositionEmbedding,
            Symbol: $"C({string.Join("+", relatedElements.Select(r => r.Symbol))})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.8,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.CreateRange(relatedElements.Select(r => r.Id)),
            GenerationPath: $"R2:compose:{string.Join(",", relatedElements.Select(r => r.Id))}"
        );
        
        newElements.Add(composition);
        return new TickResult(composition, true, "R2 compose succeeded");
    }
    
    /// <summary>R3: Probe surprising direction</summary>
    private static TickResult ExecuteR3_Surprise(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        Random rng,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (element.RelatedTo.Length < 2)
            return new TickResult(null, false, "Element needs neighbors for surprise");
        
        const double surpriseThreshold = 0.4;
        var neighbors = element.RelatedTo.Select(id => FindElementById(elements, id))
                .Where(e => e != null)
                .Cast<PlatonicElement>()
                .ToList();
        var expectedEmbedding = ComputeMeanEmbedding(neighbors, state.EmbeddingDimension);
        
        var surprise = EuclideanDistance(element.Embedding, expectedEmbedding);
        if (surprise < surpriseThreshold)
            return new TickResult(null, false, "Insufficient surprise");
        
        var directionEmbedding = new double[state.EmbeddingDimension];
        for (int d = 0; d < state.EmbeddingDimension; d++)
            directionEmbedding[d] = element.Embedding[d] - expectedEmbedding[d];
        
        // Probe magnitude: follow surprise in the unexpected direction
        var probeMagnitude = (float)Math.Sqrt(directionEmbedding.Sum(d => d * d));
        if (probeMagnitude < 1e-6f)
            probeMagnitude = 1e-6f;
        var probeEmbedding = new double[state.EmbeddingDimension];
        for (int d = 0; d < state.EmbeddingDimension; d++)
            probeEmbedding[d] = element.Embedding[d] + (directionEmbedding[d] / probeMagnitude) * 0.5;
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            neighbors,
            noveltyScore: Math.Min(1.0, surprise),
            fallback: 0.4);
        
        var probeElement = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Object,
            Embedding: probeEmbedding,
            Symbol: $"P(surprise:{element.Symbol})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: Math.Min(1.0, surprise),
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.Create(element.Id),
            GenerationPath: $"R3:surprise:{element.Id}",
            IsHypothesis: true
        );
        
        newElements.Add(probeElement);
        return new TickResult(probeElement, true, "R3 surprise probe succeeded");
    }
    
    /// <summary>R4: Transfer embedding arithmetic (analogy)</summary>
    private static TickResult ExecuteR4_Analogy(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        int[]? secondaryIds,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (secondaryIds == null || secondaryIds.Length < 2)
            return new TickResult(null, false, "R4 needs 2 secondary IDs");
        
        var neighbor = FindElementById(elements, secondaryIds[0]);
        var neighborTarget = FindElementById(elements, secondaryIds[1]);
        
        if (neighbor == null || neighborTarget == null)
            return new TickResult(null, false, "Secondary elements not found");
        
        // Transfer: if Y→Z and X≈Y, then X should go to X + (Z - Y)
        var delta = new double[state.EmbeddingDimension];
        for (int d = 0; d < state.EmbeddingDimension; d++)
            delta[d] = neighborTarget.Embedding[d] - neighbor.Embedding[d];
        
        var analogyEmbedding = new double[state.EmbeddingDimension];
        for (int d = 0; d < state.EmbeddingDimension; d++)
            analogyEmbedding[d] = element.Embedding[d] + delta[d];
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            new[] { neighbor, neighborTarget },
            noveltyScore: 0.9,
            fallback: 0.3);
        
        var analogyElement = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Object,
            Embedding: analogyEmbedding,
            Symbol: $"A({element.Symbol}→...)",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.9,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.Create(element.Id, neighbor.Id, neighborTarget.Id),
            GenerationPath: $"R4:analogy:{element.Id}~{neighbor.Id}→{neighborTarget.Id}",
            IsHypothesis: true
        );
        
        newElements.Add(analogyElement);
        return new TickResult(analogyElement, true, "R4 analogy succeeded");
    }
    
    /// <summary>R5: Close gap A→M→C by inferring A→C</summary>
    private static TickResult ExecuteR5_Gap(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        int[]? secondaryIds,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (secondaryIds == null || secondaryIds.Length < 2)
            return new TickResult(null, false, "R5 needs 2 secondary IDs");
        
        var a = FindElementById(elements, secondaryIds[0]);
        var c = FindElementById(elements, secondaryIds[1]);
        
        if (a == null || c == null)
            return new TickResult(null, false, "Gap endpoints not found");
        
        // Close gap via direct relation
        var gapEmbedding = InterpolateEmbeddings(a.Embedding, c.Embedding, state.EmbeddingDimension);
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            new[] { a, c },
            noveltyScore: 0.7,
            fallback: 0.4);
        
        var gapRelation = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Relation,
            Embedding: gapEmbedding,
            Symbol: $"Gap({a.Symbol}→{c.Symbol})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.7,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.Create(a.Id, element.Id, c.Id),
            GenerationPath: $"R5:gap:{a.Id}→{element.Id}→{c.Id}"
        );
        
        newElements.Add(gapRelation);
        return new TickResult(gapRelation, true, "R5 gap closed");
    }
    
    /// <summary>R6: Update local transform for function</summary>
    private static TickResult ExecuteR6_LocalLearn(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        string? parameter,
        double[]? auxiliaryEmbedding,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (element.Kind != ElementKind.Function || auxiliaryEmbedding == null)
            return new TickResult(null, false, "R6 requires function element and embedding");
        
        // Update LocalTransformDelta via running average
        var delta = auxiliaryEmbedding;  // This should be computed as output - input
        
        var updatedElement = element with
        {
            LocalTransformDelta = delta,
            LocalTransformConfidence = Math.Min(1.0, element.LocalTransformConfidence + 0.1),
            LocalTransformObservations = element.LocalTransformObservations + 1,
            BridgeConfidence = Clamp01((element.BridgeConfidence * 0.85) + (Math.Min(1.0, element.LocalTransformConfidence + 0.1) * 0.15))
        };
        
        // Note: This modifies in place via PlatonicState update pattern
        return new TickResult(updatedElement, true, "R6 local learn updated");
    }
    
    /// <summary>R7: Split agent at branch point</summary>
    private static TickResult ExecuteR7_BranchDetect(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (element.Kind != ElementKind.Function)
            return new TickResult(null, false, "R7 requires function element");
        
        // Create child agent for specialized handling
        var childElement = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Function,
            Embedding: element.Embedding.ToArray(),
            Symbol: $"{element.Symbol}_branch",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.6,
            BridgeConfidence: Clamp01((element.BridgeConfidence * 0.9) + 0.05),
            RelatedTo: ImmutableArray.Create(element.Id),
            GenerationPath: $"R7:branch:{element.Id}",
            DevolvedParentId: element.Id
        );
        
        newElements.Add(childElement);
        
        // Mark parent as having devolved
        var updatedParent = element with
        {
            IsDevolved = true,
            DevolvedChildIds = ImmutableArray.Create(state.NextId),
            BridgeConfidence = Clamp01(element.BridgeConfidence * 0.95)
        };
        
        return new TickResult(updatedParent, true, "R7 branch created");
    }
    
    /// <summary>R8: Discover fold chain pattern</summary>
    private static TickResult ExecuteR8_FoldChain(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        int[]? secondaryIds,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        if (secondaryIds == null || secondaryIds.Length < 1)
            return new TickResult(null, false, "R8 needs chain elements");
        
        // Compose chain elements into a fold pattern
        var chainElements = secondaryIds
            .Select(id => FindElementById(elements, id))
            .Where(e => e != null)
            .Cast<PlatonicElement>()
            .ToList();
        
        if (chainElements.Count < 1)
            return new TickResult(null, false, "Chain elements not found");
        
        var foldEmbedding = SumEmbeddings(chainElements, state.EmbeddingDimension);
        var bridgeConfidence = ComputeDerivedBridgeConfidence(
            element,
            chainElements,
            noveltyScore: 0.85,
            fallback: 0.4);
        
        var foldElement = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Composition,
            Embedding: foldEmbedding,
            Symbol: $"Fold({string.Join("→", chainElements.Select(c => c.Symbol))})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.85,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.CreateRange(chainElements.Select(c => c.Id)),
            GenerationPath: $"R8:fold:{string.Join(",", chainElements.Select(c => c.Id))}"
        );
        
        newElements.Add(foldElement);
        return new TickResult(foldElement, true, "R8 fold chain discovered");
    }

    /// <summary>R9: Invoke a first-class space-management tool as a learned tick action.</summary>
    private static TickResult ExecuteR9_SpaceTool(
        PlatonicElement element,
        ImmutableArray<PlatonicElement> elements,
        PlatonicState state,
        string? parameter,
        ImmutableArray<PlatonicElement>.Builder newElements)
    {
        var toolName = string.IsNullOrWhiteSpace(parameter)
            ? "observe"
            : parameter.Trim().ToLowerInvariant();
        var bridgeConfidence = Clamp01((element.BridgeConfidence * 0.85) + 0.1);

        var toolElement = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Function,
            Embedding: element.Embedding.ToArray(),
            Symbol: $"SM({toolName})",
            GeneratedAtTick: state.CurrentTick,
            ComplementId: null,
            NoveltyScore: 0.55,
            BridgeConfidence: bridgeConfidence,
            RelatedTo: ImmutableArray.Create(element.Id),
            GenerationPath: $"R9:space-tool:{toolName}",
            IsHypothesis: true);

        newElements.Add(toolElement);
        return new TickResult(toolElement, true, $"R9 space tool {toolName}");
    }
    
    // Helper methods
    
    private static PlatonicElement? FindElementById(ImmutableArray<PlatonicElement> elements, int id)
        => elements.FirstOrDefault(e => e.Id == id);
    
    private static PlatonicElement? FindNearestByKind(ImmutableArray<PlatonicElement> elements, PlatonicElement element)
        => elements
            .Where(e => e.Kind == element.Kind && e.Id != element.Id)
            .OrderBy(e => EuclideanDistance(e.Embedding, element.Embedding))
            .FirstOrDefault();

    private static double ComputeDerivedBridgeConfidence(
        PlatonicElement primary,
        IEnumerable<PlatonicElement> related,
        double noveltyScore,
        double fallback)
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
     
    private static double[] InterpolateEmbeddings(double[] a, double[] b, int dim)
    {
        var result = new double[dim];
        for (int d = 0; d < dim; d++)
            result[d] = (a[d] + b[d]) / 2.0;
        return result;
    }
    
    private static double[] SumEmbeddings(List<PlatonicElement> elements, int dim)
    {
        var result = new double[dim];
        foreach (var elem in elements)
            for (int d = 0; d < dim; d++)
                result[d] += elem.Embedding[d];
        return result;
    }
    
    private static double[] ComputeMeanEmbedding(List<PlatonicElement> elements, int dim)
    {
        var sum = SumEmbeddings(elements, dim);
        var count = Math.Max(1, elements.Count);
        for (int d = 0; d < dim; d++)
            sum[d] /= count;
        return sum;
    }
    
    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double Clamp01(double value)
        => Math.Max(0.0, Math.Min(1.0, value));
}
