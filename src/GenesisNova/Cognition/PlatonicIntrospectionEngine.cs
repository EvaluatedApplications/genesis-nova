using GenesisNova.Data;

namespace GenesisNova.Cognition;

public sealed partial class PlatonicIntrospectionEngine
{
    private readonly PlatonicSpaceMemory _memory;
    private readonly int _queueCapacity;
    private readonly List<QueuedEvent> _queue = [];
    private readonly Dictionary<string, double> _noveltyWeights = new(StringComparer.OrdinalIgnoreCase);

    public PlatonicIntrospectionEngine(PlatonicSpaceMemory memory, int queueCapacity = 1024)
    {
        _memory = memory;
        _queueCapacity = Math.Max(128, queueCapacity);
    }

    public int QueueSize => _queue.Count;

    public void QueueTrainingExample(GenesisExample example)
    {
        var concepts = ExtractConcepts($"{example.Input} {example.Output}");
        var eventBundle = new NoveltyEventSnapshot(
            Kind: "training",
            Input: example.Input,
            Output: example.Output,
            RouteId: example.RouteLabel,
            Confidence: 1.0,
            CreatedAtUtc: DateTime.UtcNow,
            Concepts: concepts,
            NoveltyScore: 0.0);
        Enqueue(eventBundle);
    }

    public void QueueInference(string input, string output, int routeId, double confidence)
    {
        var concepts = ExtractConcepts($"{input} {output}");
        var eventBundle = new NoveltyEventSnapshot(
            Kind: "inference",
            Input: input,
            Output: output,
            RouteId: routeId,
            Confidence: confidence,
            CreatedAtUtc: DateTime.UtcNow,
            Concepts: concepts,
            NoveltyScore: 0.0);
        Enqueue(eventBundle);
    }

    public int RunCycles(int maxCycles)
    {
        var cycles = Math.Max(0, maxCycles);
        var processed = 0;

        for (var i = 0; i < cycles; i++)
        {
            if (_queue.Count == 0)
                break;

            var next = DequeueHighestNovelty();
            var yield = ProcessEvent(next.Event);
            LearnNovelty(next.Event, yield);
            processed++;
        }

        return processed;
    }

    public double EstimateConsistencyLoss(IReadOnlyList<string> concepts)
    {
        if (concepts.Count < 2)
            return 0.0;

        var score = 0.0;
        var count = 0;
        for (var i = 0; i < concepts.Count; i++)
        {
            for (var j = i + 1; j < concepts.Count; j++)
            {
                score += _memory.GetContradiction(concepts[i], concepts[j]);
                count++;
            }
        }

        return count == 0 ? 0.0 : score / count;
    }

    public string DescribeConcept(string concept) => _memory.DescribeConcept(concept);

    public void ObserveDirectContradiction(string left, string right, double contradiction)
        => _memory.ObserveContradiction(left, right, contradiction);

    public PlatonicCognitionSnapshot ExportSnapshot()
    {
        var events = _queue
            .OrderByDescending(x => x.NoveltyScore)
            .Select(x => x.Event with { NoveltyScore = x.NoveltyScore })
            .ToArray();

        return new PlatonicCognitionSnapshot(
            Memory: _memory.ExportSnapshot(),
            Queue: new PlatonicQueueSnapshot(
                Capacity: _queueCapacity,
                Events: events,
                Learner: new NoveltyLearnerSnapshot(_noveltyWeights.ToDictionary(k => k.Key, v => v.Value, StringComparer.OrdinalIgnoreCase))));
    }

    public void ImportSnapshot(PlatonicCognitionSnapshot snapshot)
    {
        _memory.ImportSnapshot(snapshot.Memory);

        _queue.Clear();
        foreach (var item in snapshot.Queue.Events)
            _queue.Add(new QueuedEvent(item with { NoveltyScore = 0.0 }, item.NoveltyScore));

        _noveltyWeights.Clear();
        foreach (var pair in snapshot.Queue.Learner.ConceptWeights)
            _noveltyWeights[pair.Key] = pair.Value;
    }

    private void Enqueue(NoveltyEventSnapshot eventBundle)
    {
        var novelty = PredictNovelty(eventBundle);
        _queue.Add(new QueuedEvent(eventBundle, novelty));

        if (_queue.Count <= _queueCapacity)
            return;

        // Capacity guardrail: drop lowest novelty event.
        var minIndex = 0;
        var minScore = _queue[0].NoveltyScore;
        for (var i = 1; i < _queue.Count; i++)
        {
            if (_queue[i].NoveltyScore >= minScore)
                continue;
            minScore = _queue[i].NoveltyScore;
            minIndex = i;
        }
        _queue.RemoveAt(minIndex);
    }

    private QueuedEvent DequeueHighestNovelty()
    {
        var bestIndex = 0;
        var bestScore = _queue[0].NoveltyScore;
        for (var i = 1; i < _queue.Count; i++)
        {
            if (_queue[i].NoveltyScore <= bestScore)
                continue;
            bestScore = _queue[i].NoveltyScore;
            bestIndex = i;
        }

        var chosen = _queue[bestIndex];
        _queue.RemoveAt(bestIndex);
        return chosen;
    }

    private double ProcessEvent(NoveltyEventSnapshot eventBundle)
    {
        var concepts = eventBundle.Concepts.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        if (concepts.Length < 2)
            return 0.05;

        var baseContradiction = InferContradictionSignal(eventBundle);
        var energyBefore = EstimateConsistencyLoss(concepts);

        for (var i = 0; i < concepts.Length; i++)
        {
            for (var j = i + 1; j < concepts.Length; j++)
            {
                var observed = Clamp01(baseContradiction + PairBias(concepts[i], concepts[j]));
                _memory.ObserveContradiction(concepts[i], concepts[j], observed);
            }
        }

        var energyAfter = EstimateConsistencyLoss(concepts);
        var resolutionYield = Math.Abs(energyBefore - energyAfter);
        var structureYield = Math.Min(1.0, concepts.Length / 10.0);
        return Clamp01((0.7 * resolutionYield) + (0.3 * structureYield));
    }

    private double PredictNovelty(NoveltyEventSnapshot eventBundle)
    {
        var unseen = eventBundle.Concepts.Count(c => !_memory.ContainsConcept(c));
        var baseNovelty = 0.2 + (0.6 * (unseen / (double)Math.Max(1, eventBundle.Concepts.Length)));

        var learned = 0.0;
        if (eventBundle.Concepts.Length > 0)
        {
            learned = eventBundle.Concepts
                .Select(c => _noveltyWeights.TryGetValue(c, out var w) ? w : 0.0)
                .Average();
        }

        return Clamp01(baseNovelty + (0.3 * learned));
    }

    private void LearnNovelty(NoveltyEventSnapshot eventBundle, double introspectionYield)
    {
        foreach (var concept in eventBundle.Concepts)
        {
            _noveltyWeights.TryGetValue(concept, out var current);
            _noveltyWeights[concept] = (0.95 * current) + (0.05 * introspectionYield);
        }
    }

    private static double InferContradictionSignal(NoveltyEventSnapshot eventBundle)
    {
        var text = $"{eventBundle.Input} {eventBundle.Output}".ToLowerInvariant();
        if (text.Contains("opposite", StringComparison.Ordinal) ||
            text.Contains("different", StringComparison.Ordinal) ||
            text.Contains("not ", StringComparison.Ordinal))
            return 0.85;

        if (text.Contains("same", StringComparison.Ordinal) ||
            text.Contains("similar", StringComparison.Ordinal) ||
            text.Contains("like", StringComparison.Ordinal))
            return 0.2;

        return 0.45;
    }

    private static double PairBias(string left, string right)
    {
        var l = left.ToLowerInvariant();
        var r = right.ToLowerInvariant();
        var feline = new[] { "cat", "tiger", "lion", "kitten" };
        var canine = new[] { "dog", "wolf", "puppy" };

        var lFeline = feline.Contains(l, StringComparer.OrdinalIgnoreCase);
        var rFeline = feline.Contains(r, StringComparer.OrdinalIgnoreCase);
        var lCanine = canine.Contains(l, StringComparer.OrdinalIgnoreCase);
        var rCanine = canine.Contains(r, StringComparer.OrdinalIgnoreCase);

        if ((lFeline && rFeline) || (lCanine && rCanine))
            return -0.35;
        if ((lFeline && rCanine) || (lCanine && rFeline))
            return 0.25;
        return 0.0;
    }

    private static string[] ExtractConcepts(string text)
    {
        var concepts = text.ToLowerInvariant()
            .Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(token => new string(token.Where(char.IsLetter).ToArray()))
            .Where(w => w.Length >= 3 && !StopWords.Contains(w))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(12)
            .ToArray();

        return concepts.Length == 0
            ? ["unknown"]
            : concepts;
    }

    private static double Clamp01(double value)
        => Math.Max(0.0, Math.Min(1.0, value));

    private sealed record QueuedEvent(NoveltyEventSnapshot Event, double NoveltyScore);

    private static readonly HashSet<string> StopWords =
    [
        "the", "and", "for", "with", "that", "this", "from", "into", "what", "when", "where",
        "your", "you", "are", "was", "were", "have", "has", "had", "will", "would", "could",
        "should", "about", "them", "they", "then", "than", "there", "here"
    ];
}
