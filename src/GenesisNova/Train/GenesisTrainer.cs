using GenesisNova.Axioms;
using GenesisNova.Cognition;
using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Train;

public sealed class GenesisTrainer
{
    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly GenesisCompositeObjective _objective;
    private readonly PlatonicIntrospectionEngine? _cognition;
    private int _trainStepCount;
    private double _cachedConservationLoss;

    public GenesisTrainer(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicIntrospectionEngine? cognition = null,
        GenesisCompositeObjective? objective = null)
    {
        _tokenizer = tokenizer;
        _model = model;
        _cognition = cognition;
        _objective = objective ?? new GenesisCompositeObjective(
            TokenWeight: 1.0,
            RouteWeight: 0.3,
            ConsistencyWeight: 0.1,
            ConservationWeight: 0.1,
            MemoryWeight: 0.05);
    }

    public int[] EncodeInput(string input)
        => _tokenizer.Encode(input);

    public int[] EncodeTarget(string output)
        => _tokenizer.Encode(output, addEos: true);

    public GenesisStepLoss TrainStepPreTokenized(int[] inputTokens, int[] targetTokens, int? routeLabel)
    {
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var baseLoss = _model.TrainExample(inputTokens, targetTokens, _tokenizer.BosTokenId, routeLabel);
        _cognition?.QueueTrainingExample(new GenesisExample(_tokenizer.Decode(inputTokens), _tokenizer.Decode(targetTokens), routeLabel));

        var concepts = ExtractConcepts(_tokenizer.Decode(inputTokens), _tokenizer.Decode(targetTokens));
        var consistencyLoss = _cognition?.EstimateConsistencyLoss(concepts) ?? 0.0;
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = _cognition is null
            ? 0.0
            : Math.Min(1.0, _cognition.QueueSize / 1000.0);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            0.0,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            0.0,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }

    public GenesisStepLoss TrainStep(GenesisExample example)
    {
        var inputTokens = _tokenizer.Encode(example.Input);
        var targetTokens = _tokenizer.Encode(example.Output, addEos: true);
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var baseLoss = _model.TrainExample(inputTokens, targetTokens, _tokenizer.BosTokenId, example.RouteLabel);
        _cognition?.QueueTrainingExample(example);

        var concepts = ExtractConcepts(example);
        var consistencyLoss = _cognition?.EstimateConsistencyLoss(concepts) ?? 0.0;
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = _cognition is null
            ? 0.0
            : Math.Min(1.0, _cognition.QueueSize / 1000.0);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            0.0,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            0.0,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }

    public int RunIntrospectionCycles(int cycles)
        => _cognition?.RunCycles(cycles) ?? 0;

    public void CloneParametersToBreakGraph()
        => _model.CloneParametersToBreakGraph();

    /// <summary>
    /// Train a batch of examples without cloning between them (cloning happens at epoch boundary).
    /// </summary>
    public GenesisStepLoss TrainBatchPreTokenized(
       IReadOnlyList<(int[] InputTokens, int[] TargetTokens)> batch)
    {
       if (batch.Count == 0)
           return new GenesisStepLoss(0, 0, 0, 0, 0, 0);

       _model.EnsureVocabularySize(_tokenizer.VocabularySize);

       var totalTokenLoss = 0.0;
       var totalTokens = 0;

       // Train each example without cloning between them
       foreach (var (inputTokens, targetTokens) in batch)
       {
           var loss = _model.TrainExample(inputTokens, targetTokens, _tokenizer.BosTokenId, null);
           totalTokenLoss += loss.TokenLoss * targetTokens.Length;
           totalTokens += targetTokens.Length;
            
           _trainStepCount++;
           _cognition?.QueueTrainingExample(
               new GenesisExample(_tokenizer.Decode(inputTokens), _tokenizer.Decode(targetTokens), null));
       }

       var avgTokenLoss = totalTokenLoss / Math.Max(1, totalTokens);

       return new GenesisStepLoss(
           avgTokenLoss,
           0.0,
           0.0,
           EstimateConservationDrift(batch[0].InputTokens),
           0.0,
           avgTokenLoss);
    }

    public string DescribeConcept(string concept)
       => _cognition?.DescribeConcept(concept) ?? "cognition engine not enabled";

    public void ObserveDirectContradiction(string left, string right, double contradiction)
        => _cognition?.ObserveDirectContradiction(left, right, contradiction);

    public int QueueSize => _cognition?.QueueSize ?? 0;

    public PlatonicCognitionSnapshot? ExportCognitionSnapshot()
        => _cognition?.ExportSnapshot();

    public void ImportCognitionSnapshot(PlatonicCognitionSnapshot snapshot)
        => _cognition?.ImportSnapshot(snapshot);

    private double EstimateConservationDrift(IReadOnlyList<int> inputTokens)
    {
        if (_trainStepCount % 128 != 0 && _cachedConservationLoss > 0)
            return _cachedConservationLoss;

        var snapshot = _model.Export();
        var emb = snapshot.Embeddings;
        if (emb.Length == 0)
            return 0.0;

        var rows = emb.GetLength(0);
        var cols = emb.GetLength(1);

        var sum = 0.0;
        var count = 0;
        foreach (var token in inputTokens.Distinct())
        {
            if (token < 0 || token >= rows)
                continue;
            for (var h = 0; h < cols; h++)
            {
                sum += Math.Abs(emb[token, h]);
                count++;
            }
        }

        if (count == 0)
        {
            var sampleRows = Math.Min(16, rows);
            for (var i = 0; i < sampleRows; i++)
            {
                var row = (i * 997 + _trainStepCount) % rows;
                for (var h = 0; h < cols; h++)
                {
                    sum += Math.Abs(emb[row, h]);
                    count++;
                }
            }
        }

        _cachedConservationLoss = (sum / Math.Max(1, count)) * 0.001;
        return _cachedConservationLoss;
    }

    private static IReadOnlyList<string> ExtractConcepts(string input, string output)
    {
        var words = $"{input} {output}"
            .ToLowerInvariant()
            .Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(w => new string(w.Where(char.IsLetter).ToArray()))
            .Where(w => w.Length >= 3)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(12)
            .ToArray();

        return words.Length == 0 ? ["unknown"] : words;
    }

    private static IReadOnlyList<string> ExtractConcepts(GenesisExample example)
    {
        return ExtractConcepts(example.Input, example.Output);
    }
}

public sealed record GenesisStepLoss(
    double TokenLoss,
    double RouteLoss,
    double ConsistencyLoss,
    double ConservationLoss,
    double MemoryLoss,
    double TotalLoss);
