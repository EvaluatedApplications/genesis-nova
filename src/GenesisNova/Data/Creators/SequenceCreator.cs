using System.Collections.Immutable;
using System.Globalization;

namespace GenesisNova.Data.Creators;

public sealed class SequenceCreator : IExampleCreator
{
    private const int StepSize = 20;

    public string Name => "sequence:next";
    public int EstimatedComplexity => 24;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var safeDifficulty = Math.Max(0, difficulty);
        var prompts = new[]
        {
            "next in sequence: {0}",
            "continue: {0}",
            "what comes next: {0}"
        };

        var sequences = BuildSequences(safeDifficulty);
        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (display, next) = sequences[i % sequences.Length];
            var prompt = string.Format(CultureInfo.InvariantCulture, prompts[i % prompts.Length], display);
            examples.Add((prompt, next));
        }

        return examples.ToImmutable();
    }

    private static (string Display, string Next)[] BuildSequences(int difficulty)
    {
        var rng = new Random(StableHash(nameof(SequenceCreator), difficulty));
        var minLength = 3;
        var maxLength = Math.Min(6, 3 + difficulty);
        var maxStep = Math.Max(2, 2 + difficulty);
        var maxStart = Math.Max(8, 8 + (difficulty * 3));

        var result = new (string Display, string Next)[StepSize];
        for (var i = 0; i < StepSize; i++)
        {
            var length = rng.Next(minLength, maxLength + 1);
            var step = rng.Next(1, maxStep + 1);
            var start = rng.Next(0, maxStart + 1);

            var values = new int[length];
            for (var j = 0; j < length; j++)
                values[j] = start + (j * step);

            var display = string.Join(", ", values);
            var next = (start + (length * step)).ToString(CultureInfo.InvariantCulture);
            result[i] = (display, next);
        }

        return result;
    }

    private static int StableHash(string source, int extra)
    {
        uint h = 2166136261u;
        foreach (var c in source)
        {
            h ^= c;
            h *= 16777619u;
        }

        h ^= (uint)extra;
        h *= 16777619u;
        return (int)h;
    }
}
