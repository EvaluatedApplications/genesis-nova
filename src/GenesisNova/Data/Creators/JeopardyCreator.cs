using System.Collections.Immutable;
using System.Text.Json;

namespace GenesisNova.Data.Creators;

public sealed class JeopardyCreator : IExampleCreator
{
    private const int StepSize = 5;

    public string Name => "jeopardy:trivia";
    public int EstimatedComplexity => 30;

    private static readonly Lazy<IReadOnlyList<(string question, string answer)>> Pool = new(LoadPool);

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var pool = Pool.Value;
        if (pool.Count == 0)
            return ImmutableArray<(string, string)>.Empty;

        var start = Math.Min(Math.Max(0, difficulty) * StepSize, pool.Count);
        var end = Math.Min((Math.Max(0, difficulty) + 1) * StepSize, pool.Count);
        var slice = pool.Skip(start).Take(end - start).ToArray();
        if (slice.Length == 0)
            slice = pool.Take(Math.Min(StepSize, pool.Count)).ToArray();

        return Enumerable.Range(0, count).Select(i => slice[i % slice.Length]).ToImmutableArray();
    }

    private static IReadOnlyList<(string question, string answer)> LoadPool()
    {
        var file = FindDataFile();
        if (file is null)
        {
            return
            [
                ("Capital of France", "paris"),
                ("Largest planet in our solar system", "jupiter"),
                ("Fastest land animal", "cheetah"),
                ("H2O is commonly known as", "water"),
                ("The red planet", "mars"),
                ("Author of Hamlet", "shakespeare"),
                ("Square root of 81", "9"),
                ("First month of the year", "january"),
                ("Primary gas humans breathe in", "oxygen"),
                ("Instrument with six strings often used in rock", "guitar")
            ];
        }

        using var stream = File.OpenRead(file);
        var raw = JsonSerializer.Deserialize<JeopardyEntry[]>(stream) ?? [];

        return raw
            .Where(e => IsCleanAnswer(e.answer))
            .Select(e => (CleanQuestion(e.question), CleanAnswer(e.answer)))
            .Distinct()
            .Take(2000)
            .ToArray();
    }

    private static string? FindDataFile()
    {
        var dir = AppDomain.CurrentDomain.BaseDirectory;
        for (var i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "Assets", "JEOPARDY_QUESTIONS1.json");
            if (File.Exists(candidate))
                return candidate;

            candidate = Path.Combine(dir, "TestExamples", "Assets", "JEOPARDY_QUESTIONS1.json");
            if (File.Exists(candidate))
                return candidate;

            var parent = Directory.GetParent(dir);
            if (parent is null)
                break;
            dir = parent.FullName;
        }

        return null;
    }

    private static bool IsCleanAnswer(string answer)
    {
        if (string.IsNullOrWhiteSpace(answer))
            return false;

        if (answer.Contains('<') || answer.Contains('>') || answer.Contains('\\'))
            return false;

        return answer.Length <= 40;
    }

    private static string CleanQuestion(string value)
    {
        var question = value.Trim();
        if (question.StartsWith('\'') && question.EndsWith('\'') && question.Length > 1)
            question = question[1..^1].Trim();
        return question;
    }

    private static string CleanAnswer(string value) => value.Trim().ToLowerInvariant();

    private sealed record JeopardyEntry(
        string category,
        string air_date,
        string question,
        string? value,
        string answer,
        string round,
        string show_number);
}
