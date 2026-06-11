using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

/// <summary>
/// Language creator: Q&A pairs for semantic understanding.
/// All examples share a single cluster so they train into one transform.
/// </summary>
public sealed class LanguageCreator : IExampleCreator
{
    private readonly string _name;
    private readonly (string Q, string A)[] _pairs;

    public LanguageCreator(string name, (string, string)[] pairs)
    {
        _name = name;
        _pairs = pairs;
    }

    public string Name => _name;
    public int EstimatedComplexity => 25;
    public GenesisTrainingExampleKind TrainingKind => GenesisTrainingExampleKind.PromptAnswer;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        if (_pairs.Length == 0)
            return ImmutableArray<(string, string)>.Empty;

        var examples = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        for (var i = 0; i < count; i++)
        {
            var (q, a) = _pairs[i % _pairs.Length];
            examples.Add((NormalizeQuestion(q), a));
        }

        return examples.ToImmutable();
    }

    private static string NormalizeQuestion(string question)
        => question.Trim();

    private static string[] ExpandSynonyms(string[] templates, IReadOnlyDictionary<string, string[]> replacements)
    {
        var expanded = templates.ToList();
        foreach (var (token, values) in replacements)
        {
            if (values.Length == 0)
                continue;

            var next = new List<string>(expanded.Count * values.Length);
            foreach (var template in expanded)
            {
                if (!template.Contains(token, StringComparison.Ordinal))
                {
                    next.Add(template);
                    continue;
                }

                foreach (var value in values)
                    next.Add(template.Replace(token, value, StringComparison.Ordinal));
            }
            expanded = next;
        }

        return expanded.Distinct(StringComparer.Ordinal).ToArray();
    }
}

/// <summary>Pre-built language creators with common Q&A pairs.</summary>
public static class LanguageDefaults
{
    public static readonly LanguageCreator Greet = new("language:greet", new[]
    {
        ("hello", "hello!"),
        ("hi", "hi there!"),
        ("hey", "hey!"),
        ("good morning", "good morning!"),
        ("good afternoon", "good afternoon!"),
        ("how are you", "I'm doing well, thanks!"),
        ("how are you doing", "doing well, thanks!"),
        ("nice to meet you", "nice to meet you too!"),
        ("welcome", "thank you, glad to be here!"),
        ("greetings", "greetings!"),
    });

    public static readonly LanguageCreator Acknowledge = new("language:acknowledge", new[]
    {
        ("ok", "ok!"),
        ("yes", "yes!"),
        ("sure", "sure!"),
        ("got it", "got it!"),
        ("understood", "understood!"),
        ("thank you", "you're welcome!"),
        ("thanks", "you're welcome!"),
        ("sorry", "no problem!"),
        ("excuse me", "yes?"),
        ("never mind", "ok!"),
    });

    public static readonly LanguageCreator Facts = new("language:facts", new[]
    {
        ("what is the capital of france", "Paris."),
        ("what is the capital of england", "London."),
        ("what is the capital of germany", "Berlin."),
        ("what is the capital of japan", "Tokyo."),
        ("what is the capital of spain", "Madrid."),
        ("what is the sun", "The sun is a star."),
        ("what is water", "Water is H2O."),
        ("what is light", "Light is electromagnetic radiation."),
        ("what is gravity", "Gravity is a force that attracts mass."),
        ("what is fire", "Fire is rapid oxidation releasing heat and light."),
        ("how many days in a week", "Seven."),
        ("how many months in a year", "Twelve."),
        ("how many hours in a day", "Twenty-four."),
    });

    public static readonly LanguageCreator Commands = new("language:commands", new[]
    {
        ("help", "I can answer questions and solve problems."),
        ("start", "ready!"),
        ("stop", "ok."),
        ("go", "going!"),
        ("wait", "waiting."),
        ("think", "thinking..."),
        ("calculate", "provide an expression."),
        ("learn", "I'm always learning."),
        ("repeat", "I'll repeat that."),
        ("listen", "I'm listening."),
    });
}
