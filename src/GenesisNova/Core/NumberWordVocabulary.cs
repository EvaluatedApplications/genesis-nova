using System.Collections.Generic;

namespace GenesisNova.Core;

/// <summary>
/// The canonical number ↔ word REFERENCE TABLE (ground truth), shared by the number-word training
/// creator and the answer-equivalence grader. This is reference data (like the arithmetic that computes
/// a creator's true answer), NOT a heuristic baked into the model — the NN still LEARNS the equivalence
/// relationally from training pairs. Centralised so the creator and the grader can never drift.
/// </summary>
public static class NumberWordVocabulary
{
    public static readonly (int Value, string Word)[] Entries =
    [
        (0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"),
        (5, "five"), (6, "six"), (7, "seven"), (8, "eight"), (9, "nine"),
        (10, "ten"), (11, "eleven"), (12, "twelve"), (13, "thirteen"), (14, "fourteen"),
        (15, "fifteen"), (16, "sixteen"), (17, "seventeen"), (18, "eighteen"), (19, "nineteen"),
        (20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"),
        (60, "sixty"), (70, "seventy"), (80, "eighty"), (90, "ninety"),
    ];

    private static readonly Dictionary<string, int> _wordToValue = BuildWordToValue();

    public static IReadOnlyDictionary<string, int> WordToValue => _wordToValue;

    private static Dictionary<string, int> BuildWordToValue()
    {
        var map = new Dictionary<string, int>(System.StringComparer.OrdinalIgnoreCase);
        foreach (var (value, word) in Entries)
            map[word] = value;
        return map;
    }
}
