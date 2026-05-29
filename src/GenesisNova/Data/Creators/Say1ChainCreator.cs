using System.Collections.Immutable;

namespace GenesisNova.Data.Creators;

public sealed class Say1ChainCreator : IExampleCreator
{
    private static readonly string[] AllWords =
    [
        "go", "ok", "yes", "hi", "no", "stop", "hey", "wow",
        "yep", "nah", "cool", "fine", "sure", "good", "nice",
        "well", "done", "next", "back", "here", "on", "when",
        "where", "why", "who", "what", "how", "this", "that"
    ];

    public string Name => "say1:chain";
    public int EstimatedComplexity => 15;

    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        var chainLength = Math.Max(1, difficulty + 1);
        var prompt = chainLength == 1 ? "say 1 word" : $"say {chainLength} words";
        var phrase = string.Join(' ', Enumerable.Range(0, chainLength).Select(step => WordFor(chainLength, step)));

        return Enumerable.Range(0, count).Select(_ => (prompt, phrase)).ToImmutableArray();
    }

    private static string WordFor(int chainLength, int step)
    {
        var sliceStart = chainLength * (chainLength - 1) / 2;
        return AllWords[(sliceStart + step) % AllWords.Length];
    }
}
