namespace RetentionBench;

// A LARGE, procedurally-generated fact universe — the antidote to toy datasets. Facts are (entity -> category) bindings
// over NONCE tokens (so nothing is memorised from pre-training and the space must genuinely learn each binding). The
// universe is partitioned into WAVES by introduction order: the curriculum trains wave 0, then 1, ... and we measure
// whether the EARLY waves survive as later ones flood in (catastrophic forgetting vs relevance-decay retention).
//
// Scale knobs: numCategories * membersPerCategory = total facts. At membersPerCategory~50, numCategories~400 => 20k
// facts; push to fill/exceed the space's 100k-node cap to study retention UNDER eviction pressure (the real regime).
public sealed class ProceduralKnowledge
{
    public readonly (string Entity, string Category)[] Facts;
    public readonly string[] Categories;
    public readonly int WaveSize;

    public ProceduralKnowledge(int numCategories, int membersPerCategory, int waveSize, int seed)
    {
        Categories = new string[numCategories];
        for (var c = 0; c < numCategories; c++) Categories[c] = Nonce(c);          // category labels: indices [0, numCategories)

        var facts = new List<(string, string)>(numCategories * membersPerCategory);
        var e = 0;
        for (var c = 0; c < numCategories; c++)
            for (var m = 0; m < membersPerCategory; m++)
                facts.Add((Nonce(numCategories + e++), Categories[c]));            // entities: disjoint index range => never collide with labels

        // Deterministic shuffle so each wave mixes many categories (category hubs grow gradually, as in real streams).
        var rng = new Random(seed);
        for (var i = facts.Count - 1; i > 0; i--) { var j = rng.Next(i + 1); (facts[i], facts[j]) = (facts[j], facts[i]); }
        Facts = facts.ToArray();
        WaveSize = Math.Max(1, waveSize);
    }

    public int WaveCount => (Facts.Length + WaveSize - 1) / WaveSize;

    public ArraySegment<(string Entity, string Category)> Wave(int w)
    {
        var start = w * WaveSize;
        if (start >= Facts.Length) return ArraySegment<(string, string)>.Empty;
        return new ArraySegment<(string, string)>(Facts, start, Math.Min(WaveSize, Facts.Length - start));
    }

    // Injective nonce token from an index: 4 consonant-vowel syllables => 70^4 = 24.0M distinct word-like tokens
    // (e.g. 0 -> "babababa"-ish). Deterministic, so the universe is reproducible across runs/resumes.
    public static string Nonce(long i)
    {
        const string C = "bdfgklmnprstvz"; // 14
        const string V = "aeiou";          // 5
        var sb = new System.Text.StringBuilder(8);
        for (var s = 0; s < 4; s++)
        {
            sb.Append(C[(int)(i % 14)]); i /= 14;
            sb.Append(V[(int)(i % 5)]); i /= 5;
        }
        return sb.ToString();
    }
}
