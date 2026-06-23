namespace GenesisNova.Data.Creators;

/// <summary>
/// Shared deterministic text/data helpers for example creators. Extracted verbatim from the byte-identical
/// copies that lived in individual creators — same logic, same constants, same results.
/// </summary>
internal static class CreatorText
{
    /// <summary>
    /// Cartesian template expansion: for each (token → values) replacement, fan every template that contains
    /// the token out across its values (templates without the token pass through unchanged), then de-duplicate
    /// ordinally. Byte-identical to the former <c>ArithmeticCreator.ExpandSynonyms</c> / <c>LanguageCreator.ExpandSynonyms</c>.
    /// </summary>
    internal static string[] ExpandSynonyms(string[] templates, IReadOnlyDictionary<string, string[]> replacements)
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

    /// <summary>
    /// Deterministic FNV-1a 32-bit hash of <paramref name="source"/> mixed with <paramref name="extra"/>, used
    /// to seed creators' RNGs reproducibly. Byte-identical to the former <c>ArithmeticCreator.StableHash</c> /
    /// <c>SequenceCreator.StableHash</c> (the SHA256-based <c>StableHash32</c> in PublicTextCorpusCreator is a
    /// different algorithm and is intentionally NOT consolidated here).
    /// </summary>
    internal static int StableHash(string source, int extra)
    {
        uint h = 2166136261u;
        foreach (char c in source) { h ^= (uint)c; h *= 16777619u; }
        h ^= (uint)extra; h *= 16777619u;
        return (int)h;
    }

    /// <summary>
    /// Shared item → category reference table (24 ordered pairs: four members each of fruit/animal/color/
    /// vehicle/instrument/tree). Reference data — the NN still LEARNS the mapping from the emitted pairs. Used
    /// by RelationCreator and CategoryRetrievalCreator (each applies its OWN level-slicing over this table).
    /// </summary>
    internal static readonly (string Item, string Category)[] ItemCategories =
    [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"),
        ("dog", "animal"), ("cat", "animal"), ("wolf", "animal"), ("bear", "animal"),
        ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
        ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"), ("boat", "vehicle"),
        ("piano", "instrument"), ("drum", "instrument"), ("violin", "instrument"), ("flute", "instrument"),
        ("oak", "tree"), ("pine", "tree"), ("cedar", "tree"), ("maple", "tree"),
    ];
}
