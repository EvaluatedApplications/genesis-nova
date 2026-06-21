namespace GenesisNova.Train;

/// <summary>
/// REAL natural-language facts for the gym's language skills — TRUE knowledge, not throwaway nonce noise, so what
/// the model learns stays correct and transferable (a synonym of "big" really is "large"; an apple really is a
/// fruit). This is curriculum DATA (like <see cref="GenesisNova.Core.NumberWordVocabulary"/>), not engine logic —
/// the engine stays general. Public so BOTH <see cref="GymTrainer"/> and GenesisInspect's gymprobe draw from the
/// same source and cannot drift.
/// </summary>
public static class GymLanguageFacts
{
    /// <summary>Synonym GROUPS: every word is a valid synonym of every other in its group (symmetric). A query for
    /// any word accepts ANY OTHER member as correct (multiple right answers). True facts → permanent, never
    /// contradictory across levels.</summary>
    public static readonly string[][] SynonymGroups =
    {
        new[] { "big", "large", "huge", "giant" },
        new[] { "small", "little", "tiny" },
        new[] { "happy", "glad", "cheerful" },
        new[] { "sad", "unhappy", "gloomy" },
        new[] { "fast", "quick", "rapid", "swift" },
        new[] { "slow", "sluggish" },
        new[] { "smart", "clever", "bright" },
        new[] { "angry", "mad", "furious" },
        new[] { "begin", "start" },
        new[] { "end", "finish" },
        new[] { "cold", "chilly" },
        new[] { "quiet", "silent" },
        new[] { "loud", "noisy" },
        new[] { "easy", "simple" },
        new[] { "hard", "difficult", "tough" },
        new[] { "strong", "powerful", "mighty" },
    };

    /// <summary>Category MEMBERSHIP: item → its true category. Many items per category (a real many-to-one).
    /// Each word appears once, and no item/category word collides with a synonym word or a number-word.</summary>
    public static readonly (string Item, string Category)[] Categories =
    {
        ("apple", "fruit"), ("banana", "fruit"), ("grape", "fruit"), ("pear", "fruit"),
        ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"),
        ("dog", "animal"), ("cat", "animal"), ("horse", "animal"), ("cow", "animal"),
        ("robin", "bird"), ("eagle", "bird"), ("sparrow", "bird"),
        ("red", "color"), ("blue", "color"), ("green", "color"),
        ("iron", "metal"), ("gold", "metal"), ("silver", "metal"),
        ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"),
        ("chair", "furniture"), ("table", "furniture"), ("bed", "furniture"),
    };
}
