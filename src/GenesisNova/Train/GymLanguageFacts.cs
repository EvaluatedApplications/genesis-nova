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
    // Every group has >= 4 members: a 2-member cluster has no geometric MASS to form a distinct attractor, so its
    // members scatter and retrieval bleeds into a wrong-cluster synonym ("big" -> "chilly"). More true synonyms per
    // group = each cluster pulls a clean, separable region — the project's own BuildPool lesson (members per category).
    // Words are unique across groups (a cross-group word would be its own contradiction in the competing-answer set);
    // none collide with a category item, a number-word, or the "kind" category framing word.
    public static readonly string[][] SynonymGroups =
    {
        new[] { "big", "large", "huge", "giant", "enormous" },
        new[] { "small", "little", "tiny", "miniature", "petite" },
        new[] { "happy", "glad", "cheerful", "joyful", "pleased" },
        new[] { "sad", "unhappy", "gloomy", "miserable", "downcast" },
        new[] { "fast", "quick", "rapid", "swift", "speedy" },
        new[] { "slow", "sluggish", "leisurely", "unhurried" },
        new[] { "smart", "clever", "bright", "intelligent", "sharp" },
        new[] { "angry", "mad", "furious", "irate", "livid" },
        new[] { "begin", "start", "commence", "initiate" },
        new[] { "end", "finish", "conclude", "complete" },
        new[] { "cold", "chilly", "freezing", "frigid", "icy" },
        new[] { "quiet", "silent", "hushed", "soundless" },
        new[] { "loud", "noisy", "deafening", "blaring" },
        new[] { "easy", "simple", "effortless", "painless" },
        new[] { "hard", "difficult", "tough", "challenging" },
        new[] { "strong", "powerful", "mighty", "sturdy" },
        new[] { "rich", "wealthy", "affluent", "prosperous" },
        new[] { "poor", "broke", "penniless", "needy" },
        new[] { "old", "ancient", "aged", "elderly" },
        new[] { "new", "modern", "fresh", "recent" },
        new[] { "pretty", "beautiful", "lovely", "gorgeous" },
        new[] { "ugly", "hideous", "unsightly", "grotesque" },
        new[] { "brave", "bold", "fearless", "courageous" },
        new[] { "afraid", "scared", "fearful", "frightened" },
        new[] { "funny", "amusing", "hilarious", "comical" },
        new[] { "boring", "dull", "tedious", "tiresome" },
        new[] { "clean", "spotless", "immaculate", "pristine" },
        new[] { "dirty", "filthy", "grimy", "muddy" },
        new[] { "wet", "damp", "soaked", "moist" },
        new[] { "dry", "parched", "arid", "dehydrated" },
        new[] { "empty", "vacant", "bare", "hollow" },
        new[] { "full", "packed", "crammed", "stuffed" },
        new[] { "thin", "slim", "slender", "lean" },
        new[] { "wide", "broad", "expansive", "vast" },
        new[] { "shiny", "gleaming", "glossy", "polished" },
        new[] { "dark", "dim", "murky", "shadowy" },
        new[] { "tired", "weary", "exhausted", "drained" },
        new[] { "calm", "peaceful", "serene", "tranquil" },
        new[] { "gentle", "caring", "tender", "mild" }, // NOT "kind" — it doubles as the category framing word ("what KIND of thing")
        new[] { "cruel", "harsh", "mean", "brutal" },
        new[] { "wise", "sage", "insightful", "prudent" },
        new[] { "true", "correct", "accurate", "truthful" },
        new[] { "false", "wrong", "incorrect", "untrue" },
        new[] { "near", "close", "nearby", "adjacent" },
        new[] { "far", "distant", "remote", "faraway" },
    };

    /// <summary>Category MEMBERSHIP: item → its true category. Many items per category (a real many-to-one).
    /// Each word appears once, and no item/category word collides with a synonym word or a number-word.</summary>
    public static readonly (string Item, string Category)[] Categories =
    {
        ("apple", "fruit"), ("banana", "fruit"), ("grape", "fruit"), ("pear", "fruit"), ("cherry", "fruit"), ("peach", "fruit"), ("plum", "fruit"), ("lemon", "fruit"), ("mango", "fruit"),
        ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"), ("pea", "vegetable"), ("corn", "vegetable"), ("celery", "vegetable"),
        ("dog", "animal"), ("cat", "animal"), ("horse", "animal"), ("cow", "animal"), ("pig", "animal"), ("sheep", "animal"), ("goat", "animal"), ("lion", "animal"), ("tiger", "animal"), ("bear", "animal"), ("wolf", "animal"), ("fox", "animal"), ("rabbit", "animal"), ("deer", "animal"),
        ("robin", "bird"), ("eagle", "bird"), ("sparrow", "bird"), ("owl", "bird"), ("hawk", "bird"), ("crow", "bird"), ("duck", "bird"), ("goose", "bird"), ("swan", "bird"),
        ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"), ("purple", "color"), ("orange", "color"), ("pink", "color"), ("brown", "color"), ("black", "color"), ("white", "color"),
        ("iron", "metal"), ("gold", "metal"), ("silver", "metal"), ("copper", "metal"), ("tin", "metal"), ("zinc", "metal"),
        ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"), ("plane", "vehicle"), ("boat", "vehicle"), ("ship", "vehicle"), ("van", "vehicle"),
        ("chair", "furniture"), ("table", "furniture"), ("bed", "furniture"), ("desk", "furniture"), ("sofa", "furniture"), ("shelf", "furniture"), ("stool", "furniture"),
        ("piano", "instrument"), ("guitar", "instrument"), ("drum", "instrument"), ("violin", "instrument"), ("flute", "instrument"), ("trumpet", "instrument"),
        ("hammer", "tool"), ("saw", "tool"), ("drill", "tool"), ("wrench", "tool"), ("screwdriver", "tool"),
        ("shirt", "clothing"), ("hat", "clothing"), ("coat", "clothing"), ("sock", "clothing"), ("glove", "clothing"), ("scarf", "clothing"),
        ("house", "building"), ("school", "building"), ("church", "building"), ("tower", "building"), ("barn", "building"),
        ("water", "drink"), ("juice", "drink"), ("milk", "drink"), ("tea", "drink"), ("coffee", "drink"),
        ("arm", "body"), ("leg", "body"), ("hand", "body"), ("foot", "body"), ("head", "body"), ("nose", "body"), ("ear", "body"),
    };
}
