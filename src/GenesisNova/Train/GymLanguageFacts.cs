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
        new[] { "rich", "wealthy" },
        new[] { "poor", "broke" },
        new[] { "old", "ancient", "aged" },
        new[] { "new", "modern", "fresh" },
        new[] { "pretty", "beautiful", "lovely" },
        new[] { "ugly", "hideous" },
        new[] { "brave", "bold", "fearless" },
        new[] { "afraid", "scared", "fearful" },
        new[] { "funny", "amusing", "hilarious" },
        new[] { "boring", "dull" },
        new[] { "clean", "spotless" },
        new[] { "dirty", "filthy" },
        new[] { "wet", "damp", "soaked" },
        new[] { "dry", "parched" },
        new[] { "empty", "vacant" },
        new[] { "full", "packed" },
        new[] { "thin", "slim", "slender" },
        new[] { "wide", "broad" },
        new[] { "shiny", "gleaming", "glossy" },
        new[] { "dark", "dim", "murky" },
        new[] { "tired", "weary", "exhausted" },
        new[] { "calm", "peaceful", "serene" },
        new[] { "gentle", "caring", "tender" }, // NOT "kind" — it doubles as the category framing word ("what KIND of thing")
        new[] { "cruel", "harsh", "mean" },
        new[] { "wise", "sage" },
        new[] { "true", "correct", "accurate" },
        new[] { "false", "wrong", "incorrect" },
        new[] { "near", "close", "nearby" },
        new[] { "far", "distant", "remote" },
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
