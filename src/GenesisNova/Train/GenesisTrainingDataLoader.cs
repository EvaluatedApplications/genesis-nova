using GenesisNova.Data;

namespace GenesisNova.Train;

public static class GenesisTrainingDataLoader
{
    public static IReadOnlyList<GenesisExample> LoadFromFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Training file not found: {path}", path);

        var lines = File.ReadAllLines(path);
        var output = new List<GenesisExample>(lines.Length);
        for (var i = 0; i < lines.Length; i++)
        {
            var raw = lines[i].Trim();
            if (raw.Length == 0 || raw.StartsWith('#'))
                continue;

            output.Add(ParseLine(raw, i + 1));
        }

        return output;
    }

    public static void SaveToFile(string path, IReadOnlyList<GenesisExample> examples)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        var lines = new List<string>();
        foreach (var ex in examples)
        {
            // Save only input => output (no route labels)
            string line = $"{ex.Input} => {ex.Output}";
            lines.Add(line);
        }
        File.WriteAllLines(path, lines);
    }

    private static GenesisExample ParseLine(string line, int lineNumber)
    {
        var arrow = line.IndexOf("=>", StringComparison.Ordinal);
        if (arrow < 1 || arrow >= line.Length - 2)
            throw new FormatException($"Invalid line {lineNumber}: expected '<input> => <output> [| route=N]'.");

        var input = line[..arrow].Trim();
        var right = line[(arrow + 2)..].Trim();

        var pipe = right.IndexOf('|');
        if (pipe < 0)
            return new GenesisExample(input, right);

        var output = right[..pipe].Trim();
        // Note: route metadata is parsed but no longer used (routes are inferred by model)
        
        return new GenesisExample(input, output);
    }
}

