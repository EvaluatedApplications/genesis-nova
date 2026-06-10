using GenesisNova.Data;
using System.Text;

namespace GenesisNova.Train;

public static class GenesisTrainingDataLoader
{
    public static IReadOnlyList<GenesisExample> LoadFromFile(string path)
        => LoadFromFileAsync(path).GetAwaiter().GetResult();

    public static async Task<IReadOnlyList<GenesisExample>> LoadFromFileAsync(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Training file not found: {path}", path);

        var output = new List<GenesisExample>();
        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            bufferSize: 64 * 1024,
            options: FileOptions.Asynchronous | FileOptions.SequentialScan);
        using var reader = new StreamReader(stream);

        var lineNumber = 0;
        while (true)
        {
            var raw = await reader.ReadLineAsync();
            if (raw is null)
                break;

            lineNumber++;
            raw = raw.Trim();
            if (raw.Length == 0 || raw.StartsWith('#'))
                continue;

            output.Add(ParseLine(raw, lineNumber));
        }

        return output;
    }

    public static void SaveToFile(string path, IReadOnlyList<GenesisExample> examples)
        => SaveToFileAsync(path, examples).GetAwaiter().GetResult();

    public static async Task SaveToFileAsync(string path, IReadOnlyList<GenesisExample> examples)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        await using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            bufferSize: 64 * 1024,
            options: FileOptions.Asynchronous);
        await using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        foreach (var ex in examples)
        {
            // Save only input => output (no route labels)
            await writer.WriteLineAsync($"{ex.Input} => {ex.Output}");
        }

        await writer.FlushAsync();
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
