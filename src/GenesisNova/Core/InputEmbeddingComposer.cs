using System.Globalization;

namespace GenesisNova.Core;

internal static class InputEmbeddingComposer
{
    public static double[] GetInputEmbedding(string input, int dim)
    {
        if (string.IsNullOrWhiteSpace(input))
            return new double[dim];

        var tokens = input.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var maxInlineSlots = ComputeMaxInlineSlots(dim);
        return tokens.Length <= maxInlineSlots
            ? GetCharComposedEmbedding(input, dim)
            : GetMeanPoolEmbedding(input, dim);
    }

    public static double[] GetMeanPoolEmbedding(string sentence, int dim)
    {
        var tokens = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (tokens.Length == 0)
            return new double[dim];

        var acc = new double[dim];
        foreach (var token in tokens)
        {
            var tokenEmbedding = double.TryParse(token, NumberStyles.Any, CultureInfo.InvariantCulture, out var numeric)
                ? GetFreshNumericEmbedding(numeric, dim)
                : GetCharComposedEmbedding(token, dim);

            for (var d = 0; d < dim; d++)
                acc[d] += tokenEmbedding[d];
        }

        var inv = 1.0 / tokens.Length;
        for (var d = 0; d < dim; d++)
            acc[d] *= inv;

        return acc;
    }

    public static double[] GetFreshNumericEmbedding(double value, int dim)
    {
        var embedding = new double[dim];
        var numericDims = Math.Min(dim / 2, 21);
        if (numericDims <= 0)
            return embedding;

        for (var i = 0; i < numericDims; i++)
            embedding[i] = value * Math.Pow(10.0, -(i + 1));

        var logStart = numericDims;
        var logDims = Math.Min(numericDims, dim - logStart);
        if (Math.Abs(value) > 1e-12)
        {
            var logValue = Math.Log(Math.Abs(value));
            for (var i = 0; i < logDims; i++)
                embedding[logStart + i] = logValue * Math.Pow(10.0, -(i + 1));
        }

        return embedding;
    }

    public static int CharFaceStart(int dim) => Math.Min(Math.Min(dim / 2, 21) * 2, dim);

    public static bool IsMultiArg(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return false;
        var tokens = input.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return tokens.Length >= 2;
    }

    public static double[] ComposeInput(string input, CompositionMode mode, int dim)
    {
        if (!IsMultiArg(input))
            return GetInputEmbedding(input, dim);

        var args = SplitArgs(input);
        if (args.Length == 0)
            return GetInputEmbedding(input, dim);

        var argEmbeddings = args.Select(a => GetInputEmbedding(a, dim)).ToArray();
        return mode switch
        {
            CompositionMode.Sum => ComposeSum(argEmbeddings, dim),
            CompositionMode.Product => ComposeProduct(argEmbeddings, dim),
            CompositionMode.Difference => ComposeDifference(argEmbeddings, dim),
            CompositionMode.Concatenate => ComposeConcatenate(argEmbeddings, dim),
            _ => ComposeSum(argEmbeddings, dim)
        };
    }

    private static double[] GetCharComposedEmbedding(string text, int dim)
    {
        if (double.TryParse(text, NumberStyles.Any, CultureInfo.InvariantCulture, out var numeric))
            return GetFreshNumericEmbedding(numeric, dim);

        var embedding = new double[dim];
        if (text.Length == 0 || dim == 0)
            return embedding;

        var normalized = text.ToLowerInvariant();
        var charStart = Math.Min(CharFaceStart(dim), Math.Max(0, dim - 1));
        var charDims = Math.Max(1, dim - charStart);

        for (var i = 0; i < normalized.Length; i++)
        {
            var c = normalized[i];
            var code = c;
            var weight = 1.0 / Math.Sqrt(i + 1.0);
            var slot = Math.Abs(((code * 31) + (i * 17)) % charDims);
            embedding[charStart + slot] += weight;

            // Add weak global signal so non-char faces still get consistent context.
            var global = Math.Abs((code + i) % dim);
            embedding[global] += weight * 0.05;
        }

        var norm = Math.Sqrt(embedding.Sum(v => v * v));
        if (norm > 1e-9)
        {
            for (var i = 0; i < dim; i++)
                embedding[i] /= norm;
        }

        return embedding;
    }

    private static int ComputeMaxInlineSlots(int dim)
    {
        var charStart = CharFaceStart(dim);
        var charDims = Math.Max(4, dim - charStart);
        return Math.Max(2, charDims / 4);
    }

    private static string[] SplitArgs(string input)
    {
        var parts = input
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(t => !IsOperatorToken(t))
            .ToArray();
        return parts.Length > 0 ? parts : new[] { input };
    }

    private static bool IsOperatorToken(string token)
    {
        var t = token.Trim().ToLowerInvariant();
        return t is "+" or "-" or "*" or "/" or "x" or "plus" or "minus" or "times" or "multiply" or "divide";
    }

    private static double[] ComposeSum(double[][] args, int dim)
    {
        var result = new double[dim];
        if (args.Length == 0)
            return result;
        foreach (var emb in args)
        {
            for (var i = 0; i < dim; i++)
                result[i] += emb[i];
        }
        var inv = 1.0 / args.Length;
        for (var i = 0; i < dim; i++)
            result[i] *= inv;
        return result;
    }

    private static double[] ComposeProduct(double[][] args, int dim)
    {
        var result = Enumerable.Repeat(1.0, dim).ToArray();
        if (args.Length == 0)
            return new double[dim];
        foreach (var emb in args)
        {
            for (var i = 0; i < dim; i++)
                result[i] *= emb[i];
        }
        return result;
    }

    private static double[] ComposeDifference(double[][] args, int dim)
    {
        if (args.Length == 0)
            return new double[dim];
        var result = args[0].ToArray();
        for (var a = 1; a < args.Length; a++)
        {
            for (var i = 0; i < dim; i++)
                result[i] -= args[a][i];
        }
        return result;
    }

    private static double[] ComposeConcatenate(double[][] args, int dim)
    {
        if (args.Length == 0)
            return new double[dim];
        var result = new double[dim];
        var segment = Math.Max(1, dim / args.Length);
        for (var a = 0; a < args.Length; a++)
        {
            var start = a * segment;
            if (start >= dim)
                break;
            var end = Math.Min(dim, start + segment);
            var emb = args[a];
            for (var i = start; i < end; i++)
                result[i] = emb[i];
        }
        return result;
    }
}
