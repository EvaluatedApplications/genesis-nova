namespace GenesisNova.Core;

/// <summary>
/// Thin façade over <see cref="PlatonicFaceComposer"/> for the inference/training call sites.
/// Face composition (clean char slots, dedicated word face, homomorphic numeric faces, seeded
/// free dims) lives in the canonical composer so input embeddings and stored concept faces
/// share one geometry. This type only adds the multi-argument composition modes used when
/// building transform inputs.
/// </summary>
internal static class InputEmbeddingComposer
{
    public static double[] GetInputEmbedding(string input, int dim)
        => PlatonicFaceComposer.Compose(input, dim);

    public static double[] GetFreshNumericEmbedding(double value, int dim)
        => PlatonicFaceComposer.GetFreshNumericEmbedding(value, dim);

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

    private static string[] SplitArgs(string input)
    {
        var parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 ? parts : new[] { input };
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
        if (args.Length == 0)
            return new double[dim];
        var result = Enumerable.Repeat(1.0, dim).ToArray();
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
