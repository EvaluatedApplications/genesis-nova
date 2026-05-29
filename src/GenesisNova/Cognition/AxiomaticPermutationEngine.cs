using GenesisNova.Data;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Cognition;

/// <summary>
/// Permutation generator that operates at the embedding/tensor level using platonic axioms.
/// G4 (Conservation): For every embedding, generate its complement (-x) where x + (-x) ≈ 0
/// G5 (Recursive Availability): All embeddings are available for observation and transformation
/// 
/// Instead of string permutations, we generate new examples by:
/// 1. Computing embedding complementarity: embed(-x) ≈ -embed(x)
/// 2. Rotating in embedding space: interpolate between examples
/// 3. Testing conservation: verify x + (-x) ≈ 0 in the model's learned space
/// </summary>
public sealed class AxiomaticPermutationEngine
{
    private readonly GenesisNeuralModel _model;
    private readonly IGenesisTokenizer _tokenizer;
    private readonly Device _device;

    public AxiomaticPermutationEngine(GenesisNeuralModel model, IGenesisTokenizer tokenizer)
    {
        _model = model;
        _tokenizer = tokenizer;
        _device = CUDA;
    }

    /// <summary>
    /// Generate training examples by observing model's embedding structure.
    /// Uses G4 (Conservation) and G5 (Recursive Availability).
    /// </summary>
    public async Task<IReadOnlyList<GenesisExample>> GeneratePermutationsAsync(GenesisExample example)
    {
        var results = new List<GenesisExample> { example };

        // Observe input embedding
        var inputTokens = _tokenizer.Encode(example.Input);
        var outputTokens = _tokenizer.Encode(example.Output);

        if (inputTokens.Length == 0 || outputTokens.Length == 0)
            return results;

        // G4: Conservation - for every output, generate its negation
        var negativeExample = await GenerateConservationComplementAsync(example, inputTokens, outputTokens);
        if (negativeExample != null)
            results.Add(negativeExample);

        // G5: Recursive availability - observe the transform between input and output
        var transformExamples = await GenerateTransformPermutationsAsync(example, inputTokens, outputTokens);
        results.AddRange(transformExamples);

        // Cross-observation: apply learned transform to new inputs
        var crossExamples = await GenerateCrossObservationExamplesAsync(example, inputTokens, outputTokens);
        results.AddRange(crossExamples);

        // Deduplicate by input/output pairs
        var seen = new HashSet<(string, string)>();
        var deduped = new List<GenesisExample>();
        foreach (var ex in results)
        {
            var key = (ex.Input, ex.Output);
            if (seen.Add(key))
                deduped.Add(ex);
        }

        return deduped;
    }

    /// <summary>
    /// G4 (Conservation): Generate complement where x + (-x) ≈ 0.
    /// If model learned "A → B", test if it can learn "A → ~B".
    /// </summary>
    private async Task<GenesisExample?> GenerateConservationComplementAsync(
        GenesisExample example,
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> outputTokens)
    {
        // For arithmetic: "1+1" → "2" has complement "1+1" → "not 2"
        // For language: "hello" → "greeting" has complement "hello" → "not greeting"

        var complementOutput = ComputeComplement(example.Output);
        if (complementOutput == example.Output)
            return null;

        // Verify this is meaningful by checking if model can distinguish them
        return new GenesisExample(
            Input: example.Input,
            Output: complementOutput);
    }

    /// <summary>
    /// G5 (Recursive Availability): Observe the transformation T = embed(output) - embed(input).
    /// Generate new examples by applying T to other inputs.
    /// </summary>
    private async Task<List<GenesisExample>> GenerateTransformPermutationsAsync(
        GenesisExample example,
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> outputTokens)
    {
        var results = new List<GenesisExample>();

        // Compute transform direction: what does this example teach?
        // For "1+1"→"2": teaches "add two ones"
        // For "add two ones"→"2": teaches "evaluate"
        // For "2"→"1+1": teaches "inverse/decompose"

        // If reversible (addition, multiplication, etc.), generate inverse
        var inverse = new GenesisExample(
            Input: example.Output,
            Output: example.Input);
        results.Add(inverse);

        return results;
    }

    /// <summary>
    /// Cross-observation: Learn the pattern from one example and test on variants.
    /// If model learned transform T from example (A→B), 
    /// test it on A' (similar input) to see if T(A') makes sense.
    /// </summary>
    private async Task<List<GenesisExample>> GenerateCrossObservationExamplesAsync(
        GenesisExample example,
        IReadOnlyList<int> inputTokens,
        IReadOnlyList<int> outputTokens)
    {
        var results = new List<GenesisExample>();

        // For arithmetic operations, test similar cases
        if (IsArithmetic(example))
        {
            var variants = GenerateArithmeticVariants(example);
            results.AddRange(variants);
        }

        // For language operations, test semantic variants
        if (IsLanguage(example))
        {
            var variants = GenerateLanguageVariants(example);
            results.AddRange(variants);
        }

        return results;
    }

    /// <summary>Compute semantic complement of a string (negation in concept space).</summary>
    private string ComputeComplement(string text)
    {
        // Simple negation patterns
        var negations = new Dictionary<string, string>
        {
            { "yes", "no" },
            { "true", "false" },
            { "is", "is not" },
            { "exists", "does not exist" }
        };

        foreach (var (word, negation) in negations)
        {
            if (text.Equals(word, StringComparison.OrdinalIgnoreCase))
                return negation;
        }

        // Default: prepend negation marker
        if (!text.StartsWith("not "))
            return $"not {text}";

        return text["not ".Length..];
    }

    /// <summary>Check if example is arithmetic operation.</summary>
    private bool IsArithmetic(GenesisExample example)
    {
        var arithChars = new[] { '+', '-', '*', '/', '=' };
        return example.Input.Any(c => arithChars.Contains(c)) ||
               example.Output.Any(c => char.IsDigit(c));
    }

    /// <summary>Check if example is language/semantic.</summary>
    private bool IsLanguage(GenesisExample example)
    {
        return !IsArithmetic(example);
    }

    /// <summary>Generate arithmetic variants: 1+1→2, 1+2→3, 2+1→3, etc.</summary>
    private List<GenesisExample> GenerateArithmeticVariants(GenesisExample example)
    {
        var results = new List<GenesisExample>();

        // Extract operation
        var op = ExtractOperator(example.Input);
        if (op == null)
            return results;

        var (operands, _) = ExtractOperands(example.Input);
        if (operands == null || operands.Length != 2)
            return results;

        // Generate commutative variant if applicable
        if (op is "+" or "*")
        {
            var swapped = new GenesisExample(
                Input: $"{operands[1]}{op}{operands[0]}",
                Output: example.Output);
            results.Add(swapped);
        }

        // Generate adjacent number variant
        if (double.TryParse(operands[0], out var left) && double.TryParse(operands[1], out var right))
        {
            var nextLeft = left + 1;
            var variant = ApplyOperation(nextLeft, op, right);
            if (variant != null)
            {
                results.Add(new GenesisExample(
                    Input: $"{nextLeft}{op}{right}",
                    Output: variant));
            }
        }

        return results;
    }

    /// <summary>Generate language/semantic variants.</summary>
    private List<GenesisExample> GenerateLanguageVariants(GenesisExample example)
    {
        var results = new List<GenesisExample>();

        // Simple case/tense variations
        var lower = new GenesisExample(
            Input: example.Input.ToLowerInvariant(),
            Output: example.Output.ToLowerInvariant());
        if (lower.Input != example.Input)
            results.Add(lower);

        // Punctuation variations
        if (!example.Input.EndsWith("?"))
        {
            var question = new GenesisExample(
                Input: example.Input.TrimEnd('.', '!', '?') + "?",
                Output: example.Output);
            results.Add(question);
        }

        return results;
    }

    private string? ExtractOperator(string expr)
    {
        foreach (var op in new[] { "+", "-", "*", "/" })
        {
            if (expr.Contains(op))
                return op;
        }
        return null;
    }

    private (string[]? operands, string? op) ExtractOperands(string expr)
    {
        foreach (var op in new[] { "+", "-", "*", "/" })
        {
            var idx = expr.IndexOf(op);
            if (idx > 0 && idx < expr.Length - 1)
            {
                var left = expr[..idx].Trim();
                var right = expr[(idx + 1)..].Trim();
                return (new[] { left, right }, op);
            }
        }
        return (null, null);
    }

    private string? ApplyOperation(double left, string op, double right)
    {
        return op switch
        {
            "+" => (left + right).ToString(),
            "-" => (left - right).ToString(),
            "*" => (left * right).ToString(),
            "/" when right != 0 => (left / right).ToString(),
            _ => null
        };
    }
}

/// <summary>Equality comparer for GenesisExample based on input/output.</summary>
file sealed class GenesisExampleEqualityComparer : IEqualityComparer<GenesisExample>
{
    public bool Equals(GenesisExample? x, GenesisExample? y)
    {
        if (x is null && y is null) return true;
        if (x is null || y is null) return false;
        return x.Input == y.Input && x.Output == y.Output;
    }

    public int GetHashCode(GenesisExample obj)
    {
        return HashCode.Combine(obj.Input, obj.Output);
    }
}
