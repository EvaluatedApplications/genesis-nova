using GenesisNova.Cognition;

namespace GenesisNova.Core;

/// <summary>
/// Detects whether a multi-argument function follows a Sum, Difference, or Concatenate composition.
/// Uses algebraic symmetry detection on output values to avoid circular embedding bias.
/// Ported from genesis-engine CompositionDetector.
/// </summary>
public class CompositionDetector
{
    /// <summary>
    /// Composition mode for multi-argument operations.
    /// </summary>
    public enum CompositionMode
    {
        Sum,        // a + b = c (commutative)
        Difference, // a - b = c (order-sensitive)
        Concatenate // Neither — preserve both operands independently
    }

    /// <summary>
    /// Attempts to detect the composition mode from training examples.
    /// Tests algebraic symmetry on raw output values first (unbiased).
    /// Falls back to embedding accuracy only if symmetry test is inconclusive.
    /// </summary>
    public CompositionMode? DetectCompositionFromExamples(
        IReadOnlyList<(string Input, string Output)> examples)
    {
        if (examples.Count < 3)
            return null;

        // Extract numeric pairs
        var nfi = System.Globalization.NumberStyles.Any;
        var inv = System.Globalization.CultureInfo.InvariantCulture;

        double sumResidual = 0, diffResidual = 0;
        int linearFitCount = 0;

        foreach (var (inp, outp) in examples)
        {
            var args = ExtractNumericArgs(inp);
            if (args.Length < 2) continue;
            
            double a = args[0], b = args[1];
            if (!double.TryParse(outp, nfi, inv, out var c)) 
                continue;

            // Test Sum hypothesis: c ≈ a + b
            sumResidual += (c - (a + b)) * (c - (a + b));
            
            // Test Difference hypothesis: c ≈ a - b or c ≈ b - a
            double r1 = (c - (a - b)) * (c - (a - b));
            double r2 = (c - (b - a)) * (c - (b - a));
            diffResidual += Math.Min(r1, r2);
            
            linearFitCount++;
        }

        if (linearFitCount < 3)
            return null;

        double bestResidualPerPair = Math.Min(sumResidual, diffResidual) / linearFitCount;

        // If the better hypothesis has low absolute error, trust the ratio test
        if (bestResidualPerPair < 0.5)
        {
            double ratio = (sumResidual + 1e-15) / (diffResidual + 1e-15);
            if (ratio > 10.0) return CompositionMode.Difference;
            if (ratio < 0.1) return CompositionMode.Sum;
        }

        // High residual or ambiguous ratio → non-linear operation
        return CompositionMode.Concatenate;
    }

    /// <summary>
    /// Extracts numeric arguments from input string.
    /// E.g., "1+2" → [1, 2], "5x3" → [5, 3]
    /// </summary>
    private static double[] ExtractNumericArgs(string input)
    {
        var nfi = System.Globalization.NumberStyles.Any;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        
        var args = new List<double>();
        var current = string.Empty;

        foreach (char c in input)
        {
            if (char.IsDigit(c) || c == '.' || c == '-')
            {
                current += c;
            }
            else if (!string.IsNullOrEmpty(current))
            {
                if (double.TryParse(current, nfi, inv, out var val))
                    args.Add(val);
                current = string.Empty;
            }
        }

        if (!string.IsNullOrEmpty(current))
        {
            if (double.TryParse(current, nfi, inv, out var val))
                args.Add(val);
        }

        return args.ToArray();
    }
}
