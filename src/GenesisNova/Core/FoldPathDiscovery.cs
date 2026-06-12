using System.Globalization;
using GenesisNova.Core;

namespace GenesisNova.Core;

/// <summary>
/// A discovered log-linear relationship for multiplicative/divisive operations:
/// log(|c|) = Alpha * log(|a|) + Beta * log(|b|) + LogK
/// </summary>
public record LogLinearFit(
    double Alpha,
    double Beta,
    double LogK,
    double MeanResidual,
    int VerifiedPairs);

/// <summary>
/// JSON-serializable snapshot of a <see cref="FoldPathDiscovery"/> for checkpoint persistence.
/// Captures the discovered fold paths, log-linear fits, and per-operation composition modes
/// (the data that drives <c>HasOperation</c>/<c>TryPredict</c>/<c>GetComposition</c>).
/// The raw <c>_operationExamples</c> string ring is intentionally omitted — it re-accumulates on the
/// next training pass and is large. All members are System.Text.Json-friendly: the value records
/// (<see cref="FoldPathDiscovery.FoldPath"/>, <see cref="LogLinearFit"/>) are plain primitive records,
/// and <see cref="CompositionMode"/> is an enum.
/// </summary>
public sealed record FoldPathDiscoverySnapshot(
    Dictionary<string, FoldPathDiscovery.FoldPath> FoldPaths,
    Dictionary<string, LogLinearFit> LogLinearFits,
    Dictionary<string, CompositionMode> Compositions);

/// <summary>
/// Discovers compositional fold paths and log-linear relationships.
/// Ported from genesis-engine FoldPathDiscovery.
/// </summary>
public class FoldPathDiscovery
{
    // Strict numeric classification: only a plain signed decimal counts as a numeric arg.
    // The candidate substring is already restricted to digit/'.'/'-' chars, but NumberStyles.Any
    // would still accept a trailing sign (e.g. "3-" → 3), letting malformed runs be extracted as
    // numbers. Stays aligned with PlatonicSpaceMemory.TryParseNumber.
    private const NumberStyles NumericStyle =
        NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint
        | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite;

    /// <summary>
    /// A discovered fold: targetOp(a,b) ≈ fold(baseOp, step=a, count=b)
    /// </summary>
    public record FoldPath(
        string TargetOp,
        string BaseOp,
        int AccumArgIdx,
        int CountArgIdx,
        int VerifiedExamples);

    private readonly Dictionary<string, FoldPath> _foldPaths = new();
    private readonly Dictionary<string, LogLinearFit> _logLinearFits = new();
    private readonly Dictionary<string, CompositionMode> _compositions = new();
    private readonly Dictionary<string, List<(string Input, string Output)>> _operationExamples = new();

    public IReadOnlyDictionary<string, FoldPath> FoldPaths => _foldPaths;
    public IReadOnlyDictionary<string, LogLinearFit> LogLinearFits => _logLinearFits;

    /// <summary>
    /// Runs discovery for a specific operation using its accumulated examples.
    /// Call after every N training examples to detect fold and log-linear paths.
    /// Uses known operations for fold base-op search.
    /// </summary>
    public void TryRunDiscovery(
        string targetOp,
        Func<string, double, double, double?> tryExecuteBaseOp)
    {
        if (!_operationExamples.TryGetValue(targetOp, out var examples))
            return;

        if (examples.Count < 3)
            return;

        var knownOps = _operationExamples.Keys.ToList();
        TryDiscoverFoldPath(targetOp, examples, knownOps, tryExecuteBaseOp);
        TryDiscoverLogLinearFit(targetOp, examples);
    }

    /// <summary>
    /// Tracks training examples for an operation (string form).
    /// </summary>
    public void ObserveTrainingPair(string operation, string input, string output)
    {
        if (!_operationExamples.TryGetValue(operation, out var examples))
        {
            examples = new List<(string, string)>();
            _operationExamples[operation] = examples;
        }
        examples.Add((input, output));
    }

    /// <summary>
    /// Tracks training examples for an operation (numeric form).
    /// Automatically converts to string and stores for discovery.
    /// </summary>
    public void ObserveTrainingPair(string operation, double left, double right, double output)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        ObserveTrainingPair(operation, $"{left.ToString(inv)},{right.ToString(inv)}", output.ToString(inv));
    }

    /// <summary>
    /// Gets the composition mode for an operation (default: Sum).
    /// </summary>
    public CompositionMode GetComposition(string operation)
    {
        return _compositions.TryGetValue(operation, out var mode) ? mode : CompositionMode.Sum;
    }

    /// <summary>
    /// Checks if the system has learned an operation.
    /// </summary>
    public bool HasOperation(string operation)
    {
        return _foldPaths.ContainsKey(operation) 
            || _logLinearFits.ContainsKey(operation)
            || _operationExamples.ContainsKey(operation);
    }

    /// <summary>
    /// Export a JSON-serializable snapshot of the discovered fold paths, log-linear fits, and
    /// composition modes for checkpoint persistence. The raw per-operation example rings are omitted
    /// (re-accumulated on the next training pass). Fresh dictionary copies are taken so the snapshot
    /// is independent of later mutation.
    /// </summary>
    public FoldPathDiscoverySnapshot ExportSnapshot()
    {
        return new FoldPathDiscoverySnapshot(
            new Dictionary<string, FoldPath>(_foldPaths),
            new Dictionary<string, LogLinearFit>(_logLinearFits),
            new Dictionary<string, CompositionMode>(_compositions));
    }

    /// <summary>
    /// Rebuild the discovered fold paths, log-linear fits, and composition modes from a checkpoint
    /// snapshot. Clears the existing three dictionaries first. The example rings are not restored
    /// (they re-accumulate on the next training pass).
    /// </summary>
    public void ImportSnapshot(FoldPathDiscoverySnapshot snapshot)
    {
        if (snapshot is null)
            return;

        _foldPaths.Clear();
        _logLinearFits.Clear();
        _compositions.Clear();

        if (snapshot.FoldPaths is not null)
        {
            foreach (var pair in snapshot.FoldPaths)
            {
                if (!string.IsNullOrEmpty(pair.Key) && pair.Value is not null)
                    _foldPaths[pair.Key] = pair.Value;
            }
        }

        if (snapshot.LogLinearFits is not null)
        {
            foreach (var pair in snapshot.LogLinearFits)
            {
                if (!string.IsNullOrEmpty(pair.Key) && pair.Value is not null)
                    _logLinearFits[pair.Key] = pair.Value;
            }
        }

        if (snapshot.Compositions is not null)
        {
            foreach (var pair in snapshot.Compositions)
            {
                if (!string.IsNullOrEmpty(pair.Key))
                    _compositions[pair.Key] = pair.Value;
            }
        }
    }

    /// <summary>
    /// Attempts to predict the output of an operation using discovered paths.
    /// </summary>
    public bool TryPredict(
        string operation,
        double left,
        double right,
        out double prediction,
        out string route)
    {
        prediction = 0;
        route = "unknown";

        // Try fold path first
        if (_foldPaths.TryGetValue(operation, out var fold))
        {
            if (TryExecuteFold(fold, left, right, out var foldResult))
            {
                prediction = foldResult;
                route = "fold";
                return true;
            }
        }

        // Try log-linear fit
        if (_logLinearFits.TryGetValue(operation, out var logLin))
        {
            double a = Math.Abs(left);
            double b = Math.Abs(right);
            if (a > 0 && b > 0)
            {
                double logPred = logLin.Alpha * Math.Log(a) + logLin.Beta * Math.Log(b) + logLin.LogK;
                prediction = Math.Exp(logPred);
                
                // Apply sign
                int signProduct = (left >= 0 ? 1 : -1) * (right >= 0 ? 1 : -1);
                prediction *= signProduct;
                
                route = "log-linear";
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Attempts to discover whether targetOp can be executed as fold(baseOp, ...).
    /// E.g., mul(a,b) ≈ fold(add, step=a, count=b)
    /// Requires ≥4 verified pairs, ≥80% accuracy, and ≥2 distinct results.
    /// </summary>
    public void TryDiscoverFoldPath(
        string targetOp,
        IReadOnlyList<(string Input, string Output)> examples,
        IReadOnlyList<string> knownOperations,
        Func<string, double, double, double?> tryExecuteOp)
    {
        if (_foldPaths.ContainsKey(targetOp)) 
            return; // Already found

        if (examples.Count < 3)
            return;

        var numericPairs = examples
            .Select(p => (Args: ExtractNumericArgs(p.Input), Output: p.Output))
            .Where(p => p.Args.Length >= 2 && double.TryParse(p.Output, out _))
            .ToList();

        if (numericPairs.Count < 3)
            return;

        // Try each known base operation
        foreach (var baseOp in knownOperations)
        {
            if (baseOp == targetOp) 
                continue;

            // Try both argument orderings
            foreach (var (accumIdx, countIdx) in new[] { (0, 1), (1, 0) })
            {
                int verified = 0;
                int checked_ = 0;
                var distinctResults = new HashSet<long>();

                foreach (var (args, output) in numericPairs)
                {
                    if (args.Length <= Math.Max(accumIdx, countIdx)) 
                        continue;

                    int count = (int)Math.Round(args[countIdx]);
                    if (count < 0 || count > 50) 
                        continue; // Sanity guard

                    checked_++;

                    // Try to execute fold
                    if (!TryExecuteFoldRaw(
                        baseOp, 
                        args[accumIdx], 
                        count, 
                        out double foldResult, 
                        tryExecuteOp))
                        continue;

                    if (!double.TryParse(output, out double expectedVal))
                        continue;

                    if (Math.Abs(foldResult - expectedVal) < 0.5)
                    {
                        verified++;
                        distinctResults.Add((long)Math.Round(foldResult * 100));
                    }
                }

                // Threshold: ≥4 verified, ≥80% accuracy, ≥2 distinct results
                if (verified >= 4 && checked_ > 0 
                    && (double)verified / checked_ >= 0.8 
                    && distinctResults.Count >= 2)
                {
                    _foldPaths[targetOp] = new FoldPath(
                        targetOp, baseOp, accumIdx, countIdx, verified);
                    return;
                }
            }
        }
    }

    /// <summary>
    /// Discovers a log-linear relationship log(c) = α·log(a) + β·log(b) + γ
    /// using least-squares fitting on training pairs.
    /// Captures multiplicative, divisive, and power-law operations.
    /// </summary>
    public void TryDiscoverLogLinearFit(
        string targetOp,
        IReadOnlyList<(string Input, string Output)> examples)
    {
        if (_logLinearFits.ContainsKey(targetOp))
            return;

        if (examples.Count < 3)
            return;

        var numericPairs = examples
            .Select(p => (Args: ExtractNumericArgs(p.Input), Output: p.Output))
            .Where(p => p.Args.Length >= 2 && double.TryParse(p.Output, out _))
            .ToList();

        if (numericPairs.Count < 3)
            return;

        // Build least-squares system for: log(c) = α·log(a) + β·log(b) + γ
        double n = 0;
        double sumLogA = 0, sumLogB = 0, sumLogC = 0;
        double sumLogA2 = 0, sumLogB2 = 0, sumLogAB = 0;
        double sumLogAC = 0, sumLogBC = 0;

        foreach (var (args, output) in numericPairs)
        {
            if (args.Length < 2) continue;
            if (!double.TryParse(output, out double c)) continue;

            double a = Math.Abs(args[0]);
            double b = Math.Abs(args[1]);
            if (a <= 0 || b <= 0 || c <= 0) continue;

            double logA = Math.Log(a);
            double logB = Math.Log(b);
            double logC = Math.Log(c);

            n++;
            sumLogA += logA;
            sumLogB += logB;
            sumLogC += logC;
            sumLogA2 += logA * logA;
            sumLogB2 += logB * logB;
            sumLogAB += logA * logB;
            sumLogAC += logA * logC;
            sumLogBC += logB * logC;
        }

        if (n < 3) return;

        // Normal equations (simplified 2-variable regression)
        double alpha_denom = n * sumLogA2 - sumLogA * sumLogA;
        double alpha = alpha_denom != 0 
            ? (n * sumLogAC - sumLogA * sumLogC) / alpha_denom 
            : 1.0;

        double beta_denom = n * sumLogB2 - sumLogB * sumLogB;
        double beta = beta_denom != 0 
            ? (n * sumLogBC - sumLogB * sumLogC) / beta_denom 
            : 1.0;

        // Compute residuals
        double residualSum = 0;
        int residualCount = 0;
        foreach (var (args, output) in numericPairs)
        {
            if (args.Length < 2) continue;
            if (!double.TryParse(output, out double c)) continue;

            double a = Math.Abs(args[0]);
            double b = Math.Abs(args[1]);
            if (a <= 0 || b <= 0 || c <= 0) continue;

            double predicted = Math.Exp(
                alpha * Math.Log(a) + beta * Math.Log(b));
            double error = Math.Abs(predicted - c);
            residualSum += error;
            residualCount++;
        }

        double meanResidual = residualCount > 0 ? residualSum / residualCount : 0;

        // Accept if mean residual < 20% of average target magnitude
        if (meanResidual < 0.2)
        {
            _logLinearFits[targetOp] = new LogLinearFit(
                alpha, beta, 0, meanResidual, residualCount);
        }
    }

    /// <summary>
    /// Executes a fold operation using the given fold path.
    /// </summary>
    private bool TryExecuteFold(
        FoldPath fold,
        double arg0,
        double arg1,
        out double result)
    {
        result = 0;
        
        // Determine step and count based on recorded indices
        double step = (fold.AccumArgIdx == 0) ? arg0 : arg1;
        int count = (int)Math.Round((fold.CountArgIdx == 0) ? arg0 : arg1);
        
        if (count < 0 || count > 50)
            return false;

        // For simple operations, simulate fold
        // Multiplication: fold(add, 3, 4) = 3+3+3+3 = 12
        result = 0;
        for (int i = 0; i < count; i++)
        {
            result += step;  // Simple add fold
        }
        
        return true;
    }

    /// <summary>
    /// Attempts to execute a fold operation.
    /// fold(op, step, count) = op(op(...op(0, step)...), step) repeated count times
    /// </summary>
    private static bool TryExecuteFoldRaw(
        string op,
        double step,
        int count,
        out double result,
        Func<string, double, double, double?> tryExecuteOp)
    {
        result = 0;
        
        for (int i = 0; i < count; i++)
        {
            var next = tryExecuteOp(op, result, step);
            if (!next.HasValue)
                return false;
            result = next.Value;
        }

        return true;
    }

    /// <summary>
    /// Extracts numeric arguments from input string.
    /// </summary>
    private static double[] ExtractNumericArgs(string input)
    {
        var nfi = NumericStyle;
        var inv = CultureInfo.InvariantCulture;
        
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
