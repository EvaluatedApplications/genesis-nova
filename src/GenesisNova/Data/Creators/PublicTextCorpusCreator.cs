using System.Collections.Immutable;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.ComponentModel;

namespace GenesisNova.Data.Creators;

/// <summary>
/// Builds next-token style examples from public text corpora.
/// The first hydrate pass downloads the entire dataset split (or capped rows), stores it on disk,
/// then training windows are sampled deterministically from the local full corpus.
/// </summary>
public sealed class PublicTextCorpusCreator : IExampleCreator
{
    private const int MinTextLength = 40;
    private const int MinPromptLength = 8;
    private const int MinAnswerLength = 1;
    private const int MaxTextLength = 1200;
    private const int DefaultRemotePageSize = 100;
    private const int MaxTrainingGrowthPressure = 512;
    private const int MaxRetryableAttempts = 2;

    private static readonly HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromSeconds(60)
    };

    private static readonly SemaphoreSlim RemoteHydrationGate;
    private static readonly ConcurrentDictionary<string, HydrationBackoffState> HydrationBackoffStates = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, HydrationTaskState> HydrationTaskStates = new(StringComparer.OrdinalIgnoreCase);

    static PublicTextCorpusCreator()
    {
        // Network tuning uses default parallelism of 2 (limited to prevent resource exhaustion)
        RemoteHydrationGate = new SemaphoreSlim(1, 2);
    }

    private readonly string _datasetName;
    private readonly string? _config;
    private readonly string _split;
    private readonly string _textField;
    private readonly string? _answerField;
    private readonly GenesisTrainingExampleKind _trainingKind;
    private readonly int _maxRemoteRows;
    private readonly int _remotePageSize;
    private readonly int _maxInMemorySnippets;
    private readonly bool _allowRemoteFetch;
    private readonly bool _requireHydration;
    private readonly ImmutableArray<string> _fallbackSnippets;
    private readonly ImmutableArray<(string Input, string Output)> _fallbackPairs;

    public PublicTextCorpusCreator(
        string name,
        int estimatedComplexity,
        string datasetName,
        string? config,
        string split,
        string textField,
        IEnumerable<string> fallbackSnippets,
        int maxRemoteRows = 0,
        int remotePageSize = DefaultRemotePageSize,
        int maxInMemorySnippets = 20_000,
        bool allowRemoteFetch = true,
        bool requireHydration = false,
        string? answerField = null,
        GenesisTrainingExampleKind trainingKind = GenesisTrainingExampleKind.WindowedText,
        IEnumerable<(string Input, string Output)>? fallbackPairs = null)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Creator name is required.", nameof(name));
        if (string.IsNullOrWhiteSpace(datasetName))
            throw new ArgumentException("Dataset name is required.", nameof(datasetName));
        if (string.IsNullOrWhiteSpace(split))
            throw new ArgumentException("Dataset split is required.", nameof(split));
        if (string.IsNullOrWhiteSpace(textField))
            throw new ArgumentException("Text field is required.", nameof(textField));

        Name = name.Trim();
        EstimatedComplexity = Math.Max(1, estimatedComplexity);
        _datasetName = datasetName.Trim();
        _config = string.IsNullOrWhiteSpace(config) ? null : config.Trim();
        _split = split.Trim();
        _textField = textField.Trim();
        _answerField = string.IsNullOrWhiteSpace(answerField) ? null : answerField.Trim();
        _trainingKind = trainingKind;
        if (_trainingKind == GenesisTrainingExampleKind.PromptAnswer && _answerField is null)
            throw new ArgumentException("Answer field is required for prompt-answer corpora.", nameof(answerField));
        _maxRemoteRows = maxRemoteRows <= 0 ? 0 : maxRemoteRows;
        _remotePageSize = Math.Clamp(remotePageSize, 1, 100);
        _maxInMemorySnippets = Math.Clamp(maxInMemorySnippets, 512, 200_000);
        _allowRemoteFetch = allowRemoteFetch;
        _requireHydration = requireHydration;
        _fallbackSnippets = fallbackSnippets?
            .Select(NormalizeSnippet)
            .Where(x => x.Length >= MinTextLength)
            .Distinct(StringComparer.Ordinal)
            .ToImmutableArray() ?? [];
        _fallbackPairs = fallbackPairs?
            .Select(pair => (Input: NormalizeSnippet(pair.Input), Output: NormalizeSnippet(pair.Output)))
            .Where(x => x.Input.Length >= MinPromptLength && x.Output.Length >= MinAnswerLength)
            .Distinct()
            .ToImmutableArray() ?? [];
        if (_trainingKind == GenesisTrainingExampleKind.PromptAnswer)
        {
            if (_fallbackPairs.Length == 0)
                throw new ArgumentException("At least one fallback pair is required.", nameof(fallbackPairs));
        }
        else if (_fallbackSnippets.Length == 0)
        {
            throw new ArgumentException("At least one fallback snippet is required.", nameof(fallbackSnippets));
        }
    }

    public string Name { get; }
    public int EstimatedComplexity { get; }
    public GenesisTrainingExampleKind TrainingKind => _trainingKind;
    public string LocalCorpusPath => ResolveLocalCorpusPath();

    public static int ResetForFreshRun(IEnumerable<IExampleCreator> creators)
    {
        if (creators is null)
            throw new ArgumentNullException(nameof(creators));

        var creatorsReset = 0;
        HydrationBackoffStates.Clear();
        HydrationTaskStates.Clear();
        foreach (var corpusCreator in creators.OfType<PublicTextCorpusCreator>())
        {
            creatorsReset++;
        }

        return creatorsReset;
    }

    [Obsolete("Use GenerateAsync() instead. This method blocks on async operations.", false)]
    public ImmutableArray<(string Input, string Output)> Generate(int count, int difficulty, bool forTraining)
    {
        return GenerateAsync(count, difficulty, forTraining, CancellationToken.None)
            .GetAwaiter()
            .GetResult();
    }

    public async Task<ImmutableArray<(string Input, string Output)>> GenerateAsync(
        int count, int difficulty, bool forTraining,
        CancellationToken ct = default)
    {
        if (count <= 0)
            return ImmutableArray<(string, string)>.Empty;

        var builder = ImmutableArray.CreateBuilder<(string Input, string Output)>(count);
        if (_trainingKind == GenesisTrainingExampleKind.PromptAnswer)
        {
            var examples = await LoadCorpusPairsAsync(EstimateSnippetTarget(count, difficulty, forTraining), forTraining, ct);
            if (examples.Length == 0)
                throw new InvalidOperationException($"No prompt-answer examples could be built for creator '{Name}'.");

            var sampled = examples.ToArray();
            if (forTraining)
            {
                var remaining = count;
                while (remaining > 0)
                {
                    ShuffleInPlace(sampled);
                    var take = Math.Min(remaining, sampled.Length);
                    for (var i = 0; i < take; i++)
                    {
                        var (input, output) = sampled[i];
                        builder.Add((ApplyPromptVariant(input, difficulty, i), output));
                    }

                    remaining -= take;
                }
            }
            else
            {
                var seed = StableHash32($"{Name}|{difficulty}|{forTraining}|{sampled.Length}");
                var offset = seed % sampled.Length;
                for (var i = 0; i < count; i++)
                {
                    var (input, output) = sampled[(offset + i) % sampled.Length];
                    builder.Add((ApplyPromptVariant(input, difficulty, i), output));
                }
            }
        }
        else
        {
            var snippets = await LoadCorpusSnippetsAsync(EstimateSnippetTarget(count, difficulty, forTraining), forTraining, ct);
            var windows = BuildWindows(snippets, difficulty);
            if (windows.Length == 0)
                throw new InvalidOperationException($"No training windows could be built for creator '{Name}'.");

            if (forTraining)
            {
                var shuffled = windows.ToArray();
                var remaining = count;
                while (remaining > 0)
                {
                    ShuffleInPlace(shuffled);
                    var take = Math.Min(remaining, shuffled.Length);
                    for (var i = 0; i < take; i++)
                        builder.Add(shuffled[i]);

                    remaining -= take;
                }
            }
            else
            {
                var seed = StableHash32($"{Name}|{difficulty}|{forTraining}|{windows.Length}");
                var offset = seed % windows.Length;
                for (var i = 0; i < count; i++)
                {
                    var window = windows[(offset + i) % windows.Length];
                    builder.Add(window);
                }
            }
        }

        return builder.ToImmutable();
    }

    private static string ApplyPromptVariant(string input, int difficulty, int ordinal)
    {
        return NormalizeSnippet(input);
    }

    private async Task<ImmutableArray<string>> LoadCorpusSnippetsAsync(
        int minimumSnippetCount, bool trainingPressure, CancellationToken ct = default)
    {
        var localFile = ResolveLocalCorpusPath();
        var localSnippets = TryLoadLocalSnippets(localFile, trainingPressure);
        var requiredSnippets = Math.Max(1, minimumSnippetCount);
        if (trainingPressure)
        {
            var current = localSnippets.Length;
            var growthPressure = Math.Max(64, Math.Max(requiredSnippets, current / 2));
            growthPressure = Math.Min(growthPressure, MaxTrainingGrowthPressure);
            requiredSnippets = Math.Max(requiredSnippets, current + growthPressure);
        }

        if (localSnippets.Length >= requiredSnippets)
            return localSnippets;

        if (_allowRemoteFetch)
        {
            if (HasActiveHydrationTask(localFile, out var hydrationTask))
            {
                try
                {
                    var hydrated = await hydrationTask.ConfigureAwait(false);
                    if (hydrated.Hydrated)
                        localSnippets = TryLoadLocalSnippets(localFile, trainingPressure);

                    if (localSnippets.Length > 0)
                        ClearHydrationCooldown(localFile);

                    return localSnippets.Length > 0 ? localSnippets : _fallbackSnippets;
                }
                catch
                {
                    // Hydration task failed, continue with available data
                }
            }

            if (TryGetHydrationCooldown(localFile, out _))
                return localSnippets.Length > 0 ? localSnippets : _fallbackSnippets;

            if (localSnippets.Length < requiredSnippets)
            {
                var targetRows = CountTargetRows(localFile, requiredSnippets);
                await StartBackgroundHydrationAsync(localFile, targetRows, ct);
            }

            if (localSnippets.Length > 0)
                return localSnippets;
        }

        if (_requireHydration && localSnippets.Length > 0)
            ClearHydrationCooldown(localFile);

        return localSnippets.Length > 0 ? localSnippets : _fallbackSnippets;
    }

    private async Task<ImmutableArray<(string Input, string Output)>> LoadCorpusPairsAsync(
        int minimumPairCount, bool trainingPressure, CancellationToken ct = default)
    {
        var localFile = ResolveLocalCorpusPath();
        var localPairs = TryLoadLocalPairs(localFile, trainingPressure);
        var requiredPairs = Math.Max(1, minimumPairCount);
        if (trainingPressure)
        {
            var current = localPairs.Length;
            var growthPressure = Math.Max(64, Math.Max(requiredPairs, current / 2));
            growthPressure = Math.Min(growthPressure, MaxTrainingGrowthPressure);
            requiredPairs = Math.Max(requiredPairs, current + growthPressure);
        }

        if (localPairs.Length >= requiredPairs)
            return localPairs;

        if (_allowRemoteFetch)
        {
            if (HasActiveHydrationTask(localFile, out var hydrationTask))
            {
                try
                {
                    var hydrated = await hydrationTask.ConfigureAwait(false);
                    if (hydrated.Hydrated)
                        localPairs = TryLoadLocalPairs(localFile, trainingPressure);

                    if (localPairs.Length > 0)
                        ClearHydrationCooldown(localFile);

                    return localPairs.Length > 0 ? localPairs : _fallbackPairs;
                }
                catch
                {
                    // Hydration task failed, continue with available data
                }
            }

            if (TryGetHydrationCooldown(localFile, out _))
                return localPairs.Length > 0 ? localPairs : _fallbackPairs;

            if (localPairs.Length < requiredPairs)
            {
                var targetRows = CountTargetRows(localFile, requiredPairs);
                await StartBackgroundHydrationAsync(localFile, targetRows, ct);
            }

            if (localPairs.Length > 0)
                return localPairs;
        }

        if (_requireHydration && localPairs.Length > 0)
            ClearHydrationCooldown(localFile);

        return localPairs.Length > 0 ? localPairs : _fallbackPairs;
    }

    private async Task<(bool Hydrated, string? Error)> FetchAndPersistRemoteCorpusAsync(
        string localPath, int targetRows, CancellationToken ct = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(localPath) ?? ".");
        var existingRows = _trainingKind == GenesisTrainingExampleKind.PromptAnswer
            ? (int)Math.Min(int.MaxValue, CountValidLocalPairLines(localPath))
            : (int)Math.Min(int.MaxValue, CountValidLocalLines(localPath));
        var fetchedRows = existingRows;
        var offset = existingRows;
        var wroteAnyRows = existingRows > 0;
        string? firstError = null;
        var stagingPath = CreateStagingPath(localPath);
        StreamWriter? writer = null;

        try
        {
            if (existingRows > 0 && File.Exists(localPath))
                File.Copy(localPath, stagingPath, overwrite: true);

            // Add safety cap: never fetch more than 1M rows in a single session
            const int MaxRowsPerSession = 1_000_000;
            var actualTarget = Math.Min(targetRows, MaxRowsPerSession);

            while (fetchedRows < actualTarget)
            {
                ct.ThrowIfCancellationRequested();

                var remaining = _maxRemoteRows > 0 ? _maxRemoteRows - fetchedRows : int.MaxValue;
                if (remaining <= 0)
                    break;

                var requested = Math.Min(actualTarget - fetchedRows, remaining);
                if (requested <= 0)
                    break;

                var pageSize = Math.Min(_remotePageSize, requested);
                var page = await FetchRowsPageAsync(localPath, offset, pageSize, ct).ConfigureAwait(false);
                if (page.RawRowsReturned <= 0)
                {
                    if (!string.IsNullOrWhiteSpace(page.Error) && firstError is null)
                        firstError = page.Error;
                    break;
                }

                if (writer is null)
                {
                    writer = new StreamWriter(stagingPath, append: existingRows > 0, encoding: new UTF8Encoding(false))
                    {
                        AutoFlush = true
                    };
                }
                foreach (var row in page.Rows)
                    writer.WriteLine(row);

                if (page.Rows.Length > 0)
                    wroteAnyRows = true;

                fetchedRows += page.RawRowsReturned;
                offset += page.RawRowsReturned;

                if (page.RawRowsReturned < pageSize)
                    break;
            }
        }
        finally
        {
            writer?.Dispose();
        }

        if (!wroteAnyRows)
        {
            if (File.Exists(stagingPath))
                File.Delete(stagingPath);
        }
        else
        {
            ReplaceCorpusSnapshot(stagingPath, localPath);
        }

        if (wroteAnyRows)
            ClearHydrationCooldown(localPath);

        return (wroteAnyRows, firstError);
    }

    private async Task<(int RawRowsReturned, ImmutableArray<string> Rows, string? Error)> FetchRowsPageAsync(
        string localPath, int offset, int length, CancellationToken ct = default)
    {
        var query = new List<string>
        {
            $"dataset={Uri.EscapeDataString(_datasetName)}",
            $"split={Uri.EscapeDataString(_split)}",
            $"offset={offset}",
            $"length={length}"
        };
        if (!string.IsNullOrWhiteSpace(_config))
            query.Add($"config={Uri.EscapeDataString(_config)}");

        var url = $"https://datasets-server.huggingface.co/rows?{string.Join("&", query)}";
        const int maxAttempts = MaxRetryableAttempts;
        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                request.Headers.TryAddWithoutValidation("User-Agent", "GenesisNova/1.0");
                
                // Use per-request timeout
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(TimeSpan.FromSeconds(30));
                
                using var response = await SharedHttpClient.SendAsync(
                    request, HttpCompletionOption.ResponseHeadersRead, cts.Token).ConfigureAwait(false);
                
                if (!response.IsSuccessStatusCode)
                {
                    var status = (int)response.StatusCode;
                    var retryable =
                        response.StatusCode == HttpStatusCode.TooManyRequests ||
                        response.StatusCode == HttpStatusCode.RequestTimeout ||
                        response.StatusCode == HttpStatusCode.BadGateway ||
                        response.StatusCode == HttpStatusCode.ServiceUnavailable ||
                        response.StatusCode == HttpStatusCode.GatewayTimeout;
                    if (retryable)
                    {
                        RegisterHydrationFailure(localPath, response.StatusCode == HttpStatusCode.TooManyRequests);
                        if (response.StatusCode == HttpStatusCode.TooManyRequests || attempt >= maxAttempts)
                        {
                            var bodyText = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
                            var snippetText = string.IsNullOrWhiteSpace(bodyText) ? "no-body" : bodyText[..Math.Min(180, bodyText.Length)].Replace('\n', ' ').Replace('\r', ' ');
                            return (0, [], $"HTTP {status} at offset={offset} length={length}: {snippetText}");
                        }

                        await Task.Delay(ComputeRetryDelay(attempt, response), cts.Token).ConfigureAwait(false);
                        continue;
                    }

                    var body = await response.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
                    var snippet = string.IsNullOrWhiteSpace(body) ? "no-body" : body[..Math.Min(180, body.Length)].Replace('\n', ' ').Replace('\r', ' ');
                    return (0, [], $"HTTP {status} at offset={offset} length={length}: {snippet}");
                }

                // Parse JSON with depth limit for security
                using var stream = await response.Content.ReadAsStreamAsync(cts.Token).ConfigureAwait(false);
                
                var docOptions = new JsonDocumentOptions { MaxDepth = 64 };
                using var document = await JsonDocument.ParseAsync(stream, docOptions, cts.Token).ConfigureAwait(false);
                
                if (!document.RootElement.TryGetProperty("rows", out var rowsElement) ||
                    rowsElement.ValueKind != JsonValueKind.Array)
                {
                    return (0, [], $"No rows array at offset={offset} length={length}");
                }

                var rawRows = rowsElement.GetArrayLength();
                var rows = ImmutableArray.CreateBuilder<string>(rowsElement.GetArrayLength());
                foreach (var rowEntry in rowsElement.EnumerateArray())
                {
                    if (!rowEntry.TryGetProperty("row", out var rowData) || rowData.ValueKind != JsonValueKind.Object)
                        continue;

                    if (_trainingKind == GenesisTrainingExampleKind.PromptAnswer)
                    {
                        if (TryReadTextField(rowData, _textField, out var question) &&
                            _answerField is not null &&
                            TryReadTextField(rowData, _answerField, out var answer))
                        {
                            var normalizedQuestion = NormalizeSnippet(question);
                            var normalizedAnswer = NormalizeSnippet(answer);
                            if (normalizedQuestion.Length >= MinPromptLength && normalizedAnswer.Length >= MinAnswerLength)
                                rows.Add(SerializePair(normalizedQuestion, normalizedAnswer));
                        }
                    }
                    else if (TryReadTextField(rowData, _textField, out var text) ||
                        TryReadAnyStringField(rowData, out text))
                    {
                        var normalized = NormalizeSnippet(text);
                        if (normalized.Length >= MinTextLength)
                            rows.Add(normalized);
                    }
                }

                return (rawRows, rows.ToImmutable(), null);
            }
            catch (OperationCanceledException)
            {
                // Propagate cancellation
                throw;
            }
            catch (Exception ex)
            {
                RegisterHydrationFailure(localPath, rateLimited: false);
                if (attempt < maxAttempts)
                {
                    await Task.Delay(ComputeRetryDelay(attempt, null), ct).ConfigureAwait(false);
                    continue;
                }

                return (0, [], $"Request exception at offset={offset} length={length}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        return (0, [], $"Unknown request failure at offset={offset} length={length}");
    }

    private static TimeSpan ComputeRetryDelay(int attempt, HttpResponseMessage? response)
    {
        var retryAfter = response?.Headers.RetryAfter;
        if (retryAfter is not null)
        {
            if (retryAfter.Delta.HasValue && retryAfter.Delta.Value > TimeSpan.Zero)
                return TimeSpan.FromSeconds(Math.Clamp(retryAfter.Delta.Value.TotalSeconds, 1, 30));

            if (retryAfter.Date.HasValue)
            {
                var seconds = (retryAfter.Date.Value - DateTimeOffset.UtcNow).TotalSeconds;
                if (seconds > 0)
                    return TimeSpan.FromSeconds(Math.Clamp(seconds, 1, 30));
            }
        }

        var backoffSeconds = Math.Min(30, Math.Max(2, 5 * attempt * attempt));
        return TimeSpan.FromSeconds(backoffSeconds);
    }

    private static bool IsRateLimitError(string? error)
        => !string.IsNullOrWhiteSpace(error) &&
           error.Contains("HTTP 429", StringComparison.OrdinalIgnoreCase);

    private ImmutableArray<string> TryLoadLocalSnippets(string path, bool randomizedSample)
    {
        if (!File.Exists(path))
            return [];

        var validCount = CountValidLocalLines(path);
        if (validCount <= 0)
            return [];

        if (validCount <= _maxInMemorySnippets)
        {
            return ReadAllValidLines(path);
        }

        if (randomizedSample)
            return ReadRandomSample(path);

        return ReadDeterministicSample(path, validCount);
    }

    private ImmutableArray<(string Input, string Output)> TryLoadLocalPairs(string path, bool randomizedSample)
    {
        if (!File.Exists(path))
            return [];

        var validCount = CountValidLocalPairLines(path);
        if (validCount <= 0)
            return [];

        if (validCount <= _maxInMemorySnippets)
        {
            return ReadAllValidPairs(path);
        }

        if (randomizedSample)
            return ReadRandomPairSample(path);

        return ReadDeterministicPairSample(path, validCount);
    }

    private ImmutableArray<string> ReadRandomSample(string path)
    {
        var target = Math.Max(1, _maxInMemorySnippets);
        var reservoir = new List<string>(target);
        long seen = 0;
        foreach (var line in EnumerateLines(path))
        {
            var normalized = NormalizeSnippet(line);
            if (normalized.Length < MinTextLength)
                continue;

            seen++;
            if (reservoir.Count < target)
            {
                reservoir.Add(normalized);
                continue;
            }

            var replaceIndex = Random.Shared.Next((int)Math.Min(int.MaxValue, seen));
            if (replaceIndex < target)
                reservoir[replaceIndex] = normalized;
        }

        return reservoir.ToImmutableArray();
    }

    private static long CountValidLocalLines(string path)
    {
        if (!File.Exists(path))
            return 0;

        long count = 0;
        foreach (var line in EnumerateLines(path))
        {
            var normalized = NormalizeSnippet(line);
            if (normalized.Length >= MinTextLength)
                count++;
        }

        return count;
    }

    private static long CountValidLocalPairLines(string path)
    {
        if (!File.Exists(path))
            return 0;

        long count = 0;
        foreach (var line in EnumerateLines(path))
        {
            if (TryParsePairLine(line, out var input, out var output) &&
                input.Length >= MinPromptLength &&
                output.Length >= MinAnswerLength)
            {
                count++;
            }
        }

        return count;
    }

    private static ImmutableArray<string> ReadAllValidLines(string path)
    {
        return EnumerateLines(path)
            .Select(NormalizeSnippet)
            .Where(x => x.Length >= MinTextLength)
            .ToImmutableArray();
    }

    private static ImmutableArray<(string Input, string Output)> ReadAllValidPairs(string path)
    {
        return EnumerateLines(path)
            .Select(line => TryParsePairLine(line, out var input, out var output) ? (input, output) : (string.Empty, string.Empty))
            .Where(x => x.Item1.Length >= MinPromptLength && x.Item2.Length >= MinAnswerLength)
            .ToImmutableArray();
    }

    private ImmutableArray<(string Input, string Output)> ReadRandomPairSample(string path)
    {
        var target = Math.Max(1, _maxInMemorySnippets);
        var reservoir = new List<(string Input, string Output)>(target);
        long seen = 0;
        foreach (var line in EnumerateLines(path))
        {
            if (!TryParsePairLine(line, out var input, out var output))
                continue;

            if (input.Length < MinPromptLength || output.Length < MinAnswerLength)
                continue;

            seen++;
            if (reservoir.Count < target)
            {
                reservoir.Add((input, output));
                continue;
            }

            var replaceIndex = Random.Shared.Next((int)Math.Min(int.MaxValue, seen));
            if (replaceIndex < target)
                reservoir[replaceIndex] = (input, output);
        }

        return reservoir.ToImmutableArray();
    }

    private ImmutableArray<string> ReadDeterministicSample(string path, long validCount)
    {
        var target = Math.Max(1, _maxInMemorySnippets);
        var stride = Math.Max(1L, (long)Math.Ceiling(validCount / (double)target));
        var offset = StableHash32(DatasetSignature()) % (int)Math.Max(1L, stride);

        var sampled = ImmutableArray.CreateBuilder<string>(target);
        long seen = 0;
        foreach (var line in EnumerateLines(path))
        {
            var normalized = NormalizeSnippet(line);
            if (normalized.Length < MinTextLength)
                continue;

            if (((seen - offset) % stride) == 0)
                sampled.Add(normalized);

            if (sampled.Count >= target)
                break;

            seen++;
        }

        return sampled.ToImmutable();
    }

    private ImmutableArray<(string Input, string Output)> ReadDeterministicPairSample(string path, long validCount)
    {
        var target = Math.Max(1, _maxInMemorySnippets);
        var stride = Math.Max(1L, (long)Math.Ceiling(validCount / (double)target));
        var offset = StableHash32(DatasetSignature()) % (int)Math.Max(1L, stride);

        var sampled = ImmutableArray.CreateBuilder<(string Input, string Output)>(target);
        long seen = 0;
        foreach (var line in EnumerateLines(path))
        {
            if (!TryParsePairLine(line, out var input, out var output))
                continue;

            if (input.Length < MinPromptLength || output.Length < MinAnswerLength)
                continue;

            if (((seen - offset) % stride) == 0)
                sampled.Add((input, output));

            if (sampled.Count >= target)
                break;

            seen++;
        }

        return sampled.ToImmutable();
    }

    private static IEnumerable<string> EnumerateLines(string path)
    {
        if (!File.Exists(path))
            yield break;

        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            bufferSize: 64 * 1024,
            options: FileOptions.SequentialScan);
        using var reader = new StreamReader(stream);
        while (true)
        {
            var line = reader.ReadLine();
            if (line is null)
                yield break;

            yield return line;
        }
    }

    private static string[] PromptVariants(int difficulty)
        => ["{q}"];

    private string ResolveLocalCorpusPath()
    {
        var baseDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova",
            "datasets");
        Directory.CreateDirectory(baseDir);
        var safeName = DatasetSignature()
            .Replace('/', '_')
            .Replace('\\', '_')
            .Replace(':', '_')
            .Replace('?', '_')
            .Replace('*', '_');
        return Path.Combine(baseDir, $"{safeName}.txt");
    }

    private int EstimateSnippetTarget(int requestedExamples, int difficulty, bool forTraining)
    {
        var baseTarget = Math.Max(1, requestedExamples);
        var difficultyBoost = Math.Max(0, difficulty) / 2;
        var target = baseTarget + difficultyBoost;
        if (forTraining)
            target = Math.Max(target, Math.Min(_maxInMemorySnippets, baseTarget * 4 + Math.Max(32, difficultyBoost * 8)));
        return Math.Clamp(target, 1, _maxInMemorySnippets);
    }

    private static bool TryGetHydrationCooldown(string localPath, out TimeSpan remaining)
    {
        var state = HydrationBackoffStates.GetOrAdd(localPath, _ => new HydrationBackoffState());
        lock (state.Gate)
        {
            var now = DateTimeOffset.UtcNow;
            if (state.CooldownUntil <= now)
            {
                remaining = TimeSpan.Zero;
                return false;
            }

            remaining = state.CooldownUntil - now;
            return true;
        }
    }

    private static bool HasActiveHydrationTask(string localPath, out Task<HydrationResult> task)
    {
        task = null!;
        if (!HydrationTaskStates.TryGetValue(localPath, out var state))
            return false;

        lock (state.Gate)
        {
            if (state.Task is null)
                return false;

            task = state.Task;
            return true;
        }
    }

    private static bool TryObserveHydrationCompletion(
        string localPath,
        Task<HydrationResult> task,
        out HydrationResult result)
    {
        result = default!;
        if (!task.IsCompleted)
            return false;

        try
        {
            result = task.GetAwaiter().GetResult();
            if (HydrationTaskStates.TryGetValue(localPath, out var state))
            {
                lock (state.Gate)
                {
                    if (ReferenceEquals(state.Task, task))
                        state.Task = null;
                }
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private async Task StartBackgroundHydrationAsync(string localPath, int targetRows, CancellationToken ct = default)
    {
        var state = HydrationTaskStates.GetOrAdd(localPath, _ => new HydrationTaskState());
        lock (state.Gate)
        {
            if (state.Task is { IsCompleted: false })
                return;

            // Launch truly async work and track completion
            var tcs = new TaskCompletionSource<HydrationResult>();
            _ = Task.Run(async () =>
            {
                try
                {
                    var (hydrated, error) = await FetchAndPersistRemoteCorpusAsync(localPath, targetRows, ct).ConfigureAwait(false);
                    tcs.SetResult(new HydrationResult(hydrated, error));
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
                finally
                {
                    lock (state.Gate)
                    {
                        if (tcs.Task == state.Task)
                            state.Task = null;
                    }
                }
            }, ct);

            state.Task = tcs.Task;
        }

        // Yield control to allow task to start
        await Task.Yield();
    }

    private static void ReplaceCorpusSnapshot(string stagingPath, string localPath)
    {
        if (File.Exists(localPath))
        {
            File.Replace(stagingPath, localPath, destinationBackupFileName: null, ignoreMetadataErrors: true);
            return;
        }

        File.Move(stagingPath, localPath, overwrite: true);
    }

    private static string CreateStagingPath(string localPath)
    {
        var directory = Path.GetDirectoryName(localPath) ?? ".";
        var fileName = Path.GetFileName(localPath);
        return Path.Combine(directory, $"{fileName}.{Guid.NewGuid():N}.hydrating");
    }

    private static void ClearHydrationCooldown(string localPath)
    {
        if (!HydrationBackoffStates.TryGetValue(localPath, out var state))
            return;

        lock (state.Gate)
        {
            state.FailureCount = 0;
            state.CooldownUntil = DateTimeOffset.MinValue;
        }
    }

    private void RegisterHydrationFailure(string localPath, bool rateLimited)
    {
        var state = HydrationBackoffStates.GetOrAdd(localPath, _ => new HydrationBackoffState());
        lock (state.Gate)
        {
            state.FailureCount = Math.Min(state.FailureCount + 1, 12);
            var now = DateTimeOffset.UtcNow;
            var minutes = rateLimited
                ? Math.Min(60, 5 + (state.FailureCount * 5))
                : Math.Min(20, 1 + (state.FailureCount * 2));
            state.CooldownUntil = now.AddMinutes(minutes);
        }
    }

    private sealed class HydrationBackoffState
    {
        public object Gate { get; } = new();
        public int FailureCount;
        public DateTimeOffset CooldownUntil;
    }

    private sealed class HydrationTaskState
    {
        public object Gate { get; } = new();
        public Task<HydrationResult>? Task;
    }

    private sealed record HydrationResult(bool Hydrated, string? Error);

    private int CountTargetRows(string localPath, int requiredSnippets)
    {
        var existing = _trainingKind == GenesisTrainingExampleKind.PromptAnswer
            ? (int)Math.Min(int.MaxValue, CountValidLocalPairLines(localPath))
            : (int)Math.Min(int.MaxValue, CountValidLocalLines(localPath));
        var minimum = Math.Max(requiredSnippets, existing);
        if (existing == 0)
            return Math.Max(1, minimum);

        // Grow in chunks so we keep API pressure low but avoid tiny repeated fetches.
        var chunk = Math.Max(8, existing);
        return Math.Max(minimum, existing + chunk);
    }

    private string DatasetSignature()
        => $"{_datasetName}_{_config ?? "default"}_{_split}_{_textField}";

    private static string SerializePair(string input, string output)
        => $"{input.Replace('\t', ' ')}\t{output.Replace('\t', ' ')}";

    private static bool TryParsePairLine(string line, out string input, out string output)
    {
        input = string.Empty;
        output = string.Empty;
        if (string.IsNullOrWhiteSpace(line))
            return false;

        var parts = line.Split('\t', 2, StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
            return false;

        input = NormalizeSnippet(parts[0]);
        output = NormalizeSnippet(parts[1]);
        return input.Length > 0 && output.Length > 0;
    }

    private static bool TryReadTextField(JsonElement rowData, string fieldName, out string value)
    {
        value = string.Empty;
        if (!rowData.TryGetProperty(fieldName, out var field))
            return false;

        if (!TryExtractText(field, out value))
            return false;

        return value.Length > 0;
    }

    private static bool TryReadAnyStringField(JsonElement rowData, out string value)
    {
        value = string.Empty;
        foreach (var property in rowData.EnumerateObject())
        {
            if (!TryExtractText(property.Value, out var candidate))
                continue;

            value = candidate;
            return true;
        }

        return false;
    }

    private static bool TryExtractText(JsonElement element, out string value, int depth = 0)
    {
        value = string.Empty;
        if (depth > 4)
            return false;

        switch (element.ValueKind)
        {
            case JsonValueKind.String:
                value = element.GetString() ?? string.Empty;
                return !string.IsNullOrWhiteSpace(value);
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                {
                    if (TryExtractText(item, out value, depth + 1))
                        return true;
                }
                return false;
            case JsonValueKind.Object:
                foreach (var property in element.EnumerateObject())
                {
                    if (TryExtractText(property.Value, out value, depth + 1))
                        return true;
                }
                return false;
            default:
                return false;
        }
    }

    private static string NormalizeSnippet(string value)
    {
        var compact = string.Join(' ', value
            .Replace('\r', ' ')
            .Replace('\n', ' ')
            .Replace('\t', ' ')
            .Split(' ', StringSplitOptions.RemoveEmptyEntries));
        if (compact.Length > MaxTextLength)
            compact = compact[..MaxTextLength];
        return compact.Trim();
    }

    private static int StableHash32(string value)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Math.Abs(BitConverter.ToInt32(bytes, 0));
    }

    private static void ShuffleInPlace<T>(T[] values)
    {
        for (var i = values.Length - 1; i > 0; i--)
        {
            var j = Random.Shared.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static ImmutableArray<(string Input, string Output)> BuildWindows(ImmutableArray<string> snippets, int difficulty)
    {
        if (snippets.Length == 0)
            return [];

        // Context chunk (what the model reads)
        var contextLength = Math.Clamp(6 + Math.Max(0, difficulty) * 3, 4, 48);
        
        // Prediction chunk (what the model must generate)
        // Easier difficulties predict shorter chunks; harder difficulties predict longer
        var predictLength = Math.Clamp(2 + Math.Max(0, difficulty) / 2, 2, 12);
        
        var stride = Math.Clamp(contextLength / 2, 1, 8);
        var windows = ImmutableArray.CreateBuilder<(string Input, string Output)>();

        foreach (var snippet in snippets)
        {
            var tokens = snippet
                .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(CleanToken)
                .Where(t => t.Length > 0)
                .ToArray();
            
            // Need at least context + prediction tokens
            var minRequired = contextLength + predictLength;
            if (tokens.Length < minRequired)
                continue;

            var maxStart = tokens.Length - minRequired;
            for (var start = 0; start <= maxStart; start += stride)
            {
                // Input: contextLength words
                var input = string.Join(' ', tokens.Skip(start).Take(contextLength));
                
                // Output: predictLength words (the next chunk)
                var output = string.Join(' ', tokens.Skip(start + contextLength).Take(predictLength));
                
                if (input.Length > 0 && output.Length > 0)
                    windows.Add((input, output));
            }
        }

        return windows.ToImmutable();
    }

    private static string CleanToken(string token)
    {
        if (token.Length == 0)
            return string.Empty;

        var chars = token
            .Trim()
            .Where(c => char.IsLetterOrDigit(c) || c is '\'' or '-' or '_')
            .ToArray();
        return new string(chars).ToLowerInvariant();
    }
}

public static class PublicTextCorpusDefaults
{
    public static readonly IExampleCreator FineWebEdu = new PublicTextCorpusCreator(
        name: "public:fineweb-edu",
        estimatedComplexity: 20,
        datasetName: "HuggingFaceFW/fineweb-edu",
        config: "sample-10BT",
        split: "train",
        textField: "text",
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets:
        [
            "The scientific method starts with observation, then forms a testable hypothesis and repeats experiments to refine confidence in the result.",
            "Software reliability improves when teams keep changes small, measure outcomes, and rollback quickly when metrics move in the wrong direction."
        ]);

    public static readonly IExampleCreator SlimPajama = new PublicTextCorpusCreator(
        name: "public:slimpajama",
        estimatedComplexity: 24,
        datasetName: "allenai/c4",
        config: "en",
        split: "train",
        textField: "text",
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets:
        [
            "Distributed systems trade simplicity for scale, so robust services use retries with backoff, idempotency keys, and explicit timeout budgets.",
            "Machine learning pipelines benefit from deterministic preprocessing, stable data contracts, and continuous evaluation against held-out examples."
        ]);

    public static readonly IExampleCreator Gutenberg = new PublicTextCorpusCreator(
        name: "public:gutenberg",
        estimatedComplexity: 26,
        datasetName: "wikimedia/wikipedia",
        config: "20231101.en",
        split: "train",
        textField: "text",
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets:
        [
            "Public-domain literature often uses longer sentence structures that push memory and continuity, which helps models practice longer-range token prediction.",
            "Classic prose provides broad vocabulary and narrative context, improving lexical coverage without restrictive licensing constraints."
        ]);

    public static readonly IExampleCreator OpenWebMath = new PublicTextCorpusCreator(
        name: "public:openwebmath",
        estimatedComplexity: 30,
        datasetName: "open-web-math/open-web-math",
        config: "default",
        split: "train",
        textField: "text",
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets:
        [
            "If a function is continuous on a closed interval then it reaches both a maximum and minimum value, which is useful for optimization proofs.",
            "To solve a linear equation isolate the unknown by applying inverse operations equally on both sides until only the variable remains."
        ]);

    public static readonly IExampleCreator GSM8K = new PublicTextCorpusCreator(
        name: "public:gsm8k",
        estimatedComplexity: 22,
        datasetName: "openai/gsm8k",
        config: "main",
        split: "train",
        textField: "question",
        answerField: "answer",
        trainingKind: GenesisTrainingExampleKind.PromptAnswer,
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets: [],
        fallbackPairs:
        [
            ("If Alex has 5 apples and buys 3 more, how many does he have total?", "8"),
            ("Sarah spent $12 on books and $8 on pens. How much did she spend in total?", "20"),
            ("A store has 20 apples. If they sell 7 and receive 5 new ones, how many do they have?", "18"),
            ("John has 3 times as many marbles as Mary. If Mary has 4 marbles, how many does John have?", "12"),
            ("A recipe calls for 2 cups of flour per batch. If you make 5 batches, how many cups do you need?", "10"),
            ("Emma read 15 pages on Monday and 12 pages on Tuesday. How many total pages did she read?", "27"),
            ("If a pizza costs 14 dollars and you have 3 pizzas, what is the total cost?", "42"),
            ("A class has 24 students. If they form 6 equal groups, how many students are in each group?", "4"),
            ("David earns 8 dollars per hour. If he works 5 hours, how much money does he make?", "40")
        ]);

    public static readonly IExampleCreator WikidataTriples = new PublicTextCorpusCreator(
        name: "public:wikidata-triples",
        estimatedComplexity: 24,
        datasetName: "wikimedia/wikipedia",
        config: "20231101.en",
        split: "train",
        textField: "text",
        maxRemoteRows: 0,
        requireHydration: true,
        fallbackSnippets:
        [
            "Paris is the capital and largest city of France, known for the Eiffel Tower and being a major cultural center.",
            "Albert Einstein was a German theoretical physicist best known for his theory of relativity.",
            "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
            "The Great Wall of China is a series of fortifications that were built across northern China.",
            "Napoleon Bonaparte was a French military commander who rose to prominence during the French Revolution.",
            "Tokyo is the capital of Japan and the world's most populous metropolitan area.",
            "William Shakespeare was an English playwright and poet who is widely regarded as the greatest writer of the English language.",
            "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
            "Isaac Newton was an English mathematician, physicist, astronomer, and author who formulated the laws of motion and gravitation.",
            "The Mona Lisa is a painting by Leonardo da Vinci created in the Renaissance period."
        ]);
}
