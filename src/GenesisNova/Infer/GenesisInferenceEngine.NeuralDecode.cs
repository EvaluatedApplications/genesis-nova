using GenesisNova.Model;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Core;

namespace GenesisNova.Infer;

public sealed partial class GenesisInferenceEngine
{
    /// <summary>
    /// Mode 2: platonic-assisted reasoning. Generates neurally in bounded segments and, between
    /// segments, may invoke the platonic space as an internal scratchpad: it scans the working
    /// context (prompt + tokens so far) for a platonic-resolvable arithmetic sub-step, resolves it
    /// EXACTLY via the log/poly face homomorphism (TryResolveAssistSubResult), tokenizes the
    /// sub-result, INJECTS it back into the working sequence, then continues neural decode
    /// conditioned on it.
    ///
    /// Bounded by <see cref="_maxPlatonicAssistInvocations"/> tool calls per generation. If nothing
    /// fires (no resolvable sub-step, or low decode confidence) it degrades to pure neural output —
    /// never hangs and never injects garbage.
    /// </summary>
    private GenerationResult GenerateNeuralWithPlatonicAssist(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens)
    {
        var adaptiveBiasScale = GetAdaptiveBiasScale();
        // The bias context derives only from the (invariant) input tokens + vocabulary, so build it ONCE.
        var biasContext = BuildTokenBiasContext(inputTokens);

        var generated = new List<int>(Math.Max(1, request.MaxNewTokens));
        var evidence = new List<PlatonicEvidence>();
        var totalBiasCount = 0;
        var biasMagnitudeSum = 0.0;
        var prev = _tokenizer.BosTokenId;

        var assistInvocations = 0;
        var assistFired = 0;
        var injectedAny = false;
        var totalAssistConfidence = 0.0;
        var alreadyResolved = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        // ENCODE-ONCE: the encoder seed (hInput) depends only on the invariant prompt tokens, so encode
        // them ONCE before the decode loop and reuse the seed every step (O(N+M) instead of O(N·M)).
        // Scoped to THIS generation only and disposed in finally — never cached across generations.
        var promptState = _model.EncodePromptState(inputTokens);
        try
        {
        // Re-invoke the platonic tool roughly every (budget / (cap+1)) decoded tokens so calls are
        // spread across the generation rather than all bunched at the start.
        var maxInvocations = _maxPlatonicAssistInvocations;
        var segment = maxInvocations > 0
            ? Math.Max(1, request.MaxNewTokens / (maxInvocations + 1))
            : int.MaxValue;
        var nextAssistAt = segment;

        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            // Platonic tool-invocation point (bounded): try to resolve a sub-step from the current
            // working context and inject its exact result before continuing neural decode.
            if (maxInvocations > 0 && assistInvocations < maxInvocations && generated.Count >= nextAssistAt)
            {
                nextAssistAt += segment;
                assistInvocations++;
                var workingText = BuildChunkContext(request.Input, generated, _tokenizer);
                if (TryResolveAssistSubResult(workingText, alreadyResolved, out var subResult, out var subConfidence))
                {
                    var injectTokens = _tokenizer.Encode(subResult, addEos: false);
                    var room = request.MaxNewTokens - generated.Count;
                    if (injectTokens.Length > 0 && room > 0)
                    {
                        foreach (var t in injectTokens.Take(room))
                        {
                            generated.Add(t);
                            prev = t;
                        }
                        assistFired++;
                        injectedAny = true;
                        totalAssistConfidence += subConfidence;
                        // Re-align the loop counter with the now-larger sequence.
                        i = generated.Count - 1;
                        if (generated.Count >= request.MaxNewTokens)
                            break;
                    }
                }
            }

            var biases = BuildTokenBiases(biasContext, generated, adaptiveBiasScale, out var biasMagnitude, out var stepEvidence);

            var next = _model.PredictNextToken(
                inputTokens,
                prev,
                stepIndex: i,
                disallowToken: i == 0 ? _tokenizer.EosTokenId : null,
                penalizedTokens: generated,
                repetitionPenalty: 0.35,
                tokenBiases: biases,
                stopToken: _tokenizer.EosTokenId,
                promptState: promptState);
            if (biases is not null)
            {
                totalBiasCount += biases.Count;
                biasMagnitudeSum += biasMagnitude;
                if (stepEvidence.Count > 0)
                    evidence.AddRange(stepEvidence);
            }
            generated.Add(next);
            if (next == _tokenizer.EosTokenId)
                break;
            prev = next;
        }
        }
        finally
        {
            promptState.Dispose();
        }

        var avgAssistConfidence = assistFired > 0 ? totalAssistConfidence / assistFired : 0.0;
        return new GenerationResult(
            Output: _tokenizer.Decode(generated),
            GeneratedTokens: generated.ToArray(),
            UsedPlatonicQuery: injectedAny,
            UsedNeuralFallback: !injectedAny,
            DecisionPath: injectedAny
                ? $"neural+platonic-assist[{assistFired}/{assistInvocations}]"
                : "neural-token+platonic-bias(assist-miss)",
            PlatonicConfidence: avgAssistConfidence,
            AppliedBiasCount: totalBiasCount,
            AverageBiasMagnitude: totalBiasCount > 0 ? biasMagnitudeSum / Math.Max(1, generated.Count) : 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: assistFired,
            Evidence: CollapseEvidence(evidence),
            PlatonicAssistInvocations: assistInvocations,
            PlatonicAssistFired: assistFired);
    }

    /// <summary>
    /// Resolves the arithmetic sub-step in <paramref name="workingText"/> through the SAME
    /// GRU-constructed query path the direct route uses (<see cref="TryGenerateFromGruQuery"/>): the GRU
    /// op head classifies the operation from the scratchpad's CONTEXT and the face homomorphism computes
    /// it exactly. NO hardcoded symbol parser (the regex / "x"⇒multiply switch was removed 2026-06-14) —
    /// the operator is never read off a pattern, so the assist disambiguates the op the same learned way
    /// as every other arithmetic path. Returns the injected sub-result (" = N ") and a decode-consistency
    /// confidence; false when the GRU abstains, nothing resolves, or this value was already injected.
    /// </summary>
    private bool TryResolveAssistSubResult(
        string workingText,
        HashSet<string> alreadyResolved,
        out string subResult,
        out double confidence)
    {
        subResult = string.Empty;
        confidence = 0.0;
        if (string.IsNullOrWhiteSpace(workingText))
            return false;

        // The whole working context IS the query: the GRU selects the operands + op and the homomorphism
        // computes. Abstains (false) on non-arithmetic scratchpads or low decode quality — same gates as
        // the direct route, so the assist injects only what the learned query path would itself produce.
        if (!TryGenerateFromGruQuery(new GenerationRequest(workingText, 6), out var resolved))
            return false;

        var value = resolved.Output.Trim();
        if (value.Length == 0 || !alreadyResolved.Add(value))
            return false; // nothing resolved, or this value was already injected this generation

        // Inject as " = <result> " so the neural decoder continues conditioned on the resolved value.
        subResult = $" = {value} ";
        confidence = resolved.PlatonicConfidence;
        return true;
    }

    private GenerationResult GenerateNeuralTokens(
        GenerationRequest request,
        IReadOnlyList<int> inputTokens,
        bool neuralFallback)
    {
        var adaptiveBiasScale = GetAdaptiveBiasScale();
        // The bias context derives only from the (invariant) input tokens + vocabulary, so build it ONCE.
        var biasContext = BuildTokenBiasContext(inputTokens);
        var generated = new List<int>(Math.Max(1, request.MaxNewTokens));
        var evidence = new List<PlatonicEvidence>();
        var totalBiasCount = 0;
        var biasMagnitudeSum = 0.0;
        var prev = _tokenizer.BosTokenId;
        // ENCODE-ONCE: encode the invariant prompt to its seed (hInput) ONCE and reuse it every decode
        // step (O(N+M) instead of O(N·M)). Scoped to THIS generation and disposed in finally below.
        var promptState = _model.EncodePromptState(inputTokens);
        try
        {
        for (var i = 0; i < request.MaxNewTokens; i++)
        {
            var biases = BuildTokenBiases(biasContext, generated, adaptiveBiasScale, out var biasMagnitude, out var stepEvidence);

            var next = _model.PredictNextToken(
                inputTokens,
                prev,
                stepIndex: i,
                disallowToken: i == 0 ? _tokenizer.EosTokenId : null,
                penalizedTokens: generated,
                repetitionPenalty: 0.35,
                tokenBiases: biases,
                stopToken: _tokenizer.EosTokenId,
                promptState: promptState);
            if (biases is not null)
            {
                totalBiasCount += biases.Count;
                biasMagnitudeSum += biasMagnitude;
                if (stepEvidence.Count > 0)
                    evidence.AddRange(stepEvidence);
            }
            generated.Add(next);
            if (next == _tokenizer.EosTokenId)
                break;
            prev = next;
        }
        }
        finally
        {
            promptState.Dispose();
        }

        return new GenerationResult(
            Output: _tokenizer.Decode(generated),
            GeneratedTokens: generated.ToArray(),
            UsedPlatonicQuery: false,
            UsedNeuralFallback: neuralFallback,
            DecisionPath: "neural-token+platonic-bias",
            PlatonicConfidence: 0.0,
            AppliedBiasCount: totalBiasCount,
            AverageBiasMagnitude: totalBiasCount > 0 ? biasMagnitudeSum / Math.Max(1, generated.Count) : 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 0,
            Evidence: CollapseEvidence(evidence));
    }

    // Build the per-generation bias context. Mirrors the old BuildTokenBiases setup: the discriminative
    // INPUT-only context concepts, and the ascending list of vocab tokens that are non-special, non-blank
    // and carry a concept. Returns null when there are no context concepts (the old early-return null).
    private TokenBiasContext? BuildTokenBiasContext(IReadOnlyList<int> inputTokens)
    {
        var contextConcepts = BuildContextConcepts(inputTokens);
        if (contextConcepts.Count == 0)
            return null;

        var conceptTokens = new List<(int, string)>();
        for (var token = 0; token < _tokenizer.VocabularySize; token++)
        {
            if (token == _tokenizer.PadTokenId || token == _tokenizer.BosTokenId || token == _tokenizer.EosTokenId)
                continue;
            var candidate = _tokenizer.Vocabulary[token];
            if (string.IsNullOrWhiteSpace(candidate))
                continue;
            if (!_memory.ContainsConcept(candidate))
                continue;
            conceptTokens.Add((token, candidate));
        }
        return new TokenBiasContext(contextConcepts, conceptTokens);
    }

    private IReadOnlyDictionary<int, double>? BuildTokenBiases(
        TokenBiasContext? context,
        IReadOnlyList<int> generatedTokens,
        double strength,
        out double averageBiasMagnitude,
        out IReadOnlyList<PlatonicEvidence> evidence)
    {
        averageBiasMagnitude = 0.0;
        evidence = Array.Empty<PlatonicEvidence>();
        if (context is null)
            return null;

        var contextConcepts = context.ContextConcepts;
        var biases = new Dictionary<int, double>();
        var evidenceItems = new List<PlatonicEvidence>();
        var scale = 0.9 * Math.Max(0.0, strength);
        foreach (var (token, candidate) in context.ConceptTokens)
        {
            // SHARPEST relation, not the AVERAGE. A candidate is boosted by its single STRONGEST link to a query
            // anchor — not by its mean relatedness across all anchors. Mean-aggregation rewarded BREADTH of
            // connection (a token wired into ubiquitous framing-word hubs scored on every query) and DILUTED the
            // real answer's one sharp link (fruit↔banana) against the neutral 0.5s of the other anchors — both
            // collapsed decode onto the most-connected token regardless of the query. Max keeps the answer's
            // sharp signal intact and leaves unrelated candidates at ~0 (no edge ⇒ contradiction 0.5 ⇒ 0).
            var bestRelated = double.NegativeInfinity;
            string? bestAnchor = null;
            foreach (var concept in contextConcepts)
            {
                if (string.Equals(candidate, concept, StringComparison.OrdinalIgnoreCase))
                    continue;
                var related = 0.5 - _memory.GetContradiction(candidate, concept);
                if (related > bestRelated)
                {
                    bestRelated = related;
                    bestAnchor = concept;
                }
            }

            if (bestAnchor is null)
                continue;

            var bias = Math.Clamp(bestRelated * scale, -0.9, 0.9);
            if (Math.Abs(bias) > 1e-6)
            {
                biases[token] = bias;
                if (evidenceItems.Count < 512)
                    evidenceItems.Add(new PlatonicEvidence(candidate, bestAnchor, bias, 1));
            }
        }

        // REPETITION GUARD: a token already emitted THIS generation must not keep its positive platonic BOOST
        // on later steps. The query bias is static (e.g. `find X` boosts X at every step), so without this the
        // boosted answer re-wins each step and the decoder repeats it to maxTokens instead of taking the LEARNED
        // EOS ("ObserveTrainingPair" ×16). Dropping the boost for emitted tokens lets termination (or a genuinely
        // next token) win — same principle as the rule that the bias layer must not override the learned stop.
        // Only the POSITIVE boost is removed; the NN's own logit (which legitimately allows repeats like "100")
        // is untouched, and numeric tokens aren't concepts so they never carried a bias here anyway.
        for (var g = 0; g < generatedTokens.Count; g++)
            if (biases.TryGetValue(generatedTokens[g], out var prior) && prior > 0)
                biases.Remove(generatedTokens[g]);

        if (biases.Count == 0)
            return null;

        averageBiasMagnitude = biases.Values.Average(v => Math.Abs(v));
        evidence = CollapseEvidence(evidenceItems);
        return biases;
    }

    private IReadOnlyList<string> BuildContextConcepts(IReadOnlyList<int> inputTokens)
    {
        // The decode bias is conditioned on THIS query's specific cues. Candidate concepts come from the INPUT
        // tokens only — GENERATED tokens are excluded (biasing on the model's own output makes its platonic
        // SIBLINGS the most-boosted tokens, so decode cascades through the neighbourhood — "fruit grape banana"
        // — instead of taking the learned EOS) — then filtered to the DISCRIMINATIVE (lowest relation-degree)
        // cues via the SAME rule retrieval uses, so framing-word HUBS ("what"/"kind"/"thing") never enter.
        // (The whole-space checkpoint concepts were never merged — that path was inert — so it has been retired.)
        var raw = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        AddTokens(inputTokens, raw);
        var candidates = raw
            .Where(c => !_memory.IsOperationToken(c) && _memory.ContainsConcept(c))
            .ToList();
        return SelectDiscriminativeConcepts(candidates);
    }

    private void AddTokens(IReadOnlyList<int> tokens, HashSet<string> concepts)
    {
        foreach (var token in tokens)
        {
            if (token < 0 || token >= _tokenizer.VocabularySize)
                continue;
            if (token == _tokenizer.PadTokenId || token == _tokenizer.BosTokenId || token == _tokenizer.EosTokenId)
                continue;

            var value = _tokenizer.Vocabulary[token];
            if (!string.IsNullOrWhiteSpace(value))
                concepts.Add(value.ToLowerInvariant());
        }
    }

    private IReadOnlyList<string> ExtractConceptAnchors(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return Array.Empty<string>();

        var candidates = input
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim('?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '"', '\''))
            .Select(t => t.ToLowerInvariant())
            // Keep single-char DIGIT tokens as concept anchors: a bare number ("3") is a legitimate
            // concept with a learned relation to its word, and excluding it sent digit→word to the
            // neural path. Stray single letters still dropped; ContainsConcept gates.
            .Where(t => t.Length > 1 || (t.Length == 1 && char.IsDigit(t[0])))
            // An op-token (a declared ROUTE-TRIGGER verb) is never a retrieval anchor. The gym declares none — the
            // degree filter below is what keeps framing words out, data-drivenly.
            .Where(t => !_memory.IsOperationToken(t))
            .Where(t => _memory.ContainsConcept(t))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        return SelectDiscriminativeConcepts(candidates);
    }

    /// <summary>
    /// DISCRIMINATIVE concept filter, shared by retrieval anchors AND the decode-bias context — ONE rule, both
    /// paths (otherwise a framing-word hub evicted from one walks back in through the other and collapses the
    /// result). Data-driven; replaces the op-token stopword list, which doesn't scale. The cue is the most
    /// SPECIFIC token = LOWEST relation degree; a framing word ("what"/"thing"/"kind") sits near everything,
    /// accrues high degree, and is dropped — preventing a hub from collapsing the result to a constant. Only the
    /// low-degree content cue(s) survive; ties/near-ties are kept so genuine multi-concept queries still work.
    /// </summary>
    private IReadOnlyList<string> SelectDiscriminativeConcepts(IReadOnlyList<string> candidates)
    {
        if (candidates.Count <= 1)
            return candidates;
        var byDegree = candidates
            .Select(t => (Token: t, Degree: _memory.GetRelationDegree(t)))
            .OrderBy(x => x.Degree)
            .ToList();
        var minDegree = byDegree[0].Degree;
        return byDegree
            .Where(x => x.Degree <= (minDegree * 2) + 2) // keep the cue + any comparably-specific tokens; drop hubs
            .Select(x => x.Token)
            .Take(4)
            .ToArray();
    }

    private double GetAdaptiveBiasScale()
    {
        lock (_telemetryLock)
            return Math.Clamp(_adaptiveBiasScale * _trainerHint.BiasScale, MinAdaptiveBiasScale, MaxAdaptiveBiasScale);
    }
}
