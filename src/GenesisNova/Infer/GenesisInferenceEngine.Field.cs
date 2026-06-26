using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using GenesisNova.Cognition;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;

namespace GenesisNova.Infer;

/// <summary>
/// CONSCIOUS-FIELD COGNITION (PLATONIC_MIND.md / PLATONIC_CONSCIOUSNESS.md) — the model thinks by the field
/// reducing a thought to a settled state, with NO route/plan/op classifier. Capabilities are selected by the
/// STRUCTURE of the prompt (operands + operators + cues present), and every reduction runs on the substrate's own
/// general primitives (the homomorphism via PlatonicGliderInterpreter — exact, generalising to unseen operands):
///   • ARITHMETIC  — bare or worded expressions, multi-operator with precedence, and fold (a+b+c).
///   • PREDICATE   — compare-by-sign on the substrate (greater / less / equal).
///   • NUMBER-WORD — digit↔word via the linguistic codec (generative, like the spelling codec).
///   • RELAX       — retrieval / disambiguation by Reason (synonym, category).
///   • ABSTAIN     — nothing applies and nothing settles → speak nothing (no invention).
/// The plan/op heads only ever SELECTED these; the reduction logic itself was always classifier-free. Lifted here
/// and driven by structure. The persistent self threads continuity via Generate(); conditioning the reduction on
/// that self is the next layer.
/// </summary>
public sealed partial class GenesisInferenceEngine
{
    private GenerationResult GenerateFromField(GenerationRequest request)
    {
        if (_memory is not DialecticalSpace)
            return GenerateSinglePass(request, _tokenizer.Encode(request.Input)); // field cognition needs the dialectical core

        ResetRouteTelemetry();
        // Separate operator symbols from digits so compact arithmetic ("1+1", "5>3") tokenises like the spaced form
        // ("1 + 1") — the field handles either, which the gym writes spaced but a person at the REPL may not.
        var toks = TokenizeField(request.Input);

        if (TryFieldInduction(toks, request, out var r) ||
            TryFieldPredicate(toks, request, out r) ||
            TryFieldArithmetic(toks, request, out r) ||
            TryFieldNumberWord(toks, request, out r) ||
            (FieldTicksEnabled && TryFieldTick(request, out r)) ||
            TryFieldLearnedFunction(request, out r) ||
            (FieldTicksEnabled && TryFieldMeaningTick(toks, request, out r)) ||
            (MeaningOpsEnabled && TryFieldAnalogy(toks, request, out r)) ||
            (MeaningOpsEnabled && TryFieldComposeMeaning(toks, request, out r)) ||
            (TalkEnabled && TryFieldRespondDirect(request, out r)) ||  // a known persona cue SPEAKS before grammar tries to learn it
            TryFieldLearn(toks, request, out r) ||
            TryFieldRelax(toks, request, out r) ||
            (TalkEnabled && TryFieldRespondGeneralize(request, out r)))
            return r;

        return FieldAbstain(); // high free-energy, no basin — speak nothing
    }

    // ── INDUCTION: few-shot in-context rule ("fn 2 is 4 fn 5 is 10 fn 3 is" → 6). Induce the consistent transform
    //    (+k or ×k) from the demos via the homomorphism, then apply it to the query operand. No learned weights —
    //    the rule comes only from the in-prompt examples (the transform varies every prompt). ──────────────────────
    private bool TryFieldInduction(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (!toks.Contains("fn") || !toks.Contains("is")) return false;
        var demos = new List<(double In, double Out)>();
        double? query = null;
        for (var i = 0; i + 2 < toks.Count; i++)
        {
            if (toks[i] != "fn" || toks[i + 2] != "is") continue;
            if (!double.TryParse(toks[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var inV)) continue;
            if (i + 3 < toks.Count && double.TryParse(toks[i + 3], NumberStyles.Float, CultureInfo.InvariantCulture, out var outV))
                demos.Add((inV, outV));
            else
                query = inV; // the trailing "fn N is" with no answer = the operand to predict
        }
        if (demos.Count < 2 || query is null) return false;

        // Additive rule: constant difference (out − in) across all demos → apply +k.
        var addK = FieldStep(demos[0].Out, GliderOp.Subtract, demos[0].In);
        if (!double.IsNaN(addK) && demos.All(d => Math.Abs(FieldStep(d.Out, GliderOp.Subtract, d.In) - addK) < 1e-6))
            return EmitField(FieldFormat(FieldStep(query.Value, GliderOp.Add, addK)), "field-induce", request, out result);

        // Multiplicative rule: constant ratio (out ÷ in) across all demos → apply ×k.
        if (demos.All(d => Math.Abs(d.In) > 1e-9))
        {
            var mulK = FieldStep(demos[0].Out, GliderOp.Divide, demos[0].In);
            if (!double.IsNaN(mulK) && demos.All(d => Math.Abs(FieldStep(d.Out, GliderOp.Divide, d.In) - mulK) < 1e-6))
                return EmitField(FieldFormat(FieldStep(query.Value, GliderOp.Multiply, mulK)), "field-induce", request, out result);
        }
        return false;
    }

    // ── PREDICATE: two numeric operands + a comparison cue → compare-by-sign (greater/less/equal). ────────────────
    private bool TryFieldPredicate(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (!toks.Any(IsCompareCue)) return false;
        var nums = toks.Where(t => double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out _)).ToList();
        if (nums.Count != 2) return false;
        var glider = new PlatonicGlider("predicate",
            new Branch(new Compare(CompareOp.Greater, new Operand(0), new Operand(1)),
                new Literal("greater"),
                new Branch(new Compare(CompareOp.Less, new Operand(0), new Operand(1)),
                    new Literal("less"), new Literal("equal"))));
        string ans;
        try { ans = _glider.Execute(glider, new[] { nums[0], nums[1] }); }
        catch { return false; }
        return !string.IsNullOrEmpty(ans)
            && EmitField(ans, "field-predicate", request, out result);
    }

    // ── ARITHMETIC: operands + operators (infix with precedence) OR operands + a single op-cue (fold). ────────────
    private bool TryFieldArithmetic(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var merged = SignMerge(toks);

        var operands = new List<double>();
        var infixOps = new List<GliderOp>();
        foreach (var t in merged)
        {
            if (double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) operands.Add(v);
            else if (TryOpToken(t, out var op)) infixOps.Add(op);
        }
        if (operands.Count < 2) return false;

        double value;
        if (infixOps.Count == operands.Count - 1 && infixOps.Count >= 1)
        {
            value = EvalPrecedence(operands, infixOps);             // "3 x 4 + 2", "7 + 4", "what is 3 plus 4"
        }
        else if (infixOps.Count == 0)
        {
            // Resolve the op-cue ONLY by the LEARNED cue→op relation (LearnArithmeticCue/ResolveLearnedOp) — no
            // hardcoded synonym list. A word the field has been TAUGHT ("add"/"total"/"sum"/"result"→×, a coined
            // operator) resolves by what it learned; an untaught word abstains. Real learning, no crutch.
            GliderOp? ResolveOp(string t)
                => _memory is DialecticalSpace d && ResolveLearnedOp(d, t, out var lop) ? lop : (GliderOp?)null;
            var cueOps = merged.Select(ResolveOp).Where(o => o.HasValue).Select(o => o!.Value).Distinct().ToList();
            if (cueOps.Count != 1) return false;                    // "the total of 3 and 4", "add 3 and 4", learned cues
            value = EvalFold(operands, cueOps[0]);
        }
        else return false;                                          // malformed operand/operator shape

        if (double.IsNaN(value)) return false;

        // Output FORM emerges from the prompt: a "in words" cue asks for the word, otherwise the number (the
        // value-grader accepts either for non-surface-strict skills; ArithToWord is surface-strict → must be the word).
        if (merged.Any(IsToWordCue))
        {
            var word = NumberWordVocabulary.ToWords((long)Math.Round(value)); // generative linguistic codec
            return EmitField(word, "field-compute-word", request, out result);
        }
        return EmitField(FieldFormat(value), "field-compute", request, out result);
    }

    // Field tokenizer: separate operator symbols from digits ("1+1" → "1 + 1"), strip punctuation, lowercase — so a
    // compact form tokenises like the spaced one. Shared by the dispatch and the cue learner.
    private static List<string> TokenizeField(string? input)
    {
        var prepped = input ?? string.Empty;
        foreach (var sym in new[] { "+", "-", "*", "/", ">", "<", "=" })
            prepped = prepped.Replace(sym, $" {sym} ");
        return prepped
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim('?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '"', '\'').ToLowerInvariant())
            .Where(t => t.Length > 0)
            .ToList();
    }

    // UNARY SIGN re-merge: a +/- in OPERAND position (start, or after an operator/cue/separator — not right after a
    // number) is the SIGN of the following number, not a binary op. The tokenizer split every '-' into its own token,
    // which made "-7 plus -1" / "6 - -5" look malformed. An expect-operand walk re-merges the negative literals so the
    // homomorphism (and the cue learner) read the operands correctly.
    private static List<string> SignMerge(IReadOnlyList<string> toks)
    {
        var merged = new List<string>(toks.Count);
        var expectOperand = true;
        for (var i = 0; i < toks.Count; i++)
        {
            var t = toks[i];
            if ((t == "-" || t == "+") && expectOperand && i + 1 < toks.Count
                && double.TryParse(toks[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var signed))
            {
                merged.Add((t == "-" ? -signed : signed).ToString(CultureInfo.InvariantCulture));
                i++; expectOperand = false; continue;
            }
            merged.Add(t);
            expectOperand = !double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out _);
        }
        return merged;
    }

    // The four arithmetic OPERATIONS — the irreducible MATH the substrate computes via the homomorphism. Their NAMES
    // (words AND symbols) are NOT hardcoded: the cue for each is LEARNED as a relation to these internal "∘" anchors,
    // so "result"/"ratio"/"+"/a novel coined operator all resolve by what the field LEARNED, none baked in.
    private static readonly (string Anchor, GliderOp Op)[] OpAnchors =
        { ("∘add", GliderOp.Add), ("∘sub", GliderOp.Subtract), ("∘mul", GliderOp.Multiply), ("∘div", GliderOp.Divide) };

    /// <summary>LEARN an arithmetic cue from ONE example — the genesis "op classified from context", done in the field
    /// (not the GRU's stateless head, not a hardcoded list): infer which of the four operations reproduces the answer
    /// from the operands, then relate the example's cue token(s) to that operation's anchor in the space ("the result
    /// of 8 and 5" → 40 ⇒ multiply ⇒ relate result ↔ ∘mul). At inference the field resolves the op by this learned
    /// relation, self-conditioned, generalising to any word/symbol/synonym. Conversational/training-time only.</summary>
    public void LearnArithmeticCue(string input, string output)
    {
        if (_memory is not DialecticalSpace ds || string.IsNullOrWhiteSpace(input)) return;
        var inv = CultureInfo.InvariantCulture;
        const NumberStyles ns = NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint;
        var merged = SignMerge(TokenizeField(input)); // sign-merged so "added to -7" reads operand -7, not +7
        // Skip examples with an EXPLICIT operator token ("sub 3 -2" tokenises as 3, -, 2 — the op is ambiguous between
        // a cue-fold and an infix expression, and learning from it taught the wrong op). Learn only clean cue examples.
        if (merged.Any(t => TryOpToken(t, out _))) return;
        var nums = merged.Where(t => double.TryParse(t, ns, inv, out _)).Select(t => double.Parse(t, ns, inv)).ToList();
        if (nums.Count != 2 || !double.TryParse(output.Trim(), ns, inv, out var answer)) return;

        // Learn ONLY from an UNAMBIGUOUS example — exactly one operation reproduces the answer. Skip cases like
        // "sub 8 0" (add ≡ sub) that would teach a confused cue→op relation (the bug that made "sub" compute as add).
        var matches = OpAnchors.Where(a => Math.Abs(FieldStep(nums[0], a.Op, nums[1]) - answer) < 1e-6).ToList();
        if (matches.Count != 1) return;
        var anchor = matches[0].Anchor;
        // Relate EVERY non-operand token to the op anchor — no filler pre-filter. An op cue ("plus") maps to ONE op
        // across examples; a framing word ("what"/"is") appears with ALL ops, so it accrues COMPETING op relations and
        // ResolveLearnedOp ABSTAINS on it. The competing-op abstention is the general de-hardcoded hygiene here — a
        // centrality filter would wrongly drop "plus" too (an op cue is distributionally as spread as a framing word).
        foreach (var t in merged)
            if (!IsNumericLike(t))
                ds.FineEditFromExample(new[] { t }, new[] { anchor }, isNegativeExample: false);
    }

    // Resolve a cue token to an operation by its LEARNED "∘" op-anchor relations. ABSTAINS (returns false) when the
    // cue maps to COMPETING operations — honest over a confident wrong answer (the sub→{add,sub} contamination case).
    private static bool ResolveLearnedOp(DialecticalSpace ds, string token, out GliderOp op)
    {
        op = default;
        var ops = ds.GetNeighbors(token, PlatonicNeighborhoodType.Relational, maxNeighbors: 12, minConfidence: 0.0)
            .Where(n => n.Concept.StartsWith("∘", StringComparison.Ordinal))
            .Select(n => (Op: OpAnchors.First(a => a.Anchor == n.Concept).Op, n.Confidence))
            .OrderByDescending(x => x.Confidence).ToList();
        if (ops.Count == 0) return false;
        if (ops.Any(x => x.Op != ops[0].Op && x.Confidence >= ops[0].Confidence - 1e-6)) return false; // ambiguous → abstain
        op = ops[0].Op;
        return true;
    }

    // ── NUMBER-WORD: single value ↔ word via the codec ("5 in words" → five ; "five as a number" → 5). ───────────
    private bool TryFieldNumberWord(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        // digit → word
        if (toks.Any(IsToWordCue))
        {
            var digits = toks.Where(t => long.TryParse(t, out _)).ToList();
            if (digits.Count == 1)
            {
                var word = NumberWordVocabulary.ToWords(long.Parse(digits[0], CultureInfo.InvariantCulture));
                return EmitField(word, "field-numberword", request, out result);
            }
        }
        // word → digit
        if (toks.Any(IsToDigitCue))
        {
            var wordRun = toks.Where(IsNumberWord).ToList();
            if (wordRun.Count >= 1 && TryWordsToNumber(wordRun, out var v))
                return EmitField(v.ToString(CultureInfo.InvariantCulture), "field-numberword", request, out result);
        }
        return false;
    }

    // ── LEARNED FUNCTION (the field's GENERATIVE arm — composition + learned transforms, restored from the legacy
    //    ladder, PLATONIC_RECKONING.md keep-core). A persisted learned transform T(f) = avg(out−in) is applied to a
    //    NOVEL operand by COMPOSITION (embed(x)+T(f), decoded in its preferred numeric face) — true generalisation,
    //    not stored-pair lookup. The op element is the cue OR a relational neighbour of it (selected from the SPACE,
    //    no plan/op classifier). Unary → TransformAccumulator; binary → a discovered FoldPathDiscovery structure.
    //    Structure-gated: 1 or 2 numeric operands + a non-function-word cue. Inert when no transforms have been learned.
    private bool TryFieldLearnedFunction(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if ((_transformAccumulator is null && _foldPathDiscovery is null) || string.IsNullOrWhiteSpace(request.Input))
            return false;

        var inv = CultureInfo.InvariantCulture;
        const NumberStyles numStyle = NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint;
        // Raw whitespace tokens (NOT the operator-split toks) so a signed literal like "-3" stays one operand.
        var tokens = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var operands = tokens.Where(t => double.TryParse(t, numStyle, inv, out _)).ToArray();
        if (operands.Length is not (1 or 2)) return false;
        // No filler pre-filter on the cues: a framing word ("what"/"is") resolves to NO op (or competing ops →
        // abstains) below, while an op cue ("plus") resolves to one — so non-op words harmlessly drop out of resolution.
        var cues = tokens.Where(t => t.Any(char.IsLetter)).Select(t => t.ToLowerInvariant().Trim('?', '!', '.', ',', ';', ':'))
                         .Where(t => t.Length > 0).ToArray();
        if (cues.Length == 0) return false;

        var ds = (DialecticalSpace)_memory;
        var dim = _memory.FaceDimension;
        foreach (var cue in cues)
        {
            // The op is the cue itself OR a learned relational neighbour (an edge in the space) — retrieval, not a
            // name table. First candidate that carries a learned op wins.
            var candidates = new List<string>(5) { cue };
            candidates.AddRange(ds.GetNeighbors(cue, PlatonicNeighborhoodType.Relational, maxNeighbors: 4, minConfidence: 0.35)
                                  .Select(n => n.Concept));
            foreach (var fn in candidates)
            {
                if (operands.Length == 1 && _transformAccumulator is not null
                    && _transformAccumulator.TryGetTransform(fn, out var transform))
                {
                    var predicted = _transformAccumulator.Apply(fn, InputEmbeddingComposer.GetInputEmbedding(operands[0], dim));
                    if (predicted is null) continue;
                    var (value, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(predicted, dim, transform.PreferredFace);
                    if (face != "none" && quality > 0.50)
                        return EmitField(FieldFormat(value), $"field-transform:{fn}", request, out result);
                    continue;
                }
                if (operands.Length == 2 && _foldPathDiscovery is not null && _foldPathDiscovery.HasOperation(fn)
                    && double.TryParse(operands[0], numStyle, inv, out var a) && double.TryParse(operands[1], numStyle, inv, out var b)
                    && _foldPathDiscovery.TryPredict(fn, a, b, out var predValue, out _))
                    return EmitField(FieldFormat(predValue), $"field-fold:{fn}", request, out result);
            }
        }
        return false;
    }

    // ── GENERATIVE TICK (Stage 1 — the genesis tick brought live, hand-directed). The query is a FRONTIER of items
    //    (numeric VALUEs + learned-op CUEs). Each TICK selects an applicable (cue, value) adjacency, APPLIES the
    //    learned transform — manufacturing a NEW intermediate value element — and collapses the pair; the new value
    //    re-enters the frontier as the next operand (the cascade). Settles when one value remains. This BUILDS a
    //    multi-step derivation the one-shot dispatch cannot: "double incr 5" → incr(5)=6 → double(6)=12, across ticks.
    //    Selection here is a hand-σ (innermost-first); Stage 2 hands selection to the NN (the director).
    private bool TryFieldTick(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (_transformAccumulator is null || string.IsNullOrWhiteSpace(request.Input)) return false;
        var ds = (DialecticalSpace)_memory;
        var dim = _memory.FaceDimension;
        var inv = CultureInfo.InvariantCulture;
        const NumberStyles ns = NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint;

        // Build the frontier: each token is a numeric VALUE, a CUE resolving to a learned transform (the op element,
        // selected from the SPACE — cue itself or a relational neighbour), or filler (dropped).
        var items = new List<(bool IsValue, double Value, string Fn, Transform T)>();
        foreach (var tok in request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            var t = tok.Trim('?', '!', '.', ',', ';', ':');
            if (double.TryParse(t, ns, inv, out var v)) { items.Add((true, v, "", default!)); continue; }
            var lc = t.ToLowerInvariant();
            if (t.Any(char.IsLetter) && ResolveTransform(ds, lc, out var fn, out var tr)) // a framing word resolves to no transform → drops out
                items.Add((false, 0, fn, tr));
        }
        // Stage-1 scope: a single operand fed through a CHAIN of >= 2 learned ops (the genuinely multi-step case the
        // one-shot learned-function route can't do — it would apply only one).
        if (items.Count(i => i.IsValue) != 1 || items.Count(i => !i.IsValue) < 2) return false;

        var trace = new List<string>();
        var guard = 0;
        while (items.Count > 1 && guard++ < 16)
        {
            var idx = -1;
            for (var i = 0; i + 1 < items.Count; i++)
                if (!items[i].IsValue && items[i + 1].IsValue) { idx = i; break; } // innermost applicable op
            if (idx < 0) break; // no (cue, value) adjacency — frontier settled / stuck
            var cue = items[idx]; var val = items[idx + 1];
            var predicted = _transformAccumulator.Apply(cue.Fn, InputEmbeddingComposer.GetInputEmbedding(FieldFormat(val.Value), dim));
            if (predicted is null) return false;
            var (nv, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(predicted, dim, cue.T.PreferredFace);
            if (face == "none" || quality <= 0.50) return false;
            trace.Add($"{cue.Fn}({FieldFormat(val.Value)})={FieldFormat(nv)}"); // the manufactured intermediate element
            items[idx] = (true, nv, "", default!);
            items.RemoveAt(idx + 1);
        }
        if (trace.Count < 2) return false; // a single step is the one-shot route's job
        var settled = items.First(i => i.IsValue);
        return EmitField(FieldFormat(settled.Value), $"field-tick[{string.Join(" -> ", trace)}]", request, out result);
    }

    // Resolve a cue to a learned transform: the cue itself OR a relational neighbour of it (an op edge in the space).
    private bool ResolveTransform(DialecticalSpace ds, string cue, out string fn, out Transform transform)
    {
        fn = cue; transform = default!;
        if (_transformAccumulator is null) return false;
        var candidates = new List<string>(5) { cue };
        candidates.AddRange(ds.GetNeighbors(cue, PlatonicNeighborhoodType.Relational, maxNeighbors: 4, minConfidence: 0.35)
                              .Select(n => n.Concept));
        foreach (var c in candidates)
            if (_transformAccumulator.TryGetTransform(c, out transform)) { fn = c; return true; }
        return false;
    }

    // ── ANALOGY in the large (meaning) face — relation-vector arithmetic over the distributional cloud. Detects the
    //    "A is to B as C is to …" shape (split on "as"; content words left = the example pair(s), the first content
    //    word right = the query) and completes it via the space's generative Analogy primitive. No classifier — the
    //    answer is built by vector arithmetic in meaning-space, not retrieved from a stored table.
    private bool TryFieldAnalogy(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var asIdx = -1;
        for (var i = 0; i < toks.Count; i++) if (toks[i] == "as") { asIdx = i; break; }
        if (asIdx <= 0 || asIdx >= toks.Count - 1) return false;
        var left = toks.Take(asIdx).Where(IsContentWord).ToList();
        var right = toks.Skip(asIdx + 1).Where(IsContentWord).ToList();
        if (left.Count < 2 || right.Count < 1) return false;

        var pairs = new List<(string, string)>();
        for (var i = 0; i + 1 < left.Count; i += 2) pairs.Add((left[i], left[i + 1]));
        if (pairs.Count == 0) return false;

        var t = ((DialecticalSpace)_memory).Analogy(pairs, right[0]);
        if (!t.Settled || string.IsNullOrEmpty(t.Symbol)) return false;
        return EmitField(t.Symbol, "field-analogy", request, out result);
    }

    // ── COMPOSE MEANINGS in the large face — combine the query's content concepts (relax over ALL their clouds, not
    //    just the discriminative one) to retrieve the concept that fits the COMBINATION ("red fruit" → apple). Fires
    //    only on a bare compositional phrase (no question word, no retrieval-frame marker) of 2+ KNOWN content
    //    concepts — so it never competes with single-subject retrieval / disambiguation.
    private bool TryFieldComposeMeaning(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (toks.Any(IsQuestionCue) || toks.Any(IsRetrievalMarker)) return false;
        var ds = (DialecticalSpace)_memory;
        var content = toks.Where(t => IsContentWord(t) && !IsFiller(ds, t) && !IsNumericLike(t) && ds.ContainsConcept(t))
                          .Distinct(StringComparer.Ordinal).ToList();
        if (content.Count < 2) return false;

        var t2 = ds.Reason(content);
        if (!t2.Settled || t2.Confidence < ReasonMinConfidence
            || PlatonicSpaceMemory.IsReservedConcept(t2.Symbol) || ds.IsOperationToken(t2.Symbol)) return false;
        if (!DirectorApprovesCompose(ds, content, t2.Confidence)) return false; // the learned director's call
        return EmitField(t2.Symbol, "field-compose", request, out result);
    }

    // ── THE TICK OVER MEANING — the genesis cascade in the LARGE face. A frontier of content concepts is reduced over
    //    ticks: each tick COMPOSES the first two concepts (relax over both clouds) into the concept that fits them,
    //    MANUFACTURING a new MEANING element that re-enters the frontier as the next operand. So "red fruit dessert"
    //    builds (red+fruit)=apple, then (apple+dessert)=pie — a multi-step CONCEPTUAL derivation a one-shot compose
    //    (which sums all clouds at once) can't isolate. Needs 3+ concepts (2 is the one-shot compose's job).
    private bool TryFieldMeaningTick(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (toks.Any(IsQuestionCue) || toks.Any(IsRetrievalMarker)) return false;
        var ds = (DialecticalSpace)_memory;
        var frontier = toks.Where(t => IsContentWord(t) && !IsFiller(ds, t) && !IsNumericLike(t) && ds.ContainsConcept(t))
                          .Distinct(StringComparer.Ordinal).ToList();
        if (frontier.Count < 3) return false;
        // The learned director gates the meaning cascade too (it is a compose, multi-step).
        if (!DirectorApprovesCompose(ds, frontier, ds.Reason(new[] { frontier[0], frontier[1] }).Confidence)) return false;

        var trace = new List<string>();
        var guard = 0;
        while (frontier.Count > 1 && guard++ < 16)
        {
            var a = frontier[0]; var b = frontier[1];
            var t = ds.Reason(new[] { a, b }); // compose the two meanings → the concept that fits both
            if (!t.Settled || string.IsNullOrEmpty(t.Symbol)
                || PlatonicSpaceMemory.IsReservedConcept(t.Symbol) || ds.IsOperationToken(t.Symbol)) return false;
            trace.Add($"({a}+{b})={t.Symbol}");      // the manufactured intermediate concept
            frontier[0] = t.Symbol; frontier.RemoveAt(1); // it replaces the pair and re-enters the frontier
        }
        if (trace.Count < 2) return false;
        return EmitField(frontier[0], $"field-meaning-tick[{string.Join(" -> ", trace)}]", request, out result);
    }

    // ── RESPOND (talk-by-chunk, the first conductor step): a conversational cue is answered by FOLLOWING its learned
    //    cue→reply RELATION to a reply chunk (not cloud-retrieval, which drifts to clustered cue words), with the SELF
    //    picking which reply — the persona's voice + variety as the self evolves. The repertoire is the chunks the
    //    word face holds; "talking" is the NN-conditioned selection among them.
    // DIRECT talk: a conversational cue with a learned cue→reply CHUNK relation. Runs BEFORE relaxation. Requires a
    // multi-word reply CHUNK (the persona speaks in whole phrases) — so a framed RETRIEVAL query ("a synonym for big",
    // "what kind of thing is apple"), whose only relations are to single words, finds NO chunk here and falls through
    // to relaxation (which answers it correctly). Only a genuine persona cue, related to a reply phrase, talks.
    private bool TryFieldRespondDirect(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var ds = (DialecticalSpace)_memory;
        // The cue key is the WHOLE utterance first (a multi-word cue "good morning" is stored as one composite — its
        // reply relation lives there, not on the decomposed words), then the discriminative anchor as a fallback.
        var keys = new List<string>();
        var whole = (request.Input ?? string.Empty).Trim();
        if (whole.Length > 0) keys.Add(whole);
        keys.AddRange(PlatonicConceptAnchors.ExtractSpecific(ds, whole).Where(a => !IsFiller(ds, a)));

        foreach (var key in keys.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            // ONLY multi-word CHUNK replies (the persona's stored reply PHRASES). No single-token fallback: a stray
            // cue→word edge (gym decode noise) or a skill query's single-word answer must NOT be spoken as a reply.
            var candidates = ds.GetNeighbors(key, PlatonicNeighborhoodType.Relational, maxNeighbors: 16, minConfidence: 0.0)
                .Where(n => n.Concept.IndexOf(' ') >= 0
                         && !n.Concept.Equals(key, StringComparison.OrdinalIgnoreCase) && !IsNumericLike(n.Concept)
                         && !PlatonicSpaceMemory.IsReservedConcept(n.Concept) && !ds.IsOperationToken(n.Concept))
                .OrderByDescending(n => n.Confidence).ToList();
            if (candidates.Count == 0) continue;

            // The SELF picks the voice: among the cue's replies NOT just said (anti-repetition — an asshole doesn't
            // loop one line), the one whose meaning best fits who the mind has become.
            var pool = candidates.Where(c => !RecentlySaid(c.Concept)).ToList();
            if (pool.Count == 0) pool = candidates;
            var pick = pool[0].Concept;
            if (_selfField is not null && pool.Count > 1)
            {
                var best = double.NegativeInfinity;
                foreach (var c in pool)
                {
                    var v = ds.SemanticVectorOf(c.Concept);
                    if (v is null) continue;
                    var sim = 0.0; for (var i = 0; i < v.Length && i < _selfField.Length; i++) sim += v[i] * _selfField[i];
                    if (sim > best) { best = sim; pick = c.Concept; }
                }
            }
            Spoke(pick); PerceiveIntoSelfField(ds, pick); // the mind becomes what it SAYS — the persona builds in the self
            return EmitField(pick, "field-respond", request, out result);
        }
        return false;
    }

    // PERSONALITY GENERALISATION: an UNSEEN cue with no learned reply relation — runs AFTER relaxation, so a query
    // relaxation CAN answer (a synonym/category lookup) is never hijacked; only an input nothing else settled reaches
    // here. A personality isn't a lookup: if the self has a character (it has been saying rude things), say the reply
    // CHUNK nearest that self — the asshole insults whatever you said because its SELF is the asshole.
    private bool TryFieldRespondGeneralize(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var ds = (DialecticalSpace)_memory;
        var voice = NearestChunkToSelf(ds);
        if (voice is null) return false;
        Spoke(voice); PerceiveIntoSelfField(ds, voice);
        return EmitField(voice, "field-respond-self", request, out result);
    }

    // The persona's working memory of what it just said — so it ROTATES through its repertoire instead of looping —
    // and the GROWING set of every reply it has actually spoken (its earned repertoire). Generalisation draws ONLY
    // from this set, so it can never blurt out a CUE chunk ("good morning") that merely sits near the self; it says
    // something it has genuinely said as a reply before.
    private readonly List<string> _recentReplies = new();
    private readonly HashSet<string> _spokenReplies = new(StringComparer.OrdinalIgnoreCase);
    private bool RecentlySaid(string s) => _recentReplies.Contains(s, StringComparer.OrdinalIgnoreCase);
    private void Spoke(string s) { _recentReplies.Add(s); while (_recentReplies.Count > 4) _recentReplies.RemoveAt(0); _spokenReplies.Add(s); }

    // The reply the persona has SPOKEN that is nearest the current self, excluding what it just said — "what the
    // persona would say next". Drawn from its earned repertoire (replies only), never arbitrary nearby chunks.
    private string? NearestChunkToSelf(DialecticalSpace ds)
    {
        if (_selfField is null) return null;
        string? best = null; var bestSim = double.NegativeInfinity;
        foreach (var sym in _spokenReplies)
        {
            if (RecentlySaid(sym)) continue;
            var v = ds.SemanticVectorOf(sym);
            if (v is null) continue;
            var sim = 0.0; for (var i = 0; i < v.Length && i < _selfField.Length; i++) sim += v[i] * _selfField[i];
            if (sim > bestSim) { bestSim = sim; best = sym; }
        }
        return best;
    }

    // ── LEARN (continuity / the continuous I): the mind is TOLD a fact ("the password is plum") and remembers it,
    //    so a later question recalls it. This is the self conditioning cognition across time — the same mind using
    //    what it has lived. Conservative: fires only on a clear assertion (a copula, a complete content object, NO
    //    question word and NO retrieval-frame marker) so it never mistakes a gym question ("apple is a kind of") for
    //    a statement. Numbers never form relation edges (hard rule). Gated to the conscious-field (living) mode. ────
    private bool TryFieldLearn(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (toks.Count < 3) return false;
        if ((request.Input ?? string.Empty).Contains('?')) return false;
        if (toks.Any(IsQuestionCue) || toks.Any(IsRetrievalMarker)) return false; // a question / retrieval frame, not a statement

        var ds = (DialecticalSpace)_memory;
        // The subject KEY + asserted VALUE come ONLY from the LEARNED grammar roles (the NN structure recogniser),
        // word-order-free. NO hardcoded copula/possessive fallback: if the NN hasn't learned to parse this yet, the
        // mind simply doesn't learn the fact (honest — it must be taught the grammar, which the gym does). Real
        // learning only; the parser is the NN, not a word-list.
        if (RoleParse(toks, ds, out var subject, out var obj) != FrameKind.Assertion) return false;
        if (subject is null || obj is null || subject == obj) return false;
        if (IsNumericLike(subject) || IsNumericLike(obj)) return false; // numbers never form relation edges
        // BELIEF REVISION — staying coherent and CURRENT (genesis G2 non-contradiction; the free-energy principle:
        // update the model when reality contradicts it). A fresh assertion makes `obj` the subject's CURRENT belief, so
        // the mind WEAKENS the subject's prior (now-contradicted) beliefs and does not keep answering a stale truth.
        // The world changed; a living mind changes with it. G6: weakened toward dormancy, never destroyed — history
        // persists and re-asserting the old fact brings it back. (Conversational only — the gym never learns here.)
        foreach (var n in ds.GetNeighbors(subject, PlatonicNeighborhoodType.Relational, maxNeighbors: 8, minConfidence: 0.0).ToList())
            if (!n.Concept.Equals(obj, StringComparison.OrdinalIgnoreCase) && !IsNumericLike(n.Concept))
                ds.DisruptAssociation(subject, n.Concept);
        ds.FineEditFromExample(new[] { subject }, new[] { obj }, isNegativeExample: false);
        PerceiveIntoSelfField(ds, subject); PerceiveIntoSelfField(ds, obj); // the mind becomes what it learns
        return EmitField(obj, "field-learn", request, out result); // acknowledge what it now holds
    }

    private static readonly System.Collections.Generic.HashSet<string> Framing =
        new(StringComparer.Ordinal) { "the", "a", "an", "of", "to", "is", "are", "was", "were", "my", "your", "his", "her", "its", "their", "that", "this" };
    private static bool IsQuestionCue(string t) => t is "what" or "who" or "where" or "when" or "why" or "which" or "how" or "whose";
    // Words that mark a gym RETRIEVAL frame (a question shaped like a statement), never a learnable assertion.
    private static bool IsRetrievalMarker(string t) => t is "kind" or "type" or "sort" or "group" or "category" or "example"
        or "classified" or "belongs" or "synonym" or "word" or "means" or "meaning" or "similar" or "same" or "close" or "match" or "near" or "like";
    private static bool IsNumericLike(string t) => double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out _);
    private static bool IsContentWord(string t) => t.Length > 1 && t.All(char.IsLetter) && !Framing.Contains(t);
    // FILLER — a token that must never leak out as a subject / answer / cue. The OPEN class of function words
    // (prepositions/conjunctions/articles: of/for/with/the/by/from…) is now LEARNED from the space's own centrality
    // distribution (DialecticalSpace.IsFunctionLike — a filler word's cloud collapses toward the global centroid) rather
    // than a hardcoded stoplist: the code is the general framework, the gym's training teaches WHAT counts as filler
    // (see FunctionWordResearch). The learned signal SELF-ABSTAINS in a cold/untrained space, so interrogatives stay a
    // tiny structural floor — a question word is never an answer in ANY space, warm or cold.
    private static bool IsFiller(DialecticalSpace ds, string t) => IsQuestionCue(t) || ds.IsFunctionLike(t);


    private enum FrameKind { None, Assertion, Question }

    // WORD-ORDER-FREE parse from the LEARNED NN ROLES ONLY (the structure recogniser): identify the subject KEY and
    // asserted VALUE by the NN's per-token roles — no copula/question/possessive list, no centrality fallback, no
    // position rule. Returns None when the NN hasn't learned to parse this (untrained/unsure) — then the mind simply
    // doesn't act on it. The parser IS the NN; if it generalises, that is the LEARNED model, with nothing else helping.
    private FrameKind RoleParse(IReadOnlyList<string> toks, DialecticalSpace ds, out string? subject, out string? value)
    {
        subject = value = null;
        // Only personal-fact frames: abstain on arithmetic/operator frames AND the gym's RETRIEVAL frames (synonym/
        // category/etc.) so the grammar parse never hijacks a skill query — those keep their normal retrieval route.
        if (toks.Count < 2 || toks.Any(IsNumericLike) || toks.Any(ds.IsOperationToken) || toks.Any(IsRetrievalMarker)) return FrameKind.None;
        var nn = NnRoles(toks);
        if (nn is null) return FrameKind.None;                        // NN untrained / not aligned → no parse (honest)
        Cognition.GrammarRoleLearner.Role RoleAt(int i) => nn[i];     // PURELY the NN's tag

        var hasQuery = false;
        var queryIdx = new List<int>();
        var content = new List<int>();
        for (var i = 0; i < toks.Count; i++)
        {
            var t = toks[i];
            if (!t.All(char.IsLetter) || t.Length < 2) continue;
            var role = RoleAt(i);
            if (role == Cognition.GrammarRoleLearner.Role.Query) { hasQuery = true; queryIdx.Add(i); continue; }
            if (role == Cognition.GrammarRoleLearner.Role.Filler) continue; // the NN says filler/determiner/copula
            content.Add(i);                              // a content token (key / value / unsure-but-content)
        }
        if (content.Count == 0) return FrameKind.None;

        // COPULA-POSITION BOUNDING. A SUBJECT can span several tokens ("my favorite color"), and the per-token role head
        // sometimes mis-tags the phrase's final noun as VALUE — collapsing the span ("my favorite") so the assert key no
        // longer matches the recall key. The copula (is/was/…, which the NN tags reliably as Filler) is the true
        // boundary: in an ASSERTION the subject is the content BEFORE it and the value the content AFTER it; in a
        // QUESTION the subject is the content AFTER it. This still uses ONLY the learned tags (no copula word-list) —
        // it just trusts the reliable Filler boundary over a fragile per-token KEY/VALUE split for the span. Falls back
        // to the pure-tag split when there is no copula (e.g. "whats my name", a cue with no copula).
        bool IsWord(int i) => i >= 0 && i < toks.Count && toks[i].Length >= 2 && toks[i].All(char.IsLetter);
        string? Span(int from, int step) // contiguous word run from `from` walking by `step`, stopping at a query cue
        {
            int b = -1, e = -1;
            for (var i = from; IsWord(i) && RoleAt(i) != Cognition.GrammarRoleLearner.Role.Query; i += step)
            { if (b < 0) { b = e = i; } else { b = Math.Min(b, i); e = Math.Max(e, i); } }
            return b < 0 ? null : string.Join(" ", Enumerable.Range(b, e - b + 1).Select(k => toks[k]));
        }
        if (hasQuery)
        {
            for (var i = queryIdx[^1] + 1; i < toks.Count; i++)
                if (RoleAt(i) == Cognition.GrammarRoleLearner.Role.Filler && content.Any(c => c > i))
                { subject = Span(i + 1, +1); break; } // subject = the phrase AFTER the copula
            if (subject is not null) return FrameKind.Question;
        }
        else
        {
            var lastContent = content[^1];
            for (var i = lastContent - 1; i >= 0; i--)
                if (RoleAt(i) == Cognition.GrammarRoleLearner.Role.Filler && content.Any(c => c < i))
                { value = toks[lastContent]; subject = Span(i - 1, -1); break; } // value AFTER, subject BEFORE the copula
            if (subject is not null && subject != value) return FrameKind.Assertion;
            subject = value = null; // the copula path didn't settle → try the pure-tag split below
        }

        // FALLBACK — no copula: split by the NN's KEY/VALUE tags alone.
        var keyIdx = content.Where(i => RoleAt(i) == Cognition.GrammarRoleLearner.Role.Key).ToList();
        var valIdx = content.Where(i => RoleAt(i) == Cognition.GrammarRoleLearner.Role.Value).ToList();
        if (keyIdx.Count == 0 && valIdx.Count == 0) return FrameKind.None; // the NN placed no key/value → no parse
        if (hasQuery) // a query cue is present → it's a question; the subject is the KEY (or the lone content)
        {
            var ni = keyIdx.Count > 0 ? keyIdx[^1] : content[^1];
            subject = BuildSubjectPhrase(toks, ni, nn);
            return subject is not null ? FrameKind.Question : FrameKind.None;
        }
        // ASSERTION: the VALUE is the asserted thing; the KEY is the subject. Either may be inferred when the other is
        // known and there are exactly two content tokens (a new value like "zorptron" the NN/alignment hasn't placed).
        int? vi = valIdx.Count > 0 ? valIdx[^1] : (content.Count == 2 && keyIdx.Count == 1 ? content.First(i => i != keyIdx[0]) : (int?)null);
        int? ki = keyIdx.Count > 0 ? keyIdx[^1] : (content.Count == 2 && valIdx.Count == 1 ? content.First(i => i != valIdx[0]) : (int?)null);
        if (vi is null || ki is null || vi == ki) return FrameKind.None;
        value = toks[vi.Value];
        subject = BuildSubjectPhrase(toks, ki.Value, nn);
        return subject is not null && subject != value ? FrameKind.Assertion : FrameKind.None;
    }

    // The NN STRUCTURE RECOGNISER's per-token roles, aligned to the field tokens. Grammar frames are number-free so the
    // model tokenization matches TokenizeField; returns null if the head is untrained or the tokenizations don't align,
    // and Unknown for a token the NN isn't confident about (→ centrality fallback). Maps the head's class ids to the
    // role enum (0=NONE→Filler, 1=SUBJECT→Key, 2=VALUE→Value, 3=QUERY→Query — the same map as DeriveRoleLabels).
    private const double NnRoleMinConfidence = 0.5;
    private Cognition.GrammarRoleLearner.Role[]? NnRoles(IReadOnlyList<string> toks)
    {
        if (toks.Count == 0) return null;
        var ids = _tokenizer.Encode(string.Join(" ", toks)).ToArray();
        if (ids.Length != toks.Count) return null;
        var pred = _model.PredictRoles(ids);
        if (pred.Length != toks.Count) return null;
        var roles = new Cognition.GrammarRoleLearner.Role[toks.Count];
        for (var i = 0; i < toks.Count; i++)
            roles[i] = pred[i].Confidence >= NnRoleMinConfidence
                ? pred[i].Role switch
                {
                    0 => Cognition.GrammarRoleLearner.Role.Filler,
                    1 => Cognition.GrammarRoleLearner.Role.Key,
                    2 => Cognition.GrammarRoleLearner.Role.Value,
                    3 => Cognition.GrammarRoleLearner.Role.Query,
                    _ => Cognition.GrammarRoleLearner.Role.Unknown,
                }
                : Cognition.GrammarRoleLearner.Role.Unknown;
        return roles;
    }

    // The subject PHRASE = the maximal CONTIGUOUS SUBJECT(Key) span (per the NN's tags) containing the noun. The
    // determiner ("my"/"your") is itself tagged SUBJECT by the NN (it co-occurs in both assert and recall inputs), so
    // the span naturally keeps the possessor — "my name" stays distinct from "your name", learned, no possessive list.
    private string? BuildSubjectPhrase(IReadOnlyList<string> toks, int nounIndex, Cognition.GrammarRoleLearner.Role[] nn)
    {
        if (nounIndex < 0 || nounIndex >= toks.Count) return null;
        var begin = nounIndex;
        var end = nounIndex;
        while (begin - 1 >= 0 && nn[begin - 1] == Cognition.GrammarRoleLearner.Role.Key) begin--;
        while (end + 1 < toks.Count && nn[end + 1] == Cognition.GrammarRoleLearner.Role.Key) end++;
        return string.Join(" ", Enumerable.Range(begin, end - begin + 1).Select(i => toks[i]));
    }

    // The mind's recent FOCUS — the content it has been attending to, threaded across thoughts (the continuous I).
    // It conditions reasoning as a tiebreaker for ambiguous queries.
    private readonly List<string> _focus = new();
    private const int FocusSize = 4;

    // The LEARNED GRAMMAR — structural roles (copula/query/key/value) inferred from the assert/recall alignment of
    // training examples, NOT hardcoded word-lists (see GrammarRoleLearner / nova-learned-grammar-roles). Shared with
    // the trainer via ObserveGrammar; consulted by the field parser via GrammarRole. Warmed by the gym's grammar
    // curriculum; abstains (Unknown) until warm, the same warm-start stance as the filler signal.
    private readonly Cognition.GrammarRoleLearner _grammar = new();

    /// <summary>Observe one training example's ASSERT/RECALL structure so the grammar roles are learned. Text-only:
    /// arithmetic/operator frames have their own routes and would only add noise to the role tallies. Called from the
    /// trainer's ObserveLearningSignals, alongside the op-cue learner.</summary>
    public void ObserveGrammar(string input, string output)
    {
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output)) return;
        if (_memory is not DialecticalSpace ds) return;
        var toks = TokenizeField(input);
        // Only clean ASSERT/RECALL fact frames: skip arithmetic/operator frames AND the gym's RETRIEVAL frames (a
        // category/synonym question like "apple is a kind of fruit" would otherwise feed the structural token "is"
        // spurious answer-absent counts and misclassify it). IsRetrievalMarker gates the gym's question frames here —
        // a training-frame filter, not parse logic.
        if (toks.Count == 0 || toks.Any(IsNumericLike) || toks.Any(ds.IsOperationToken) || toks.Any(IsRetrievalMarker)) return;
        _grammar.Observe(toks, output.Trim());
    }

    // The token's LEARNED structural role, resolved against the space's learned FILLER signal (IsFunctionLike).
    private Cognition.GrammarRoleLearner.Role GrammarRole(DialecticalSpace ds, string token)
        => _grammar.Classify(token, ds.IsFunctionLike);

    /// <summary>DIAGNOSTIC: the NN recogniser's per-token role tag + confidence for an input (untrained → empty/low),
    /// so a test can see whether the NN learned to recognise structure, separately from retrieval. 0=NONE 1=SUBJECT
    /// 2=VALUE 3=QUERY.</summary>
    public IReadOnlyList<(string Token, int Role, double Confidence)> DiagnoseRoles(string input)
    {
        if (_memory is not DialecticalSpace) return System.Array.Empty<(string, int, double)>();
        var toks = TokenizeField(input ?? string.Empty);
        var ids = _tokenizer.Encode(string.Join(" ", toks)).ToArray();
        var pred = ids.Length == toks.Count ? _model.PredictRoles(ids) : System.Array.Empty<(int, double)>();
        var result = new List<(string, int, double)>();
        for (var i = 0; i < toks.Count; i++)
            result.Add((toks[i], i < pred.Length ? pred[i].Item1 : -1, i < pred.Length ? pred[i].Item2 : 0.0));
        return result;
    }

    /// <summary>SELF-SUPERVISED per-token ROLE LABELS for training the NN recogniser — one class per input token
    /// (0=NONE/filler, 1=SUBJECT, 2=VALUE, 3=QUERY) from the learned assert/recall alignment, or a negative "ignore"
    /// for a token whose role isn't settled yet. Null for a non-grammar frame (numeric/operator/retrieval) or when
    /// nothing is confident — so the trainer only role-supervises clean fact frames once the alignment has warmed.
    /// Aligned to TokenizeField, which matches the model tokenization for the number-free frames this fires on.</summary>
    public int[]? DeriveRoleLabels(string input)
    {
        if (_memory is not DialecticalSpace ds) return null;
        var toks = TokenizeField(input);
        if (toks.Count == 0 || toks.Any(IsNumericLike) || toks.Any(ds.IsOperationToken) || toks.Any(IsRetrievalMarker)) return null;
        var labels = new int[toks.Count];
        var any = false;
        for (var i = 0; i < toks.Count; i++)
        {
            // CENTRALITY-FREE labels straight from the alignment counters (LabelFor) — robust where the geometric
            // role classifier is fragile. -1 = ignore (the per-token CE skips it).
            labels[i] = _grammar.LabelFor(toks[i]);
            if (labels[i] >= 0) any = true;
        }
        return any ? labels : null;
    }

    // THE PERSISTENT SELF, in the mind's own meaning-space (PLATONIC_CONSCIOUSNESS.md / "a self that LEARNS, not a
    // learning thing with a self tacked on"). Where _focus is the discrete last-N attention (working memory, evicted
    // by distraction), this is the CONTINUOUS standing wave of what the mind has been living — a decaying accumulation
    // of the MEANING of every concept it attends to. It survives intervening unrelated thoughts (decay, not eviction),
    // so the mind reasons from WHO IT HAS BECOME even after distraction. It is built from concept clouds, which the
    // long training run sharpens — so the self the mind reasons from is itself shaped by learning. Null before the
    // first perception (no self before life). It conditions ONLY genuinely ambiguous reasoning — a dominant known fact
    // is answered directly, never overridden by mood.
    private double[]? _selfField;
    private const double SelfDecay = 0.82;          // how much of the standing self survives each new perception
    private const double SelfReasonWeight = 0.6;    // the self's pull on an ambiguous query (anchor stays weight 1)

    /// <summary>When true (default, in the living field mode), the persistent self conditions ambiguous reasoning.
    /// Turn OFF to ABLATE the self — proving the agent's cognition genuinely DEPENDS on it (else it is decorative).</summary>
    public bool SelfConditionsCognition { get; set; } = true;

    /// <summary>GENERATIVE TICK LOOP (Stage 1, PLATONIC_MIND.md / the genesis tick). When true, the field can RUN a
    /// query as a cascade — a frontier of elements reduced over ticks, each tick APPLYING a learned op to a value and
    /// manufacturing a NEW intermediate element that re-enters the frontier (so multi-step derivations the one-shot
    /// dispatch can't reach are BUILT mid-inference). Selection is a hand-coded heuristic here; handing the wheel to
    /// the NN (the director σ) is Stage 2. Default OFF — the dispatch is byte-identical until the NN drives it.</summary>
    public bool FieldTicksEnabled { get; set; }

    /// <summary>CONVERSATION (talk-by-chunk, experimental). When true, a cue is answered by following its learned
    /// cue→reply RELATION to a reply chunk and letting the SELF pick the voice — instead of cloud-retrieval, which
    /// drifts to clustered cue words. The persona's repertoire is the chunks; the self is the character. Default OFF.</summary>
    public bool TalkEnabled { get; set; }

    /// <summary>GENERATIVE MEANING OPS in the LARGE (word) face — the field reasons IN meaning-space, not just retrieves
    /// from it: COMPOSE two meanings into the concept that fits both ("red fruit" → apple), and ANALOGY by relation-
    /// vector arithmetic ("paris is to france as tokyo is to" → japan). Uses the ~60% of the substrate that shallow
    /// retrieval ignored. Default OFF (byte-identical) until the detection is hardened and the NN can direct it.</summary>
    public bool MeaningOpsEnabled { get; set; }

    /// <summary>The mind's current self as a meaning-space vector (empty before the first perception) — for inspection
    /// and for tests that ablate or probe the self.</summary>
    public IReadOnlyList<double> SelfField => _selfField ?? Array.Empty<double>();

    /// <summary>THE LEARNED DIRECTOR (Stage 2) — gates the generative MEANING routes (compose / meaning-tick): it
    /// decides, from substrate features, whether a query genuinely WANTS to compose vs plain retrieval. Null = the raw
    /// routes fire un-gated (the experiment/test path). In production a conservative director is attached so the routes
    /// default to retrieval (safe) and open only as the director learns. See <see cref="FieldDirector"/>.</summary>
    public FieldDirector? Director { get; set; }

    // Substrate features for the director's compose-vs-retrieve decision on the query's content concepts.
    private bool DirectorApprovesCompose(DialecticalSpace ds, IReadOnlyList<string> content, double composeConf)
    {
        if (Director is null) return true; // no director attached → raw route (un-gated)
        var degs = content.Select(ds.GetRelationDegree).ToList();
        var minDeg = degs.Count > 0 ? degs.Min() : 0;
        var subject = content[Math.Max(0, degs.IndexOf(minDeg))];
        var retrieveConf = ds.Reason(new[] { subject }).Confidence; // what plain retrieval of the most-specific concept gives
        return Director.ShouldCompose(new[]
        {
            minDeg / (minDeg + 4.0), content.Count / 4.0, composeConf, retrieveConf, composeConf - retrieveConf
        });
    }

    // PERCEIVE — fold one attended concept's MEANING into the persistent self (decay the standing self, admit the new
    // meaning, renormalize). This is how living accumulates into a self: every thought leaves a trace, recent ones
    // weigh more, but nothing is sharply evicted. Unknown/numeric tokens carry no held meaning and are skipped.
    private void PerceiveIntoSelfField(DialecticalSpace ds, string concept)
    {
        var v = ds.SemanticVectorOf(concept);
        if (v is null) return;
        if (_selfField is null || _selfField.Length != v.Length) { _selfField = (double[])v.Clone(); }
        else for (var i = 0; i < v.Length; i++) _selfField[i] = SelfDecay * _selfField[i] + (1.0 - SelfDecay) * v[i];
        var n = 0.0; for (var i = 0; i < _selfField.Length; i++) n += _selfField[i] * _selfField[i];
        n = Math.Sqrt(n); if (n > 1e-9) for (var i = 0; i < _selfField.Length; i++) _selfField[i] /= n;
    }

    // ── RELAX: recall what the mind HOLDS about the subject. RELATION-FIRST (follow the explicit association — robust
    //    to hub dilution at scale, where a populous category's distributional cloud washes out a member's signal),
    //    context-disambiguated, falling back to semantic relaxation over the clouds when there is no held association. ─
    private bool TryFieldRelax(IReadOnlyList<string> toks, GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var ds = (DialecticalSpace)_memory;

        // PHRASE-SUBJECT FIRST — "what is my name" vs "what is your name" key on the whole noun PHRASE ("my name" /
        // "your name"), so the possessor distinguishes the two held facts. The subject phrase comes from the LEARNED
        // grammar ROLES when warm (word-order-free, no possessive list); cold it falls back to the structural
        // ContentNounPhrase. Only fires when the mind actually HOLDS a relation on that multi-word phrase; otherwise it
        // falls through to the discriminative single anchor below (gym single-word retrieval is untouched).
        // The subject phrase comes ONLY from the LEARNED grammar roles (the NN). No hardcoded ContentNounPhrase fallback.
        var phrase = RoleParse(toks, ds, out var qsub, out _) == FrameKind.Question ? qsub : null;
        if (phrase is not null && phrase.Contains(' ') && ds.ContainsConcept(phrase))
        {
            var prel = ds.GetNeighbors(phrase, PlatonicNeighborhoodType.Relational, maxNeighbors: 12, minConfidence: 0.0)
                .Where(n => !string.IsNullOrEmpty(n.Concept) && !PlatonicSpaceMemory.IsReservedConcept(n.Concept)
                         && !ds.IsOperationToken(n.Concept) && !n.Concept.Equals(phrase, StringComparison.Ordinal))
                .OrderByDescending(n => n.Confidence).ThenByDescending(n => n.ObservationCount).ToList();
            if (prel.Count > 0 && (prel.Count == 1 || prel[0].Confidence > prel[1].Confidence + 0.15))
            {
                PerceiveIntoSelfField(ds, phrase);
                return EmitPlatonicResult(prel[0].Concept, "field-relax", Math.Clamp(prel[0].Confidence, 0.0, 1.0),
                    hops: 1, request, evidence: null, out result);
            }
        }

        // Drop FUNCTION WORDS (the/of/to/is..., what/who/where...) from the candidate subjects: when an arithmetic or
        // other prompt falls through to here, its framing words must not become the query's subject — nor the answer.
        // Without this the relaxation seeded on "what" / "to" / "of" emitted those very words instead of abstaining.
        var anchors = PlatonicConceptAnchors.ExtractSpecific(ds, request.Input ?? string.Empty)
            .Where(a => !IsFiller(ds, a)).ToList();
        if (anchors.Count == 0) return false;
        var subject = anchors[0]; // the most-discriminative cue = the query's likely subject

        // A relation TARGET is legitimate even if its cloud is central — a populous CATEGORY hub ("animal", "vehicle")
        // sits near the centroid exactly like a filler word, but it IS a valid answer because a STRONG explicit relation
        // reaches it (a filler is never the target of one). So the filler-filter must NOT veto relation targets; it
        // guards only subject selection and the cloud-relaxation fallback (where filler can surface as noise).
        bool BadTarget(string s) => string.IsNullOrEmpty(s) || PlatonicSpaceMemory.IsReservedConcept(s)
            || ds.IsOperationToken(s) || s.Equals(subject, StringComparison.Ordinal);
        bool Bad(string s) => BadTarget(s) || IsFiller(ds, s);
        // Attending to a subject both threads the discrete focus AND folds its meaning into the persistent self —
        // the mind becomes, a little, what it has just thought about.
        void Attend() { _focus.Remove(subject); _focus.Add(subject); while (_focus.Count > FocusSize) _focus.RemoveAt(0); PerceiveIntoSelfField(ds, subject); }
        bool Valid(DialecticalSpace.Thought t) => t.Settled && t.Confidence >= ReasonMinConfidence && !Bad(t.Symbol);

        // RELATION-FIRST when the mind holds a DOMINANT explicit association — a single strong relation IS the answer,
        // and a relation does not dilute as the body grows, so a known fact stays recallable however large the space
        // gets (the cloud-only path silently rotted at scale — a member of a populous category fell below the settle
        // threshold and abstained). For COMPARABLE associations (genuine ambiguity) we fall through to the clouds.
        var rels = ds.GetNeighbors(subject, PlatonicNeighborhoodType.Relational, maxNeighbors: 12, minConfidence: 0.0)
                     .Where(n => !BadTarget(n.Concept))
                     .OrderByDescending(n => n.Confidence).ThenByDescending(n => n.ObservationCount).ToList();
        if (rels.Count > 0 && (rels.Count == 1 || rels[0].Confidence > rels[1].Confidence + 0.15))
        {
            Attend();
            return EmitPlatonicResult(rels[0].Concept, "field-relax", Math.Clamp(rels[0].Confidence, 0.0, 1.0),
                hops: 1, request, evidence: null, out result);
        }

        // AMBIGUOUS (several comparable associations) or none — relax over the clouds, the SELF tipping the basin
        // toward who the mind has become. The PERSISTENT self leads (it survives distraction); the sharp recent _focus
        // is the fallback for immediate context the standing self hasn't yet absorbed. Disambiguation is INDIRECT:
        // "cash" pulls toward the money sense of "bank", and it still does after several unrelated thoughts.
        var bare = ds.Reason(new[] { subject });
        var chosen = bare;
        var viaSelf = false;
        if (SelfConditionsCognition && _selfField is not null)
        {
            var withSelf = ds.Reason(new[] { subject }, selfContext: _selfField, selfWeight: SelfReasonWeight);
            if (Valid(withSelf) && withSelf.Symbol != bare.Symbol && withSelf.Confidence > bare.Confidence + 0.05)
            { chosen = withSelf; viaSelf = true; }
        }
        if (!viaSelf && _focus.Count > 0)
        {
            var ctx = new List<string> { subject };
            foreach (var f in _focus) if (f != subject && !ds.IsOperationToken(f)) ctx.Add(f);
            var withCtx = ds.Reason(ctx);
            if (Valid(withCtx) && withCtx.Symbol != bare.Symbol && withCtx.Confidence > bare.Confidence + 0.05)
            { chosen = withCtx; viaSelf = true; }
        }
        Attend();
        if (!Valid(chosen)) return false;
        return EmitPlatonicResult(chosen.Symbol, viaSelf ? "field-relax-self" : "field-relax", chosen.Confidence,
            Math.Max(1, chosen.Steps), request, evidence: null, out result);
    }

    // ── Reduction helpers (all classifier-free; the homomorphism does the compute) ────────────────────────────────

    // One binary step on the substrate (generalises to ANY operands — fresh numeric embedding, no stored fact).
    private double FieldStep(double l, GliderOp op, double r)
    {
        var g = new PlatonicGlider("step", new Compute(op, new GliderBlock[] { new Const(l), new Const(r) }));
        return double.TryParse(_glider.Execute(g, Array.Empty<string>()), NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v : double.NaN;
    }

    private double EvalPrecedence(List<double> operands, List<GliderOp> ops)
    {
        var vals = operands.ToList();
        var oo = ops.ToList();
        for (var i = 0; i < oo.Count;)
        {
            if (oo[i] is GliderOp.Multiply or GliderOp.Divide)
            {
                var res = FieldStep(vals[i], oo[i], vals[i + 1]);
                if (double.IsNaN(res)) return double.NaN;
                vals[i] = res; vals.RemoveAt(i + 1); oo.RemoveAt(i);
            }
            else i++;
        }
        var acc = vals[0];
        for (var i = 0; i < oo.Count; i++)
        {
            acc = FieldStep(acc, oo[i], vals[i + 1]);
            if (double.IsNaN(acc)) return double.NaN;
        }
        return acc;
    }

    private double EvalFold(List<double> operands, GliderOp op)
    {
        var acc = operands[0];
        for (var i = 1; i < operands.Count; i++)
        {
            acc = FieldStep(acc, op, operands[i]);
            if (double.IsNaN(acc)) return double.NaN;
        }
        return acc;
    }

    private bool EmitField(string answer, string path, GenerationRequest request, out GenerationResult result)
    {
        if (EmitPlatonicResult(answer, path, 1.0, hops: 1, request, evidence: null, out result))
        {
            RecordRouteDecision(1, 1, true, true, true, 1, result.DecisionPath, 1.0);
            return true;
        }
        return false;
    }

    private GenerationResult FieldAbstain()
    {
        var r = new GenerationResult(
            Output: string.Empty,
            GeneratedTokens: Array.Empty<int>(),
            UsedPlatonicQuery: false,
            UsedNeuralFallback: false,
            DecisionPath: "field-abstain",
            PlatonicConfidence: 0.0,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 0,
            PlatonicHopCount: 0);
        RecordRouteDecision(0, 0, true, false, false, 0, r.DecisionPath, 0.0);
        return r;
    }

    private static string FieldFormat(double value) =>
        Math.Abs(value - Math.Round(value)) <= 1e-9
            ? ((long)Math.Round(value)).ToString(CultureInfo.InvariantCulture)
            : value.ToString("0.##########", CultureInfo.InvariantCulture);

    // Operator SYMBOLS / unambiguous op-words (infix). 'x'/'*' = multiply only flanked by digits — handled by being
    // parsed only in the operator position of the alternating scan, so a stray "x" word can't become multiply.
    private static bool TryOpToken(string t, out GliderOp op)
    {
        op = t switch
        {
            "+" or "plus" => GliderOp.Add,
            "-" or "minus" => GliderOp.Subtract,
            "x" or "*" or "times" => GliderOp.Multiply,
            "/" or "over" => GliderOp.Divide,
            _ => (GliderOp)(-1),
        };
        return (int)op >= 0;
    }

    private static bool IsCompareCue(string t) => t is "compared" or "compare" or "bigger" or "larger"
        or "smaller" or "greater" or "less" or "lesser" or "next" or "versus" or "vs";

    private static bool IsToWordCue(string t) => t is "words" or "word" or "spell" or "written";
    private static bool IsToDigitCue(string t) => t is "number" or "numeral" or "digit" or "digits";

    private static bool IsNumberWord(string t) =>
        NumberWordVocabulary.WordToValue.ContainsKey(t) || t is "hundred" or "thousand";

    // Reverse of NumberWordVocabulary.ToWords — parse an English number phrase ("forty seven", "three hundred two").
    private static bool TryWordsToNumber(IReadOnlyList<string> words, out long value)
    {
        value = 0;
        long current = 0; var any = false;
        foreach (var w in words)
        {
            if (NumberWordVocabulary.WordToValue.TryGetValue(w, out var v)) { current += v; any = true; }
            else if (w == "hundred") { current = (current == 0 ? 1 : current) * 100; any = true; }
            else if (w == "thousand") { value += (current == 0 ? 1 : current) * 1000; current = 0; any = true; }
            else { value = 0; return false; }
        }
        value += current;
        return any;
    }
}
