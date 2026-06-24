using GenesisNova.Model;
using GenesisNova.Cognition;
using GenesisNova.Tokenization;
using GenesisNova.Core;

namespace GenesisNova.Infer;

public sealed partial class GenesisInferenceEngine
{
    /// <summary>
    /// The GRU-CONSTRUCTED platonic query: the model itself selects the face operation (learned op
    /// head, face-derived vocabulary, 0 = abstain) and the operand tokens (learned per-token head),
    /// and the platonic faces execute the query exactly. This is the learned successor to the compact
    /// arithmetic grammar — it handles framed natural-language arithmetic ("what is 1 + 1") because
    /// the GRU LEARNED which tokens are operands and which are framing, from the examples' own
    /// numeric structure. Ordered AFTER the exact compact parser (which stays authoritative for bare
    /// expressions) and BEFORE the concept-chain. Abstains — returns false — when the op head says
    /// none, operands don't resolve, or face-decode quality is poor.
    /// </summary>
    private bool TryGenerateFromGruQuery(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;

        var tokenIds = _tokenizer.Encode(request.Input);
        var (opId, opConfidence, flags) = _model.PredictQuery(tokenIds, AnchorPerception(tokenIds));
        if (opId <= 0 || opId >= GenesisNova.Model.GenesisNeuralModel.QueryOpCount || flags.Length != tokenIds.Length)
            return false;

        // Group the SELECTED tokens into operands, in order: adjacent selected digit tokens merge
        // into one number (the tokenizer's own digit-run convention); selected word tokens resolve
        // through the LEARNED relation carrier (one -> 1). Unselected tokens are the framing the GRU
        // learned to ignore.
        var vocab = _tokenizer.Vocabulary;
        string TokenText(int index)
        {
            var id = tokenIds[index];
            return id >= 0 && id < vocab.Count ? vocab[id] : string.Empty;
        }
        static bool IsDigitText(string s) => s.Length == 1 && s[0] is >= '0' and <= '9';

        var operands = new List<double>();
        var i = 0;
        while (i < flags.Length)
        {
            if (!flags[i]) { i++; continue; }
            var text = TokenText(i);
            // Unary minus: a selected '-' not preceded by a digit, followed by a selected digit, is
            // the operand's sign (mirrors the signed-run convention the label derivation supervises).
            var negative = false;
            if (text == "-"
                && i + 1 < flags.Length && flags[i + 1] && IsDigitText(TokenText(i + 1))
                && (i == 0 || !IsDigitText(TokenText(i - 1))))
            {
                negative = true;
                i++;
                text = TokenText(i);
            }
            if (IsDigitText(text))
            {
                var runValue = 0.0;
                while (i < flags.Length && flags[i] && IsDigitText(TokenText(i)))
                {
                    runValue = (runValue * 10.0) + (TokenText(i)[0] - '0');
                    i++;
                }
                operands.Add(negative ? -runValue : runValue);
                continue;
            }
            if (text.Any(char.IsLetter) &&
                TryResolveTokenToNumber(text, out var digit) &&
                double.TryParse(digit, System.Globalization.NumberStyles.AllowLeadingSign
                        | System.Globalization.NumberStyles.AllowDecimalPoint,
                    System.Globalization.CultureInfo.InvariantCulture, out var wordValue))
            {
                operands.Add(wordValue);
            }
            i++;
        }
        if (operands.Count < 2)
            return false;

        var left = operands[0];
        var right = operands[1];
        var additive = opId is 1 or 2;          // poly face: add/sub
        var sign = opId is 2 or 4 ? -1.0 : 1.0; // negative face for sub/div
        if (!TryFaceArithmetic(left, right, additive, sign, out var faceValue, out var quality))
            return false;
        if (quality <= 0.50)
            return false; // decode self-consistency gate — same bar as the assist sub-step path.

        return EmitPlatonicResult(FormatNumber(faceValue), "platonic-gru-query",
            Math.Min(quality, opConfidence), hops: 1, request, evidence: null, out result);
    }

    /// <summary>
    /// LEARNED-OPERATION route. An operation LEARNED from examples (not a fixed op vocabulary, not parsed
    /// by name) is a first-class element of the space, SELECTED by following a learned RELATION from a cue
    /// concept (so "twice 7" reaches "double" through the learned twice↔double edge), and APPLIED:
    ///  • UNARY (one operand) — a transform vector T(f)=avg(embed(out)−embed(in)) applied BY COMPOSITION
    ///    (predicted = embed(x)+T(f), a Sum-composition of the argument with the function element), decoded
    ///    in the function's own face (PreferredFace: poly +k / log ×k; the other face holds a spurious but
    ///    clean reading, so decoding the wrong one would silently mislead).
    ///  • BINARY (two operands) — a discovered fold / log-linear STRUCTURE (mul = fold of add; c = a^α·b^β),
    ///    evaluated by <see cref="FoldPathDiscovery.TryPredict"/>.
    /// Both generalize to UNSEEN operands from a few examples (measured). Abstains when no store is wired,
    /// the operand count isn't 1 or 2, no cue resolves to a learned op, or the decode is low-quality.
    /// </summary>
    private bool TryGenerateFromLearnedFunction(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if ((_transformAccumulator is null && _foldPathDiscovery is null) || string.IsNullOrWhiteSpace(request.Input))
            return false;

        var tokens = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 2)
            return false;

        const System.Globalization.NumberStyles numericStyle =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;

        var operandTokens = tokens.Where(t => double.TryParse(t, numericStyle, inv, out _)).ToArray();
        if (operandTokens.Length is not (1 or 2)) // unary or binary learned op (binary arithmetic with an
            return false;                          // operator token is handled by the GRU-query route first)
        var cues = tokens.Where(t => t.Any(char.IsLetter)).ToArray();
        if (cues.Length == 0)
            return false;

        var dim = _memory.FaceDimension;

        foreach (var cue in cues)
        {
            // The op element is the cue itself OR a relational neighbour of it (learned edge) — retrieval
            // from the space, not a name lookup. First candidate carrying a learned op wins.
            var candidates = new List<string>(5) { cue };
            candidates.AddRange(_memory
                .GetNeighbors(cue, PlatonicNeighborhoodType.Relational, maxNeighbors: 4, minConfidence: 0.35)
                .Select(n => n.Concept));

            foreach (var fn in candidates)
            {
                // UNARY — learned transform applied by composition.
                if (operandTokens.Length == 1 && _transformAccumulator is not null
                    && _transformAccumulator.TryGetTransform(fn, out var transform))
                {
                    var predicted = _transformAccumulator.Apply(fn, InputEmbeddingComposer.GetInputEmbedding(operandTokens[0], dim));
                    if (predicted is null)
                        continue;
                    var (value, quality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(predicted, dim, transform.PreferredFace);
                    if (face != "none" && quality > 0.50
                        && TryBuildNumericResult(value, quality, $"platonic-learned-function:{fn}", request, out result))
                        return true;
                    continue;
                }

                // BINARY — learned fold / log-linear structure evaluated on the operands.
                if (operandTokens.Length == 2 && _foldPathDiscovery is not null && _foldPathDiscovery.HasOperation(fn)
                    && double.TryParse(operandTokens[0], numericStyle, inv, out var a)
                    && double.TryParse(operandTokens[1], numericStyle, inv, out var b)
                    && _foldPathDiscovery.TryPredict(fn, a, b, out var predValue, out _)
                    && TryBuildNumericResult(predValue, 0.85, $"platonic-learned-op:{fn}", request, out result))
                    return true;
            }
        }
        return false;
    }

    // Encode a numeric value into a platonic-credited GenerationResult; false if it doesn't tokenize.
    private bool TryBuildNumericResult(double value, double confidence, string decisionPath,
        GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var outTokens = _tokenizer.Encode(FormatNumber(value), addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;
        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: decisionPath,
            PlatonicConfidence: confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: 1);
        return true;
    }

    /// <summary>
    /// Shared emit tail for the platonic routes: encode <paramref name="answer"/> (EOS-terminated, capped at
    /// <c>Max(1, request.MaxNewTokens)</c>), abstain (false) on an empty encoding, else build the standard
    /// platonic <see cref="GenerationResult"/>. Each caller keeps its OWN <paramref name="decisionPath"/> /
    /// <paramref name="confidence"/> / <paramref name="hops"/> / <paramref name="evidence"/>; only the
    /// boilerplate (UsedPlatonicQuery:true, UsedNeuralFallback:false, bias 0/0, ChunksGenerated:1) is shared.
    /// Byte-identical to the previously inlined constructions.
    /// </summary>
    private bool EmitPlatonicResult(string answer, string decisionPath, double confidence, int hops,
        GenerationRequest request, IReadOnlyList<PlatonicEvidence>? evidence, out GenerationResult result)
    {
        result = default!;
        var outTokens = _tokenizer.Encode(answer, addEos: true)
            .Take(Math.Max(1, request.MaxNewTokens))
            .ToArray();
        if (outTokens.Length == 0)
            return false;
        result = new GenerationResult(
            Output: _tokenizer.Decode(outTokens),
            GeneratedTokens: outTokens,
            UsedPlatonicQuery: true,
            UsedNeuralFallback: false,
            DecisionPath: decisionPath,
            PlatonicConfidence: confidence,
            AppliedBiasCount: 0,
            AverageBiasMagnitude: 0.0,
            ChunksGenerated: 1,
            PlatonicHopCount: hops,
            Evidence: evidence);
        return true;
    }

    /// <summary>
    /// LEARNED-COMPOSER route. The GRU's plan head (<see cref="GenesisNeuralModel.PredictPlan"/>) selects a
    /// block-composition SHAPE; this assembles the corresponding glider and runs it on the substrate. The
    /// blocks are the vocabulary, the GRU is the composer — no hardcoded per-token resolver. Increment 1
    /// wires PlanKind=Predicate: a Compare→Branch tree over the two numeric operands → greater/less/equal,
    /// computed element-natively (the difference's sign) and generalizing to unseen operands. Other plan
    /// kinds (arithmetic/retrieval) are served by their existing routes. Abstains otherwise.
    /// </summary>
    private bool TryGenerateFromGliderPlan(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;
        var tokenIds = _tokenizer.Encode(request.Input);
        var (planKind, conf) = _model.PredictPlan(tokenIds, AnchorPerception(tokenIds));
        // Execute the shapes the dedicated routes DON'T serve: predicate (2), arithmetic→word (4),
        // fold-sum (5), fold-product (6), seq (7). Digit-arithmetic (1) defers to GruQuery, retrieval (3)
        // to relation-first, and the MULTI-operator expression-chain (8) to its own route below — keeping
        // their decision paths intact. Each shape's WORK is on the substrate.
        if (planKind is not (2 or 4 or 5 or 6 or 7))
            return false;

        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var operands = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Where(t => double.TryParse(t, ns, inv, out _)).ToArray();
        if (operands.Length < 2)
            return false;

        // The interpreter carries the shape registry's library, so Ref blocks resolve recursively against
        // the shapes-as-Function-elements — no per-call inline glider construction.
        var interp = _glider;
        GliderBlock root;
        switch (planKind)
        {
            case 2: // Predicate: Compare→Branch over the two operands (element-native difference-sign).
                root = new Branch(
                    new Compare(CompareOp.Greater, new Operand(0), new Operand(1)),
                    new Literal("greater"),
                    new Branch(new Compare(CompareOp.Less, new Operand(0), new Operand(1)),
                        new Literal("less"), new Literal("equal")));
                break;
            case 5: // Fold-sum: variadic reduce of + over ALL operands (one N-way R2 compose).
                root = new Fold(GliderOp.Add, 0);
                break;
            case 6: // Fold-product: variadic reduce of × over ALL operands.
                root = new Fold(GliderOp.Multiply, 0);
                break;
            case 7: // SEQ — Concatenate-Composition: a scaffold chunk RETRIEVED from the chunk-element store
                // (mined from graded-correct outputs — NOT a literal baked in here) bound to a substrate-
                // computed value. The COMPUTE part is substrate-native (Fold(Add) → one R2 compose over all
                // operands / the homomorphism); the assembly is the interpreter's Seq block (concatenation =
                // CompositionMode.Concatenate). Abstain until a scaffold has been learned.
                if (!_memory.TryGetTopChunk(PlatonicSpaceMemory.SeqScaffoldTag, out var scaffold))
                    return false;
                root = new Seq(new GliderBlock[]
                {
                    new Literal(scaffold),
                    new Fold(GliderOp.Add, 0),
                });
                break;
            default: // planKind == 4: arithmetic FORMATTED AS A WORD — Hop(Compute(op,operands), Word).
                var (opId, _, _) = _model.PredictQuery(tokenIds, AnchorPerception(tokenIds));
                var gop = opId switch
                {
                    1 => GliderOp.Add,
                    2 => GliderOp.Subtract,
                    3 => GliderOp.Multiply,
                    4 => GliderOp.Divide,
                    _ => (GliderOp?)null
                };
                if (gop is null)
                    return false;
                root = new Hop(new Compute(gop.Value, new GliderBlock[] { new Operand(0), new Operand(1) }), HopTarget.Word);
                break;
        }

        string answer;
        try
        {
            answer = interp.Execute(new PlatonicGlider("plan", root), operands);
        }
        catch (InvalidOperationException)
        {
            return false; // unresolvable on the substrate (e.g. no learned digit→word edge) — fall through
        }
        if (string.IsNullOrEmpty(answer))
            return false;

        var decisionPath = "platonic-glider-plan:" + planKind switch
        {
            2 => "predicate",
            4 => "arith-word",
            5 => "fold-sum",
            6 => "fold-product",
            7 => "seq",
            _ => "shape" + planKind
        };
        return EmitPlatonicResult(answer, decisionPath, conf, hops: 1, request, evidence: null, out result);
    }

    /// <summary>TARGET-AGNOSTIC perception of the input's anchor region (SPACE_AWARE_GRU.md §A), shared by the
    /// space-aware plan/op heads — the same construction the route head uses. Null when perception is off or
    /// there is no anchor. Numeric/arithmetic inputs have no relational anchor → ≈0 vector → graceful no-op.</summary>
    private double[]? AnchorPerception(IReadOnlyList<int> tokenIds)
    {
        if ((!_model.PerceptionPlan && !_model.PerceptionQuery) || tokenIds.Count == 0)
            return null;
        var toks = _tokenizer.Decode(tokenIds).Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (toks.Length == 0)
            return null;
        var transformReliability = _model.TransformReliabilityRouting && _transformAccumulator is not null
            ? _transformAccumulator.BestReliabilityUcb()
            : 0.0;
        // KEEP-CORE: perceive the discriminative anchor (matching the trainer's reinforcement), not the last token.
        var anchorTok = KeepCoreControl ? (DiscriminativeAnchorToken(tokenIds) ?? toks[^1]) : toks[^1];
        return _memory.ComputeRoutePerception(anchorTok, transformReliability);
    }

    /// <summary>
    /// EXPRESSION-CHAIN route (plan-kind 8). The GRU plan head selects "this is a multi-operator expression";
    /// this evaluates it by CHAINING compute-elements on the substrate. Each operator is classified from
    /// CONTEXT by the learned op head (the op head runs on the operator's local binary window) — there is NO
    /// hardcoded symbol→op map, so "x" is multiply only when the surrounding tokens make it so. Evaluation
    /// uses standard precedence (× ÷ before + −, each pass left-to-right); every binary step is one substrate
    /// R2 compose + homomorphic decode (PlatonicGliderInterpreter), so the answer GENERALISES to any operands.
    /// Control flow lives here; the compute is on the substrate. Abstains unless the plan head picked kind 8
    /// AND the input parses as a ≥2-operator expression AND the op head classifies every operator.
    /// </summary>
    private bool TryGenerateFromExpressionChain(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (string.IsNullOrWhiteSpace(request.Input))
            return false;
        var tokenIds = _tokenizer.Encode(request.Input);
        var (planKind, conf) = _model.PredictPlan(tokenIds, AnchorPerception(tokenIds));
        if (planKind != 8)
            return false; // the GRU plan head must SELECT the expression-chain shape

        const System.Globalization.NumberStyles ns =
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var inv = System.Globalization.CultureInfo.InvariantCulture;

        // Parse the (space-separated) expression into alternating operand values and operator tokens.
        var raw = request.Input.Trim().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var operands = new List<double>();
        var opTokens = new List<string>();
        var expectOperand = true;
        foreach (var tok in raw)
        {
            if (expectOperand)
            {
                if (!double.TryParse(tok, ns, inv, out var v))
                    return false; // framing/garbage where an operand was expected → abstain
                operands.Add(v);
                expectOperand = false;
            }
            else { opTokens.Add(tok); expectOperand = true; }
        }
        if (!expectOperand || opTokens.Count < 2 || operands.Count != opTokens.Count + 1)
            return false; // MULTI-operator only; single binary defers to GruQuery

        // Classify EACH operator from CONTEXT via the learned op head — its own local binary window. No
        // symbol→op lookup: the head decides add/sub/mul/div from the operand+operator+operand context.
        var ops = new List<GliderOp>(opTokens.Count);
        for (var i = 0; i < opTokens.Count; i++)
        {
            var window = _tokenizer.Encode($"{FormatNumber(operands[i])} {opTokens[i]} {FormatNumber(operands[i + 1])}");
            var (opId, _, _) = _model.PredictQuery(window);
            GliderOp? g = opId switch
            {
                1 => GliderOp.Add,
                2 => GliderOp.Subtract,
                3 => GliderOp.Multiply,
                4 => GliderOp.Divide,
                _ => (GliderOp?)null
            };
            if (g is null)
                return false; // op head couldn't classify this operator from context → abstain
            ops.Add(g.Value);
        }

        // One binary step = one substrate R2 compose + homomorphic decode (a chained compute-element).
        double Step(double l, GliderOp op, double r)
        {
            var glider = new PlatonicGlider("expr-step",
                new Compute(op, new GliderBlock[] { new Const(l), new Const(r) }));
            var txt = _glider.Execute(glider, Array.Empty<string>());
            return double.TryParse(txt, ns, inv, out var res) ? res : double.NaN;
        }

        // Precedence: resolve × ÷ left-to-right, then + − left-to-right — chaining the compute-elements.
        var vals = operands.ToList();
        var oo = ops.ToList();
        for (var i = 0; i < oo.Count;)
        {
            if (oo[i] is GliderOp.Multiply or GliderOp.Divide)
            {
                var r = Step(vals[i], oo[i], vals[i + 1]);
                if (double.IsNaN(r)) return false;
                vals[i] = r;
                vals.RemoveAt(i + 1);
                oo.RemoveAt(i);
            }
            else i++;
        }
        var acc = vals[0];
        for (var i = 0; i < oo.Count; i++)
        {
            acc = Step(acc, oo[i], vals[i + 1]);
            if (double.IsNaN(acc)) return false;
        }

        // chain length = number of chained compute-elements
        return EmitPlatonicResult(FormatNumber(acc), "platonic-expression-chain", conf,
            hops: opTokens.Count, request, evidence: null, out result);
    }

    // The platonic reasoning ladder (Open/Closed): try each route in priority order; the first that does NOT abstain
    // wins. Edge-following routes are dropped when proximity-kNN-only is selected. Order/membership = `_platonicRoutes`
    // (built in the constructor) — never edit this loop to add a route. Route bodies follow (glider-plan first, so
    // framed/compact arithmetic never falls into a relational text walk; abstaining routes leave every test unaffected).
    private bool TryGenerateFromPlatonicPlan(GenerationRequest request, out GenerationResult result)
    {
        foreach (var route in _platonicRoutes)
        {
            if (route.EdgeFollowing && !EdgeRoutingEnabled)
                continue;
            if (route.TryGenerate(request, out result))
                return true;
        }
        result = default!;
        return false;
    }

    // CONCEPT-CHAIN retrieval (edge-following): a multi-hop relational beam walk. The chain is a SEARCH WALK, not the
    // surface answer — its text concatenates every beam candidate across every hop ("fruit orange banana grape" for
    // 'apple'), which was the root of the answer-wrapped-in-noise outputs. The platonic-direct ANSWER is the TOP
    // retrieval (highest-scoring hop-1 candidate); the rest of the walk is retained as evidence, not emitted as text.
    private bool TryGenerateFromConceptChain(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count == 0)
            return false;

        var maxHops = ResolveMaxPlatonicHops(anchors.Count);
        var beamWidth = ResolvePlatonicBeamWidth(anchors.Count);
        var conceptResult = _memory.QueryConceptChain(
            anchors,
            maxHops: maxHops,
            beamWidth: beamWidth,
            evidence: out var evidence);
        if (string.IsNullOrWhiteSpace(conceptResult.Text))
            return false;

        var topConcept = conceptResult.Text.Split(
            (char[]?)null, StringSplitOptions.RemoveEmptyEntries)[0];
        return EmitPlatonicResult(topConcept, "platonic-query-concept-chain", conceptResult.Confidence,
            hops: conceptResult.Hops, request, evidence, out result);
    }

    /// <summary>
    /// REASON route (PLATONIC_RECKONING.md keep-core) — retrieval/disambiguation by RELAXATION, the keeper
    /// mechanism from the conscious-field work. The anchor tokens form a query cloud that relaxes over the stored
    /// concept clouds (modern-Hopfield/attention); the settled basin IS the answer. Unlike geometric retrieval it
    /// accepts MULTIPLE anchors (summed into one query) and abstains HONESTLY: when the raw query has no near basin
    /// it reports Settled=false and this route falls through (no invention). Inert unless KeepCoreControl, and only
    /// over the DialecticalSpace (Reason is its method, not on the IPlatonicSpace contract). Placed ABOVE geometric
    /// retrieval so relaxation is the primary retrieval path; geometric/relation/chain remain as fallbacks.
    /// </summary>
    private bool TryGenerateFromReason(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        if (!KeepCoreControl)
            return false;
        if (_memory is not GenesisNova.Cognition.Platonic.DialecticalSpace ds)
            return false;
        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count == 0)
            return false;
        var thought = ds.Reason(anchors);
        if (!thought.Settled || thought.Confidence < ReasonMinConfidence || string.IsNullOrEmpty(thought.Symbol))
            return false; // nothing settled / surprised query → defer to geometric/relation/chain
        if (PlatonicSpaceMemory.IsReservedConcept(thought.Symbol) || _memory.IsOperationToken(thought.Symbol))
            return false;
        return EmitPlatonicResult(thought.Symbol, "platonic-reason", thought.Confidence,
            hops: Math.Max(1, thought.Steps), request, evidence: null, out result);
    }

    /// <summary>
    /// GEOMETRIC retrieval — the original dimensional-space design: a concept's POSITION in the semantic face
    /// IS its identity, and the nearest stored concept (lattice VP-Tree, <see cref="PlatonicSpaceMemory.GetNearestConcepts"/>)
    /// is the answer. Single-concept queries only. Skips reserved/op-token neighbours, and ABSTAINS below
    /// <see cref="GeometricMinConfidence"/> so it only wins when the geometry is trustworthy (the relation
    /// edge is the fallback). This is the re-promotion of geometry over symbol-keyed relations.
    /// </summary>
    private bool TryGenerateFromGeometricRetrieval(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count != 1)
            return false;

        foreach (var (sym, dist) in _memory.GetNearestConcepts(anchors[0], candidates: null, maxNeighbors: 5))
        {
            if (PlatonicSpaceMemory.IsReservedConcept(sym) || _memory.IsOperationToken(sym))
                continue;
            var confidence = 1.0 / (1.0 + Math.Max(0.0, dist));
            if (confidence < GeometricMinConfidence)
                return false; // geometry not confident here → defer to the relation edge
            return EmitPlatonicResult(sym, "platonic-geometric", confidence, hops: 1, request,
                evidence: null, out result);
        }
        return false;
    }

    private bool TryGenerateFromRelationEdge(GenerationRequest request, out GenerationResult result)
    {
        result = default!;
        var anchors = ExtractConceptAnchors(request.Input);
        if (anchors.Count != 1)
            return false;

        var neighbours = _memory.GetNeighbors(
            anchors[0],
            PlatonicNeighborhoodType.Relational,
            maxNeighbors: 5,
            minConfidence: RelationFirstMinConfidence);

        // Pick the strongest neighbour that is a real, user-facing concept — never a RESERVED internal marker
        // (face:poly/face:log) or an op-token. Those are routing affinity, not answers.
        string? answer = null;
        var confidence = 0.0;
        foreach (var n in neighbours)
        {
            if (PlatonicSpaceMemory.IsReservedConcept(n.Concept) || _memory.IsOperationToken(n.Concept))
                continue;
            answer = n.Concept;
            confidence = n.Confidence;
            break;
        }
        if (answer is null)
            return false;

        return EmitPlatonicResult(answer, "platonic-relation-edge", confidence, hops: 1, request,
            evidence: null, out result);
    }

    // Resolves a single token to a digit string using ONLY the learned relation graph — no hardcoded
    // word→number table. A token that already strict-parses as a number is returned unchanged. Otherwise
    // the strongest NUMERIC relational neighbour (highest ObservationCount, tiebreak higher Confidence)
    // is used. Returns false when no numeric relation exists, so unlearned words stay verbatim.
    private bool TryResolveTokenToNumber(string token, out string digit)
    {
        digit = token;
        if (string.IsNullOrEmpty(token))
            return false;

        // Strict numeric parse: deliberately NOT NumberStyles.Any (which would accept noise like "0+").
        const System.Globalization.NumberStyles strictStyles =
            System.Globalization.NumberStyles.AllowLeadingSign |
            System.Globalization.NumberStyles.AllowDecimalPoint |
            System.Globalization.NumberStyles.AllowLeadingWhite |
            System.Globalization.NumberStyles.AllowTrailingWhite;

        if (double.TryParse(token, strictStyles, System.Globalization.CultureInfo.InvariantCulture, out _))
            return true; // already a number — leave unchanged

        string? bestConcept = null;
        var bestObs = int.MinValue;
        var bestConf = double.MinValue;
        foreach (var neighbor in _memory.GetNeighbors(
                     token,
                     GenesisNova.Cognition.PlatonicNeighborhoodType.Relational,
                     maxNeighbors: 16,
                     minConfidence: 0.4))
        {
            if (!double.TryParse(neighbor.Concept, strictStyles, System.Globalization.CultureInfo.InvariantCulture, out _))
                continue; // only numeric relational neighbours qualify
            if (neighbor.ObservationCount > bestObs ||
                (neighbor.ObservationCount == bestObs && neighbor.Confidence > bestConf))
            {
                bestObs = neighbor.ObservationCount;
                bestConf = neighbor.Confidence;
                bestConcept = neighbor.Concept;
            }
        }

        if (bestConcept is null)
            return false;

        digit = bestConcept;
        return true;
    }

    /// <summary>
    /// Performs arithmetic in face embedding space by BLENDING the operand faces (preserving the
    /// homomorphism poly(a)±poly(b)=poly(a±b), log(a)±log(b)=log(a*b or a/b)) and then decoding the
    /// resulting predicted embedding through the source-of-truth
    /// <see cref="PlatonicFaceDecoder.DecodeNumericFromPrediction"/> — the exact algebraic inverse of
    /// the composer. The decode formula AND the self-consistency quality come from the decoder, so
    /// they match the canonical genesis-engine codec rather than an ad-hoc CV metric.
    /// <para>
    /// preferFace hint: 1=poly for add/sub, 2=log for mul/div. The log face decodes positive, so the
    /// operand sign is reapplied here (sign lives in the poly face / operands, not the log magnitude).
    /// </para>
    /// </summary>
    private bool TryFaceArithmetic(double left, double right, bool additiveFace, double rightFaceSign, out double result, out double quality)
    {
        result = 0;
        quality = 0;

        var leftKey = FormatNumber(left);
        var rightKey = FormatNumber(right);

        if (!_memory.TryGetConceptFace(leftKey, out var faceA) ||
            !_memory.TryGetConceptFace(rightKey, out var faceB))
            return false;

        var dim = _memory.FaceDimension;
        if (faceA.Length < dim || faceB.Length < dim)
            return false;

        if (!additiveFace)
        {
            // Zero is outside log-space; preserve exact arithmetic behavior for zero cases
            // (mul by zero = 0; div of zero by nonzero = 0). Both faces would otherwise be NaN/Inf.
            if (Math.Abs(left) < 1e-12 || (rightFaceSign > 0.0 && Math.Abs(right) < 1e-12))
            {
                result = 0;
                quality = 1.0;
                return true;
            }
        }

        // Blend the operand faces element-wise. The poly block [0..numericDims) carries the additive
        // homomorphism and the log block [logStart..) carries the multiplicative one, so a single
        // signed sum yields the predicted embedding for the result on BOTH faces simultaneously.
        var blended = new double[dim];
        for (var i = 0; i < dim; i++)
            blended[i] = faceA[i] + rightFaceSign * faceB[i];

        // Decode via the canonical codec, hinting the face the operation actually evaluates on.
        var preferFace = additiveFace ? 1 : 2; // 1=poly (add/sub), 2=log (mul/div)
        var (value, decodeQuality, face) = PlatonicFaceDecoder.DecodeNumericFromPrediction(blended, dim, preferFace);
        if (face == "none")
            return false;

        // The log face always decodes a positive magnitude; reapply the operand sign for mul/div.
        if (!additiveFace && face == "log")
            value *= Math.Sign(left) * Math.Sign(right);

        result = value;
        quality = decodeQuality;
        return !double.IsNaN(result) && !double.IsInfinity(result) && !double.IsNaN(quality) && !double.IsInfinity(quality);
    }

    private static int ResolveMaxPlatonicHops(int anchorCount)
        => Math.Clamp(1 + anchorCount / 2, 2, 4);

    private static int ResolvePlatonicBeamWidth(int anchorCount)
        => Math.Clamp(1 + anchorCount / 3, 2, 4);

    private static string FormatNumber(double value)
    {
        if (Math.Abs(value - Math.Round(value)) <= 1e-9)
            return ((long)Math.Round(value)).ToString(System.Globalization.CultureInfo.InvariantCulture);

        return value.ToString("0.##########", System.Globalization.CultureInfo.InvariantCulture);
    }
}
