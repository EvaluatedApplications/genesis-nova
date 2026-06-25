using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// Learns the STRUCTURAL ROLES of tokens (the grammar) from the ASSERT/RECALL alignment alone — NO hardcoded
/// copula/question/possessive word-lists, and NO word-order assumption. The signal is self-supervised from the
/// input→output structure of examples (the same well the op-cue learner draws from):
///
///  • An ASSERTION ("my name is sam" → "sam") has its answer token PRESENT in the input.
///  • A RECALL    ("what is my name" → "sam") has its answer ABSENT from the input (the answer is what's asked for).
///
/// From many such examples, per token:
///  • a FILLER token (recognised by the space's learned centrality signal, passed in) that shows up ONLY in
///    answer-ABSENT inputs is a QUERY cue ("what"); one in both is generic filler ("is"/"my").
///  • a CONTENT token that is ever an ANSWER is a VALUE ("sam"); one only ever queried is a KEY/subject ("name").
///
/// This is set membership + co-occurrence, NOT position, so it holds for any word order (SVO/SOV/VSO). A foreign or
/// NONCE structural word is classified by how it behaves, never by being in a list (proven: a nonce copula reads in
/// the filler band like "is"). See the nova-learned-grammar-roles note. Plain, persistable counters — the learned
/// grammar is just these tallies over what the mind has been told.
/// </summary>
public sealed class GrammarRoleLearner
{
    public enum Role { Unknown, Filler, Query, Key, Value }

    private sealed class Tally { public int InAnswerPresent; public int InAnswerAbsent; public int AsAnswer; public int AsCopula; }
    private readonly Dictionary<string, Tally> _stats = new(StringComparer.OrdinalIgnoreCase);

    private static string N(string t) => t.Trim().ToLowerInvariant();
    private Tally Get(string t) { var k = N(t); if (!_stats.TryGetValue(k, out var v)) _stats[k] = v = new Tally(); return v; }

    /// <summary>Observe one example's structure: which tokens were in the input, and the single answer token. The
    /// answer-PRESENT (assertion) vs ABSENT (recall) distinction is the whole self-supervised signal.</summary>
    public void Observe(IReadOnlyList<string> inputTokens, string output)
    {
        if (inputTokens is null || inputTokens.Count == 0 || string.IsNullOrWhiteSpace(output)) return;
        var outNorm = N(output);
        var raw = inputTokens.Select(N).Where(t => t.Length > 0).ToList(); // keep ORDER (for the copula position)
        var inNorm = raw.Distinct().ToList();
        var answerPresent = inNorm.Contains(outNorm); // assertion (answer is stated) vs recall (answer is asked for)
        foreach (var t in inNorm)
        {
            var s = Get(t);
            if (answerPresent) s.InAnswerPresent++; else s.InAnswerAbsent++;
        }
        if (!string.IsNullOrWhiteSpace(outNorm)) Get(outNorm).AsAnswer++;
        // COPULA position: in an ASSERTION the token IMMEDIATELY BEFORE the value is the copula. This is what tells the
        // copula ("is") apart from the subject noun ("name") — both appear in assert AND "what is …" recall frames, so
        // the present/absent counters alone label both SUBJECT; the copula's POSITION (adjacent to the value) does not.
        if (answerPresent)
        {
            var valuePos = raw.LastIndexOf(outNorm);
            if (valuePos > 0) Get(raw[valuePos - 1]).AsCopula++;
        }
    }

    /// <summary>True once a token has been seen enough to classify it (cold tokens stay Unknown — honest abstention,
    /// the same warm-start stance as the learned filler signal).</summary>
    public bool Knows(string token) => _stats.TryGetValue(N(token), out var s) && (s.InAnswerPresent + s.InAnswerAbsent + s.AsAnswer) >= MinObservations;
    private const int MinObservations = 3;

    /// <summary>Classify a token's structural role from the accumulated alignment stats plus the space's learned
    /// FILLER signal (<paramref name="isFiller"/> = DialecticalSpace.IsFunctionLike) — no word-order, no word-lists.</summary>
    public Role Classify(string token, Func<string, bool> isFiller)
    {
        if (!_stats.TryGetValue(N(token), out var s)) return Role.Unknown;
        if (s.InAnswerPresent + s.InAnswerAbsent + s.AsAnswer < MinObservations) return Role.Unknown;

        if (isFiller(token))
        {
            // A query cue lives in answer-ABSENT frames (the value is what's being asked) and stays out of assertions.
            var absentRate = (s.InAnswerPresent + s.InAnswerAbsent) > 0
                ? s.InAnswerAbsent / (double)(s.InAnswerPresent + s.InAnswerAbsent) : 0.0;
            return absentRate >= QueryAbsentRate ? Role.Query : Role.Filler;
        }
        // Content: ever an ANSWER → it's a VALUE; only ever the queried thing → it's a KEY/subject.
        return s.AsAnswer > 0 && s.AsAnswer >= s.InAnswerPresent ? Role.Value : Role.Key;
    }

    private const double QueryAbsentRate = 0.80; // appears overwhelmingly in answer-absent (recall) inputs

    /// <summary>SELF-SUPERVISED TRAINING LABEL for the NN role head — derived PURELY from the alignment counters, with
    /// NO centrality/filler signal (so it's robust where the geometric classifier is fragile; the NN then learns to
    /// recognise and generalise). 2=VALUE (ever an answer); 3=QUERY (only ever in recall inputs, never asserted/answer);
    /// 0=NONE (present-only, never an answer — a copula); 1=SUBJECT (in BOTH assert and recall inputs — the subject,
    /// determiner lumped in, which is correct for the phrase); -1=ignore (too few observations).</summary>
    public int LabelFor(string token)
    {
        if (!_stats.TryGetValue(N(token), out var s)) return -1;
        if (s.InAnswerPresent + s.InAnswerAbsent + s.AsAnswer < MinObservations) return -1;
        // COPULA first: a token that is PREDOMINANTLY the value-adjacent slot in assertions is a copula -> NONE, even if
        // it also shows up in "what is …" recall frames (which would otherwise mislabel it SUBJECT via the both-counter).
        if (s.AsCopula > 0 && s.AsCopula >= 0.5 * s.InAnswerPresent) return 0; // NONE — copula
        if (s.AsAnswer > 0) return 2;                                       // VALUE — has been an answer
        if (s.InAnswerPresent == 0 && s.InAnswerAbsent > 0) return 3;       // QUERY — only ever in recall inputs
        if (s.InAnswerAbsent == 0) return 0;                               // NONE — present-only (copula)
        return 1;                                                          // SUBJECT — in both present and absent inputs
    }

    /// <summary>Snapshot for persistence (the learned grammar is just these tallies).</summary>
    public IReadOnlyList<(string Token, int Present, int Absent, int AsAnswer)> Export()
        => _stats.Select(kv => (kv.Key, kv.Value.InAnswerPresent, kv.Value.InAnswerAbsent, kv.Value.AsAnswer)).ToList();

    public void Import(IEnumerable<(string Token, int Present, int Absent, int AsAnswer)> rows)
    {
        foreach (var (token, present, absent, asAnswer) in rows)
            _stats[N(token)] = new Tally { InAnswerPresent = present, InAnswerAbsent = absent, AsAnswer = asAnswer };
    }
}
