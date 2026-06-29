using System;
using GenesisNova.Cognition.Platonic;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE QUERY-CONDITIONED EGOCENTRIC OBSERVATION  (PLATONIC_NAVIGATOR.md §2/§3, PLATONIC_MIND.md §3 "an answer is
//  wherever a query relaxes to").
//
//  The goal-conditioned navigator (NavFeatures) hands the policy the ANSWER coordinate (goalFace) and asks it to close
//  the gap to it. That cannot be inference: at inference time the answer is exactly what we do NOT have. THE CHANGE is
//  to condition on a QUERY-CONTEXT the walker possesses WITHOUT knowing the answer:
//        query-context = (anchorFace, cue)
//  where `anchor` is the concept being asked about (KNOWN — it is the start of the walk) and `cue` is a small learned
//  embedding over a TARGET-ASPECT set {GENUS, DOMAIN, ROOT} ("what immediate kind / what domain / what is it
//  ultimately"). The SAME anchor with different cues must relax to DIFFERENT answers — none of them supplied.
//
//  So the per-candidate differential references the ANCHOR (fixed, answer-free) instead of the goal:
//        per candidate c →  concat[ candFace − anchorFace , candFace − curFace , candFace , edgeκ ]   (3·dim + 1)
//  Identical width/layout to NavFeatures (the first block's reference point changes from the answer to the anchor), so
//  the recurrent trunk, candidate scoring and the lattice landing are all reused verbatim. The cue is NOT a feature row
//  here — it is mixed into the GRU self/context and the halt/value heads by the net (NavQueryPolicyNet), so "resolved
//  for THIS cue" is what the halt head learns.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>The target-aspect cue — the learned "question-tension" that selects which abstraction level the walk
/// relaxes to. GENUS = the immediate kind (one hop up); DOMAIN = the domain ancestor; ROOT = the ultimate ancestor.</summary>
public enum NavCue
{
    Genus = 0,
    Domain = 1,
    Root = 2,
}

/// <summary>
/// Builds the query-conditioned egocentric observation at a node. Delegates to <see cref="NavFeatures.Build"/> with the
/// ANCHOR face substituted for the (absent) goal face, so training and inference see the EXACT same candidate
/// enumeration and feature layout as the goal-conditioned path — only the reference point of the first differential
/// block changes from the answer to the answer-free anchor.
/// </summary>
public static class NavQueryFeatures
{
    /// <summary>Total cues in the target-aspect set (sizes the learned cue embedding table).</summary>
    public const int CueCount = 3;

    /// <summary>Per-candidate feature length: 3·dim (three differential blocks) + 1 (edge κ) — same as NavFeatures.</summary>
    public static int FeatureLength(int dim) => NavFeatures.FeatureLength(dim);

    /// <summary>
    /// Enumerate the K relational candidates of <paramref name="cur"/> and build their differential rows against the
    /// ANSWER-FREE <paramref name="anchorFace"/> and <paramref name="curFace"/> — i.e. [cand−anchor, cand−cur, cand, κ].
    /// No reference to the answer coordinate anywhere.
    /// </summary>
    public static NavObservation Build(DialecticalSpace space, string cur, double[] curFace, double[] anchorFace, int k, double minConfidence)
    {
        ArgumentNullException.ThrowIfNull(anchorFace);
        // The math is identical to the goal-conditioned builder with anchorFace in place of goalFace; reuse it so the
        // two paths share one candidate enumeration and one feature layout (no drift between train and inference).
        return NavFeatures.Build(space, cur, curFace, anchorFace, k, minConfidence);
    }
}
