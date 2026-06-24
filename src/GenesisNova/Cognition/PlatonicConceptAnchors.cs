using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Cognition;

/// <summary>
/// THE one discriminative concept-anchor rule, shared by inference retrieval AND trainer supervision so both
/// query the SAME cue (the reckoning's seam fix — `PLATONIC_RECKONING.md` §8). Before this, inference dropped
/// framing-word hubs and anchored on the content cue ("a synonym for big" → "big") while the trainer's
/// route-label + perception anchored on the first surface token ("a") — so a healthy retrieval geometry was
/// invisible to the controller it was supposed to teach. Logic is lifted verbatim from the inference engine's
/// former private methods; it depends on nothing but <see cref="IPlatonicSpace"/>, so it lives here next to the
/// contract and both <c>Infer</c> and <c>Train</c> forward to it.
/// </summary>
public static class PlatonicConceptAnchors
{
    private static readonly char[] Trim = { '?', '!', '.', ',', ';', ':', '(', ')', '[', ']', '"', '\'' };

    /// <summary>
    /// Extract retrieval/supervision anchors from a raw input string: drop op-tokens and stray single letters,
    /// keep only known concepts, then reduce to the most SPECIFIC (lowest-degree) cue(s). Identical to the
    /// inference engine's prior <c>ExtractConceptAnchors</c>.
    /// </summary>
    public static IReadOnlyList<string> Extract(IPlatonicSpace space, string input)
    {
        if (space is null) throw new ArgumentNullException(nameof(space));
        if (string.IsNullOrWhiteSpace(input))
            return Array.Empty<string>();

        var candidates = input
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim(Trim))
            .Select(t => t.ToLowerInvariant())
            // Keep single-char DIGIT tokens ("3" is a real concept with a learned relation to its word); drop
            // stray single letters. ContainsConcept gates below.
            .Where(t => t.Length > 1 || (t.Length == 1 && char.IsDigit(t[0])))
            // An op-token (declared ROUTE-TRIGGER) is never a retrieval anchor; the degree filter keeps framing
            // words out data-drivenly.
            .Where(t => !space.IsOperationToken(t))
            .Where(t => space.ContainsConcept(t))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        return SelectDiscriminative(space, candidates);
    }

    /// <summary>
    /// Like <see cref="Extract"/>, but ABSTAINS (returns empty) when no surviving anchor is a SPECIFIC cue — i.e.
    /// every survivor is a hub, its relation degree at or above the space's average connectivity (2·rels/nodes).
    /// This is the keep-core abstention made cue-aware (PLATONIC_RECKONING.md): when an unknown content word is
    /// filtered out and only framing-word hubs remain, relaxing over them settles on a POPULARITY answer (the
    /// decode-collapse failure mode). Retrieving only on a cue more specific than typical is scale-relative — no
    /// fixed threshold — so it holds as the space grows. Used by inference under KeepCoreControl; training is
    /// unaffected (a training example always carries its known content cue).
    /// </summary>
    public static IReadOnlyList<string> ExtractSpecific(IPlatonicSpace space, string input)
    {
        var anchors = Extract(space, input);
        if (anchors.Count == 0)
            return anchors;
        // Honest abstention about an UNKNOWN referent. The failure to avoid: a query names a content word the space
        // does not know ("a synonym for zzqqxx"); it gets filtered out, leaving only the framing-word HUBS, and
        // relaxing over those settles on a popularity answer. So: if the query references an unknown content word AND
        // every surviving anchor is itself a hub (degree at/above the space's average connectivity), abstain. The
        // unknown-word gate means a known-cue query is NEVER affected; the degree check means a real cue alongside an
        // incidental unknown word still retrieves. Scale-relative — no fixed threshold.
        var nodes = space.NodeCount;
        if (nodes <= 0)
            return anchors;
        var avgDegree = 2.0 * space.RelationCount / nodes;
        if (avgDegree < 2.0)
            return anchors; // space too sparse for the average to mean anything yet
        var referencesUnknownContent = input
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(t => t.Trim(Trim).ToLowerInvariant())
            .Any(t => t.Length > 1 && t.All(char.IsLetter) && !space.IsOperationToken(t) && !space.ContainsConcept(t));
        if (referencesUnknownContent && anchors.All(a => space.GetRelationDegree(a) >= avgDegree))
            return Array.Empty<string>(); // unknown subject + only hubs survived → no content cue → abstain
        return anchors;
    }

    /// <summary>
    /// DISCRIMINATIVE filter: the cue is the most SPECIFIC token = LOWEST relation degree; a framing word
    /// ("what"/"thing"/"a") sits near everything, accrues high degree, and is dropped — so a hub can't collapse
    /// the result to a constant. Only the low-degree content cue(s) survive; ties/near-ties are kept so genuine
    /// multi-concept queries still work. Identical to the inference engine's prior <c>SelectDiscriminativeConcepts</c>.
    /// </summary>
    public static IReadOnlyList<string> SelectDiscriminative(IPlatonicSpace space, IReadOnlyList<string> candidates)
    {
        if (space is null) throw new ArgumentNullException(nameof(space));
        if (candidates is null || candidates.Count <= 1)
            return candidates ?? Array.Empty<string>();
        var byDegree = candidates
            .Select(t => (Token: t, Degree: space.GetRelationDegree(t)))
            .OrderBy(x => x.Degree)
            .ToList();
        var minDegree = byDegree[0].Degree;
        return byDegree
            .Where(x => x.Degree <= (minDegree * 2) + 2) // keep the cue + comparably-specific tokens; drop hubs
            .Select(x => x.Token)
            .Take(4)
            .ToArray();
    }
}
