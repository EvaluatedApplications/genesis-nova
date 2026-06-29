using System;
using System.Collections.Generic;
using GenesisNova.Cognition.Platonic;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  THE EGOCENTRIC, DIFFERENTIAL OBSERVATION  (PLATONIC_NAVIGATOR.md §2 first-person observation, §6 thin recogniser).
//
//  Meaning is DIFFERENTIAL (a concept is its contrasts κ to others); position emerges from contradiction-minimisation;
//  relations are observable (G5). So the policy must NOT see absolute coordinates and memorise a per-node lookup (the
//  prior BC net's 10% held-out failure). It must read the LOCAL RELATIONAL STRUCTURE: for each candidate neighbour, the
//  contrast of that candidate to WHERE IT WANTS TO BE (goal) and to WHERE IT STANDS (current). That rule —
//  "step toward the candidate whose differential closes the gap to the goal" — is UNIVERSAL across graphs, so it
//  generalises to chains the net never trained on.
//
//  Per candidate c the feature is  concat[ candFace − goalFace , candFace − curFace , candFace , edgeStrength κ ]
//  of length 3·dim + 1. The candidate set is the node's RELATIONAL neighbourhood (GetNeighbors Relational) — the same
//  graph the flow-field oracle flows over, so the oracle's expert next move is always among the candidates.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>The egocentric observation at one node: the K candidate neighbours, their differential feature rows, a
/// validity mask (fewer than K neighbours → trailing rows are zero/masked), and each candidate's symbol + face (so the
/// chosen candidate's FACE can be emitted as the target the lattice lands, §5.1).</summary>
public readonly record struct NavObservation(
    IReadOnlyList<string> CandidateSymbols, // length = valid count (≤ K)
    IReadOnlyList<double[]> CandidateFaces, // parallel to CandidateSymbols — the face to emit when chosen
    float[] FeaturesFlat,                   // length K·F (row-major, K rows of F = 3·dim+1); padded rows are zero
    float[] Mask,                           // length K (1 = valid candidate, 0 = padding)
    int ValidCount);

/// <summary>
/// Builds the differential egocentric observation at a node (PLATONIC_NAVIGATOR.md §2/§6). Shared by the trainer
/// (label = which candidate is the oracle's Next) and the live policy (argmax candidate → emit its face), so training
/// and inference see the EXACT same candidate enumeration and feature layout.
/// </summary>
public static class NavFeatures
{
    /// <summary>Per-candidate feature width as a multiple of dim: [candFace−goal, candFace−cur, candFace].</summary>
    public const int DiffBlocks = 3;

    /// <summary>Per-candidate feature length given the face dimension: 3·dim (three differential blocks) + 1 (κ).</summary>
    public static int FeatureLength(int dim) => DiffBlocks * dim + 1;

    /// <summary>
    /// Enumerate the K relational candidates of <paramref name="cur"/> and build their differential feature rows
    /// against <paramref name="goalFace"/> and <paramref name="curFace"/>. Candidates come from
    /// <see cref="DialecticalSpace.GetNeighbors"/> (Relational, ordered by edge strength) — capped to <paramref name="k"/>.
    /// Returns an observation whose <see cref="NavObservation.FeaturesFlat"/> is a K·F row-major block (padded rows zero)
    /// and a parallel mask. The candidate symbols/faces are returned so the caller can map an argmax back to a target.
    /// </summary>
    public static NavObservation Build(DialecticalSpace space, string cur, double[] curFace, double[] goalFace, int k, double minConfidence)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(curFace);
        ArgumentNullException.ThrowIfNull(goalFace);
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));

        var dim = goalFace.Length;
        var f = FeatureLength(dim);
        var featuresFlat = new float[k * f];
        var mask = new float[k];

        var syms = new List<string>(k);
        var faces = new List<double[]>(k);

        var neighbours = space.GetNeighbors(cur, PlatonicNeighborhoodType.Relational, k, minConfidence);
        var valid = 0;
        for (var i = 0; i < neighbours.Count && valid < k; i++)
        {
            var nb = neighbours[i];
            if (!space.TryGetConceptFace(nb.Concept, out var candFace) || candFace.Length != dim) continue;

            var baseIdx = valid * f;
            for (var d = 0; d < dim; d++)
            {
                var cf = (float)candFace[d];
                featuresFlat[baseIdx + d] = cf - (float)goalFace[d];        // contrast to the GOAL  (closing the gap?)
                featuresFlat[baseIdx + dim + d] = cf - (float)curFace[d];   // contrast to HERE      (the step taken)
                featuresFlat[baseIdx + 2 * dim + d] = cf;                   // the candidate itself  (absolute hint)
            }
            featuresFlat[baseIdx + 3 * dim] = (float)nb.Confidence;          // edge strength κ
            mask[valid] = 1f;

            syms.Add(nb.Concept);
            faces.Add(candFace);
            valid++;
        }

        return new NavObservation(syms, faces, featuresFlat, mask, valid);
    }
}
