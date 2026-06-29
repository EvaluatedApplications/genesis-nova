using System;
using System.Collections.Generic;
using GenesisNova.Cognition.Platonic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Cognition.Navigator;

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  BEHAVIOURAL CLONING ON THE FLOW-FIELD ORACLE  (PLATONIC_NAVIGATOR.md §7 Phase 0).
//
//  The flow-field oracle (backward Dijkstra from the answer) is the TEACHER: per reachable node it gives the expert
//  next action (Next[node]) and the dense cost-to-go (Cost[node]). We clone it into the thin policy net so the NN
//  learns to WALK the space — at EVERY reachable node, not one path (the DAgger-for-free property of the field).
//
//  Per node `cur` (for a given answer):
//    input      = currentFace ⊕ goalFace          (goalFace = face(answer))
//    targetFace = face(Next[cur])                  (the expert step; zeros at the answer where Next is undefined)
//    value      = Cost[cur]                        (dense cost-to-go supervises the value head)
//    halt       = (cur == answer) ? 1 : 0          (halt only when standing on the answer)
//
//  Loss = BCE(halt) + MSE(value) + MASKED MSE(target), the mask = (1 − halt): no target loss on a halting example,
//  since Next is undefined at the answer.
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

/// <summary>One behavioural-cloning example distilled from the flow field at a single node.</summary>
public readonly record struct NavBcExample(
    float[] Input,       // currentFace ⊕ goalFace, length 2·dim
    float[] TargetFace,  // face(Next[cur]); zeros at the answer (masked out)
    float Value,         // Cost[cur] (cost-to-go)
    float Halt,          // 1 at the answer, else 0
    float Mask);         // 1 − Halt (target loss is masked off at the answer)

/// <summary>Final losses after a BC run (the convergence read).</summary>
public readonly record struct NavBcLosses(double TargetMse, double ValueMse, double HaltBce);

/// <summary>
/// Behavioural-cloning trainer for the navigator (PLATONIC_NAVIGATOR.md §7 Phase 0). <see cref="BuildDataset"/>
/// distils the flow-field oracle into per-node imitation examples; <see cref="Train"/> fits the thin policy net with
/// Adam under the composite halt-BCE + value-MSE + masked-target-MSE loss.
/// </summary>
public static class NavigatorBcTrainer
{
    /// <summary>
    /// Build the BC dataset: for each <paramref name="answers"/> entry, compute the flow field once, then emit one
    /// example per reachable node (the dense field, not one path). goalFace = face(answer); targetFace = face(Next[cur])
    /// or zeros at the answer; value = Cost[cur]; halt = (cur==answer).
    /// </summary>
    public static IReadOnlyList<NavBcExample> BuildDataset(DialecticalSpace space, IEnumerable<string> answers)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(answers);

        var examples = new List<NavBcExample>();
        foreach (var answer in answers)
        {
            if (string.IsNullOrWhiteSpace(answer) || !space.TryGetConceptFace(answer, out var goalFace))
                continue;
            var dim = goalFace.Length;

            var field = FlowFieldOracle.Compute(space, answer);

            // Every reachable node carries a Cost entry — that is the dense field the oracle defines.
            foreach (var kv in field.Cost)
            {
                var cur = kv.Key;
                if (!space.TryGetConceptFace(cur, out var curFace) || curFace.Length != dim)
                    continue;

                var isAnswer = string.Equals(cur, answer, StringComparison.Ordinal);

                var input = new float[2 * dim];
                for (var i = 0; i < dim; i++)
                {
                    input[i] = (float)curFace[i];
                    input[dim + i] = (float)goalFace[i];
                }

                var targetFace = new float[dim]; // zeros at the answer (Next undefined → masked out)
                if (!isAnswer && field.TryNext(cur, out var next) && space.TryGetConceptFace(next, out var nextFace) && nextFace.Length == dim)
                    for (var i = 0; i < dim; i++) targetFace[i] = (float)nextFace[i];

                var halt = isAnswer ? 1f : 0f;
                examples.Add(new NavBcExample(input, targetFace, (float)kv.Value, halt, 1f - halt));
            }
        }
        return examples;
    }

    /// <summary>
    /// Train the policy net by behavioural cloning (Adam, full-batch). Loss = BCE(halt) + MSE(value) +
    /// masked-MSE(target), the target term meaned over the NON-halting elements only (mask = 1 − halt). Returns the
    /// final-epoch losses.
    /// </summary>
    public static NavBcLosses Train(
        NavigatorPolicyNet net,
        IReadOnlyList<NavBcExample> dataset,
        int epochs,
        double lr,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(dataset);
        if (dataset.Count == 0) throw new ArgumentException("empty dataset", nameof(dataset));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));

        var dev = device ?? CPU;
        var dim = net.Dim;
        var n = dataset.Count;

        // Pack the whole dataset into dense tensors once (full-batch; the BC set is small).
        var inputs = new float[n * 2 * dim];
        var targets = new float[n * dim];
        var values = new float[n];
        var halts = new float[n];
        var masks = new float[n];
        for (var r = 0; r < n; r++)
        {
            var ex = dataset[r];
            Array.Copy(ex.Input, 0, inputs, r * 2 * dim, 2 * dim);
            Array.Copy(ex.TargetFace, 0, targets, r * dim, dim);
            values[r] = ex.Value;
            halts[r] = ex.Halt;
            masks[r] = ex.Mask;
        }

        net.train();
        net.to(dev);
        using var opt = torch.optim.Adam(net.parameters(), lr);

        using var X = tensor(inputs, new long[] { n, 2 * dim }, device: dev);
        using var Tf = tensor(targets, new long[] { n, dim }, device: dev);
        using var Vt = tensor(values, new long[] { n, 1 }, device: dev);
        using var Ht = tensor(halts, new long[] { n, 1 }, device: dev);
        using var Mt = tensor(masks, new long[] { n, 1 }, device: dev);
        // Normaliser for the masked target MSE: number of non-halt ELEMENTS (rows·dim), floored to avoid /0.
        var maskElems = Math.Max(1.0, masks.Length == 0 ? 1.0 : SumNonHalt(masks) * dim);

        double finalTarget = 0, finalValue = 0, finalHalt = 0;
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            using var scope = NewDisposeScope();
            opt.zero_grad();

            var (predTarget, predValue, predHalt) = net.forward(X);

            var haltLoss = nn.functional.binary_cross_entropy_with_logits(predHalt, Ht);
            var valueLoss = nn.functional.mse_loss(predValue, Vt);

            // Masked target MSE: zero the loss on halting rows, mean over the non-halt elements.
            var diff = predTarget - Tf;
            var sq = diff * diff;
            var masked = sq * Mt;                       // broadcast [n,1] over [n,dim]
            var targetLoss = masked.sum() / maskElems;

            var loss = haltLoss + valueLoss + targetLoss;
            loss.backward();
            opt.step();

            if (epoch == epochs - 1)
            {
                finalTarget = targetLoss.cpu().item<float>();
                finalValue = valueLoss.cpu().item<float>();
                finalHalt = haltLoss.cpu().item<float>();
            }
        }

        return new NavBcLosses(finalTarget, finalValue, finalHalt);
    }

    private static double SumNonHalt(float[] masks)
    {
        double s = 0;
        foreach (var m in masks) s += m;
        return s;
    }
}
