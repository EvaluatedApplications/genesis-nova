using System;
using System.Collections.Generic;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// THE NN LAYER, REWORKED — first kernel (PLATONIC_MIND.md). NOT an encoder→router→decoder. The model's working
/// state is a THOUGHT in the field, and it generates by RELAXATION: inject the prompt, relax to a basin (reason),
/// SPEAK the settled concept, fold it back, relax again. Every token is spoken FROM a settled state; when the field
/// does not settle it ABSTAINS instead of inventing (the cure for the old neural decoder's hallucinations).
///
/// This is the pure-relaxation SKELETON — it proves the loop and the abstention with no learned weights at all. The
/// next milestones make the dynamics learnable and then immanent (a self-model face), so the NN and the field finish
/// becoming one entity. The old GRU model is left untouched until this earns its place.
/// </summary>
public sealed class FieldCognition
{
    private readonly DialecticalSpace _field;
    private readonly IReadOnlyList<IFieldOperator> _operators;

    public FieldCognition(DialecticalSpace field, IReadOnlyList<IFieldOperator>? operators = null)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _operators = operators ?? Array.Empty<IFieldOperator>();
    }

    /// <summary>
    /// Think a continuation of <paramref name="prompt"/>: at each step the thought either APPLIES an operator
    /// (compute / compose — the homomorphism etc.) when one fires for the context, or RELAXES to a concept basin
    /// (retrieval). The previous results are folded back in so the thought MOVES. Stops the moment nothing fires and
    /// the field no longer settles — the mind falls silent rather than inventing.
    /// </summary>
    public IReadOnlyList<string> Think(IReadOnlyList<string> prompt, int tokens, double settleThreshold = 0.2)
    {
        var context = new List<string>(prompt ?? Array.Empty<string>());
        var spoken = new List<string>();
        for (var t = 0; t < Math.Max(0, tokens); t++)
        {
            // (compute) the lowest-surprise move when the context affords an operator — apply it.
            string? produced = null;
            foreach (var op in _operators)
            {
                produced = op.TryApply(context);
                if (produced != null) break;
            }
            if (produced == null)
            {
                // (retrieve) else relax to a concept basin.
                var thought = _field.Reason(context, settleThreshold: settleThreshold);
                if (!thought.Settled || string.IsNullOrEmpty(thought.Symbol))
                    break; // nothing to compute and nothing coherent to recall → abstain
                produced = thought.Symbol;
            }
            spoken.Add(produced);
            context.Add(produced); // fold the result into the thought
        }
        return spoken;
    }
}
