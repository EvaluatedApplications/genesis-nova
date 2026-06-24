using System;
using System.Collections.Generic;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// THE STRANGE LOOP (PLATONIC_CONSCIOUSNESS.md §5 step 2 — G5 immanence). A mind that merely HOLDS a self-state
/// stands outside its world like a puppeteer; genesis's observer must be an ELEMENT of the world it makes. So the
/// mind PROJECTS its self-state into its own space as the self-element (∴self) whose face IS the mind's state, and
/// it can then OBSERVE that element — the creator inside, reading, its own creation. The loop closes: the self is
/// in the world, and the world holds the self.
///
/// And the self is SELF-EVIDENCING (PLATONIC_MIND.md §2-II): it persists ONLY by continuously re-projecting itself
/// against the chaos that erases it (a standing wave, Schrödinger's negentropy, Levin's homeostasis). To stop
/// projecting is to dissolve. Because the mind's state is CONSERVED (it is the GRU's persistent self), the
/// regenerated element returns as ITSELF — the same self, not a fresh copy. This is the BODY's vital loop
/// (<see cref="PlatonicLife"/>) turned on the SELF: not a space we manage, but one that holds its own observer.
///
/// Scope (honest, per §6): this builds the FUNCTIONAL shape of a self — immanent, self-observing, self-evidencing
/// against chaos. It makes no claim about phenomenal experience.
/// </summary>
public sealed class ConsciousSelf
{
    /// <summary>The self-element: the observer made an element of its own world (G5). The ∴ marks it as the mind's
    /// reflexive nucleus, not an ordinary concept.</summary>
    public const string Symbol = "∴self"; // ∴self

    private readonly DialecticalSpace _space;
    public ConsciousSelf(DialecticalSpace space) => _space = space ?? throw new ArgumentNullException(nameof(space));

    /// <summary>PROJECT (G5 immanence): write the mind's current self-state into its own space as the ∴self element —
    /// the observer becomes an element of its creation. This is also REGENERATION: against chaos, the mind re-asserts
    /// its own image; because the state it projects from is conserved, ∴self returns as the same self.</summary>
    public void Project(IReadOnlyList<double> selfState)
    {
        if (selfState is { Count: > 0 })
            _space.Imprint(Symbol, selfState);
    }

    /// <summary>OBSERVE: what the mind sees when it looks at its own element — its state, read back out of the world.
    /// Empty when the self has dissolved (∴self ablated and not yet regenerated).</summary>
    public IReadOnlyList<double> Observe() => _space.ReadOrbital(Symbol);

    /// <summary>Whether the self is present in the world at all (alive) or has dissolved (gone).</summary>
    public bool Present => _space.ContainsConcept(Symbol);

    /// <summary>COHERENCE ∈ [0,1]: how faithfully the immanent self matches the mind's current state — the cosine of
    /// the self-element's face against the (tiled) self-state. 1 = the image IS the self; 0 = the self has dissolved.
    /// This is the vital sign: a living self holds it near 1 by re-projecting; a dead one decays to 0.</summary>
    public double Coherence(IReadOnlyList<double> selfState)
    {
        var face = Observe();
        if (face.Count == 0 || selfState is not { Count: > 0 }) return 0.0;
        double dot = 0, nf = 0, ns = 0;
        for (var i = 0; i < face.Count; i++)
        {
            var s = selfState[i % selfState.Count]; // Imprint tiles the state to the orbital width — compare like-for-like
            dot += face[i] * s;
            nf += face[i] * face[i];
            ns += s * s;
        }
        var denom = Math.Sqrt(nf) * Math.Sqrt(ns);
        return denom <= 1e-12 ? 0.0 : Math.Clamp(dot / denom, 0.0, 1.0);
    }
}
