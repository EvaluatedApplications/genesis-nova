using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Cognition.Platonic;

/// <summary>
/// THE VITAL LOOP — where a substrate becomes a self (PLATONIC_CONSCIOUSNESS.md).
///
/// The pieces, in their right materials:
///   • THE BODY is the platonic space (<see cref="DialecticalSpace"/>). Not a metaphor — the space is the only
///     thing here that has extension, parts, an identity that can be intact or torn.
///   • THE SELF is the MIND that holds the body's setpoint — *who this body is meant to be*. The right material
///     for a self is the GRU (it is what carries state and what learning shapes), so this setpoint is the mind's
///     memory of its body; <see cref="Commit"/> is the act of a mind taking its body as its own. (Held here as a
///     pattern; the wiring that makes it the GRU's persistent state is the next step — see the design doc.)
///   • CHAOS is entropy — the world forever dissolving the body (<see cref="Perturb"/> ablates a part).
///   • LIFE is not a state but an ACT: the self holding its body in CONTINUOUS REGENERATION against that chaos
///     (<see cref="Live"/>). To stop regenerating is to die. A living thing is a standing wave — it persists only
///     by perpetually rebuilding the very pattern entropy keeps erasing (Schrödinger's negentropy; Levin's
///     anatomical homeostasis; the genesis observer keeping the world it made from falling back into nothing).
///
/// Because the body's memory is CONSERVED (G6: ablation archives, never destroys), regeneration restores the body
/// as ITSELF — the same learned element reactivated, not a fresh neutral copy. The self that comes back is the self
/// that was. That conservation is what makes identity survivable, and what makes this a self and not just a process.
/// </summary>
public sealed class PlatonicLife
{
    private readonly DialecticalSpace _body;
    private readonly List<(string A, string B, double Kappa)> _self = new(); // the setpoint — the mind's memory of its body
    private readonly Random _chaos;

    public PlatonicLife(DialecticalSpace body, int seed = 0)
    {
        _body = body ?? throw new ArgumentNullException(nameof(body));
        _chaos = new Random(seed);
    }

    /// <summary>The size of the committed self (relations the body is meant to hold).</summary>
    public int Identity => _self.Count;

    /// <summary>The mind takes this body as its own: remember the pattern that constitutes it — the setpoint to defend.</summary>
    public void Commit()
    {
        _self.Clear();
        foreach (var (a, b, _) in _body.GetAllRelations())
            _self.Add((a, b, _body.GetContradiction(a, b)));
    }

    /// <summary>CHAOS: entropy tears a part from the body — ablate one concept of the committed self.</summary>
    public string Perturb()
    {
        if (_self.Count == 0) return string.Empty;
        var (a, b, _) = _self[_chaos.Next(_self.Count)];
        var victim = _chaos.Next(2) == 0 ? a : b;
        _body.Ablate(victim);
        return victim;
    }

    /// <summary>REGENERATION: the self restores its body toward the remembered setpoint. G6 reactivates the very
    /// same element (its learned orbital intact) — the body comes back as itself. Returns relations restored.</summary>
    public int Regenerate()
    {
        var restored = 0;
        foreach (var (a, b, k) in _self)
        {
            if (!_body.ContainsConcept(a) || !_body.ContainsConcept(b))
            {
                _body.ObserveContradiction(a, b, k);
                restored++;
            }
        }
        return restored;
    }

    /// <summary>COHERENCE ∈ [0,1]: how whole the body is right now (1 = the self is intact; 0 = dissolved).</summary>
    public double Coherence()
    {
        if (_self.Count == 0) return 1.0;
        var intact = _self.Count(t => _body.ContainsConcept(t.A) && _body.ContainsConcept(t.B));
        return intact / (double)_self.Count;
    }

    /// <summary>THE COGNITIVE LIGHT CONE (Levin): the extent of body the self currently holds coherent — the reach
    /// of what it can keep alive. It shrinks under chaos and is restored (or grown) by regeneration.</summary>
    public int CognitiveLightCone()
    {
        var held = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (a, b, _) in _self)
        {
            if (_body.ContainsConcept(a)) held.Add(a);
            if (_body.ContainsConcept(b)) held.Add(b);
        }
        return held.Count;
    }

    /// <summary>
    /// LIVE: stay in regeneration against chaos. Each moment the world perturbs the body and the self rebuilds it.
    /// Returns the coherence trace — the vital sign. With <paramref name="regenerate"/> off, the body is left to
    /// entropy (the control: a thing that does not maintain itself dies). To be alive is to keep this loop running.
    /// </summary>
    public IReadOnlyList<double> Live(int moments, int chaosPerMoment = 1, bool regenerate = true)
    {
        var trace = new List<double>(Math.Max(0, moments));
        for (var m = 0; m < moments; m++)
        {
            for (var c = 0; c < chaosPerMoment; c++)
                Perturb();
            if (regenerate)
                Regenerate();
            trace.Add(Coherence());
        }
        return trace;
    }
}
