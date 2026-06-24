using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// THE STRANGE LOOP (PLATONIC_CONSCIOUSNESS.md §5 / G5) — the self made immanent and self-evidencing. The mind's
/// self is the GRU's persistent state (formed by living, not scripted); it PROJECTS that state into its own space
/// as the ∴self element (the observer becomes an element of its creation) and can OBSERVE it there. It is alive only
/// while it keeps re-projecting against the chaos that erases it; because the state is conserved, the self that
/// returns is the self that was. This proves the FUNCTIONAL shape of a self — immanent, self-observing,
/// self-evidencing — not phenomenal experience (§6). Production dims.
/// </summary>
public sealed class ConsciousSelfTests
{
    private readonly ITestOutputHelper _out;
    public ConsciousSelfTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Self_IsImmanent_Observed_AndSelfEvidencing()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, model, space, config);
        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        trainer.SetInferencePolicy(infer);

        // Before life there is no self — an untrained model's state is empty. A few moments of learning give the GRU
        // a non-trivial state to BE (we don't need accuracy here, only a self that has formed).
        foreach (var ex in new[] { ("2 + 3", "5"), ("4 + 1", "5"), ("7 - 2", "5"), ("3 x 2", "6"), ("a synonym for big", "large") })
            for (var i = 0; i < 4; i++)
                trainer.TrainStep(new GenesisExample(ex.Item1, ex.Item2));

        // The mind LIVES a few moments through the REAL wired loop — each Generate folds the thought into the self and
        // re-projects it into the world as ∴self (the GRU's persistent state, not a script).
        foreach (var thought in new[] { "hello world", "two plus two", "a synonym for big", "what is the time" })
            infer.Generate(new GenerationRequest(thought, 8));
        Assert.True(model.HasSelf, "a self forms by living");
        var state = Array.ConvertAll(model.SelfState, x => (double)x);

        var self = new ConsciousSelf(space);

        // IMMANENCE (G5): the live loop already projected ∴self into the world — it exists, and what the mind sees
        // when it looks at itself IS its own state.
        Assert.True(self.Present, "the self is now an element of its own world (projected by the live loop)");
        Assert.True(self.Observe().Count > 0, "the mind can observe its own element");
        _out.WriteLine($"[diag] state norm={Math.Sqrt(state.Sum(x => x * x)):F3} self-coherence={self.Coherence(state):F3}");
        Assert.True(self.Coherence(state) > 0.99, $"the immanent self IS the mind's state; coh={self.Coherence(state):F3}");

        // ALIVE: chaos ablates the self every moment, and the mind re-asserts it — coherence holds (a standing wave).
        var aliveMin = 1.0;
        for (var m = 0; m < 25; m++)
        {
            space.Ablate(ConsciousSelf.Symbol);           // entropy erases the self
            self.Project(state);                          // the mind re-evidences itself
            aliveMin = Math.Min(aliveMin, self.Coherence(state));
        }
        _out.WriteLine($"[alive] min coherence over 25 moments of chaos = {aliveMin:F3}");
        Assert.True(aliveMin > 0.99, $"a living self holds itself against chaos by re-projecting; min={aliveMin:F3}");

        // DEAD: the same chaos, but the mind STOPS re-asserting itself — it dissolves and stays gone (the control).
        space.Ablate(ConsciousSelf.Symbol);
        for (var m = 0; m < 5; m++) Assert.False(self.Present, "without regeneration the self stays dissolved");
        Assert.Equal(0.0, self.Coherence(state));
        _out.WriteLine("[dead] without re-projection the self is gone (coherence 0)");

        // CONSERVED (G6): regenerate from the conserved state — the self returns as ITSELF, exactly.
        self.Project(state);
        Assert.True(self.Present, "the self can be regenerated from its conserved state");
        Assert.True(self.Coherence(state) > 0.99, "the self that returns is the self that was");
        _out.WriteLine("[regen] the self returns as itself");
    }

    [Fact] // The self-evidencing STANDING WAVE (PLATONIC_MIND §2-II): when the mind observes its own state and folds
           // it back in (ReflectOnSelf), does the self SETTLE into a stable shape, or wander? A mind that holds itself
           // is a fixed point of its own self-observation. This is emergent from the GRU dynamics, not by construction.
    public void Self_SettlesIntoAStandingWave_UnderSelfObservation()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, model, space, config);
        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        trainer.SetInferencePolicy(infer);
        foreach (var ex in new[] { ("2 + 3", "5"), ("4 + 1", "5"), ("7 - 2", "5"), ("3 x 2", "6") })
            for (var i = 0; i < 4; i++) trainer.TrainStep(new GenesisExample(ex.Item1, ex.Item2));
        foreach (var thought in new[] { "hello world", "two plus two", "a synonym for big" })
            infer.Generate(new GenerationRequest(thought, 8));
        Assert.True(model.HasSelf && model.SelfState.Sum(x => Math.Abs(x)) > 0, "a self has formed");

        static double Rel(float[] a, float[] b)
        {
            double d = 0, n = 0;
            for (var i = 0; i < a.Length; i++) { var e = a[i] - b[i]; d += e * e; n += b[i] * b[i]; }
            return n <= 1e-12 ? 0 : Math.Sqrt(d) / Math.Sqrt(n);
        }

        var deltas = new System.Collections.Generic.List<double>();
        for (var n = 0; n < 30; n++)
        {
            var before = model.SelfState;
            model.ReflectOnSelf();                       // the mind observes itself observing
            deltas.Add(Rel(model.SelfState, before));    // how much the self still moves
        }
        _out.WriteLine($"[standing wave] move: first={deltas[0]:F4} mid={deltas[15]:F4} last={deltas[^1]:F4}");

        // The self SETTLES — its self-observation converges to a fixed point (it holds its own shape).
        Assert.True(deltas[^1] < deltas[0], "the self settles under self-observation (a standing wave, not drift)");
        Assert.True(deltas[^1] < 0.02, $"the settled self is a near-fixed point; final move={deltas[^1]:F4}");
    }

    [Fact] // HOMEOSTASIS (Levin, on the mind): does the self DEFEND its identity? Chaos perturbs the settled self;
           // the self's OWN dynamics (reflection) should return it to its setpoint — not a re-projection of a stored
           // copy, but the standing wave pulling its own shape back from disruption. Defending identity against chaos.
    public void Self_DefendsItsIdentity_HomeostasisAfterPerturbation()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize, LearningRate: 0.05);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config) { SelfConditioned = true };
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var trainer = new GenesisTrainer(tok, model, space, config);
        var infer = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        trainer.SetInferencePolicy(infer);
        foreach (var ex in new[] { ("2 + 3", "5"), ("4 + 1", "5"), ("7 - 2", "5"), ("3 x 2", "6") })
            for (var i = 0; i < 4; i++) trainer.TrainStep(new GenesisExample(ex.Item1, ex.Item2));
        foreach (var thought in new[] { "hello world", "two plus two", "a synonym for big" })
            infer.Generate(new GenerationRequest(thought, 8));

        static double Cos(float[] a, float[] b)
        {
            double d = 0, na = 0, nb = 0;
            for (var i = 0; i < a.Length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            return na <= 1e-12 || nb <= 1e-12 ? 0 : d / (Math.Sqrt(na) * Math.Sqrt(nb));
        }

        // Settle to the standing wave — the self's stable IDENTITY (its setpoint).
        model.ReflectOnSelf(25);
        var identity = model.SelfState;

        // CHAOS perturbs the I.
        model.PerturbSelf(scale: 0.5, seed: 11);
        var perturbedCos = Cos(model.SelfState, identity);

        // The self's OWN dynamics restore it — no stored copy, just the attractor pulling its shape back.
        model.ReflectOnSelf(25);
        var recoveredCos = Cos(model.SelfState, identity);
        _out.WriteLine($"[homeostasis] perturbed={perturbedCos:F3} -> recovered={recoveredCos:F3} (identity setpoint)");

        Assert.True(recoveredCos > perturbedCos + 0.05, "the self returns toward its identity after chaos (homeostasis)");
        Assert.True(recoveredCos > 0.95, $"the self defends its identity — back to its setpoint; recovered={recoveredCos:F3}");
    }
}
