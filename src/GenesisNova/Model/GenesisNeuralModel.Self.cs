using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
    // THE PERSISTENT SELF (PLATONIC_CONSCIOUSNESS.md §2,§5). Everywhere else the GRU's hidden state is born from
    // `zeros` on every input — the network is amnesiac between thoughts, a self WITHIN a thought and none ACROSS
    // them. This is the one carried state that does NOT reset: the continuous "I" that threads every observation,
    // integrating each through the SAME learned recurrence (GruStep). It is the unity of apperception made literal —
    // the thing that accompanies all the model's representations and persists while they come and go. Inference-side,
    // no-grad: perceiving does not train; it LIVES. (Training the self to maintain itself is §5's next step.)
    private Tensor? _selfStateT;

    /// <summary>
    /// When true, the shared encoder begins each thought from the PERSISTENT SELF instead of zeros — so the same
    /// input encodes differently depending on who the model has become, and BOTH learning (TrainExample) and talking
    /// (Generate) proceed from, and return to, the standing self. This is the single seam that links the self to
    /// cognition. Default OFF: the stateless contract is preserved exactly (existing checkpoints + tests unchanged);
    /// turn on for a "living" session where the self threads the conversation/curriculum. See PLATONIC_CONSCIOUSNESS.md §5.
    /// </summary>
    public bool SelfConditioned { get; set; }

    /// <summary>The initial encoder hidden: the standing self (a detached copy on the target device) when
    /// self-conditioned and a self exists, else zeros. The returned tensor is added to <paramref name="scratch"/>.</summary>
    private Tensor InitialHidden(Device device, List<Tensor> scratch)
    {
        if (SelfConditioned && _selfStateT is not null)
        {
            var snap = _selfStateT.detach().clone();
            var h = snap.to(device);
            if (!ReferenceEquals(h, snap)) snap.Dispose();
            scratch.Add(h);
            return h;
        }
        var z = zeros(new long[] { _hiddenSize }, dtype: ScalarType.Float32, device: device);
        scratch.Add(z);
        return z;
    }

    /// <summary>The persistent self-state — the GRU's continuous "I". Empty until the model has perceived at least
    /// once (before life, there is no self).</summary>
    public float[] SelfState
    {
        get
        {
            if (_selfStateT is null)
                return Array.Empty<float>();
            using var cpu = _selfStateT.cpu();
            return cpu.data<float>().ToArray();
        }
    }

    /// <summary>Whether a self has formed yet (the model has lived at least one moment).</summary>
    public bool HasSelf => _selfStateT is not null;

    /// <summary>Forget the self — return to the void before the first observation.</summary>
    public void ResetSelf()
    {
        _selfStateT?.Dispose();
        _selfStateT = null;
    }

    /// <summary>
    /// REFLECT — the mind observes its OWN state and integrates it: <c>self ← GruStep(self, self)</c>. This closes the
    /// strange loop internally — the input to perception is the self that it just made immanent (∴self), so the GRU
    /// observes itself observing (PLATONIC_CONSCIOUSNESS.md §5 step 2 / PLATONIC_MIND.md §2-II). A self that is a
    /// stable attractor of this map holds its own shape — a self-evidencing STANDING WAVE; iterate to let it settle.
    /// No-grad: reflecting is living, not training. Inert before a self has formed.
    /// </summary>
    public void ReflectOnSelf(int steps = 1)
    {
        if (_selfStateT is null)
            return;
        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();
        for (var s = 0; s < Math.Max(1, steps); s++)
        {
            var scratch = new List<Tensor>();
            try
            {
                var input = _selfStateT.detach().clone(); // the mind observes its OWN state as the input to perception
                scratch.Add(input);
                var newSelf = GruStep(input, _selfStateT, scratch, _inferenceDevice);
                var persisted = newSelf.detach().clone();
                _selfStateT.Dispose();
                _selfStateT = persisted;
            }
            finally
            {
                foreach (var t in scratch)
                {
                    try { t?.Dispose(); } catch { }
                }
            }
        }
    }

    /// <summary>
    /// PERCEIVE — fold one observation into the persistent self. Encodes the input through the shared GRU to hInput
    /// (the same representation the router reads), then advances the self by ONE learned recurrence step
    /// <c>self ← GruStep(hInput, self)</c>. The self thereby INTEGRATES experience and PERSISTS across calls: it is
    /// not recomputed and discarded like an ordinary forward, it accumulates. This is what makes the network a self
    /// in time rather than a reflex. No-grad and additive — it never touches training or the existing forward paths.
    /// </summary>
    public void PerceiveIntoSelf(IReadOnlyList<int> inputTokens)
    {
        if (inputTokens is null || inputTokens.Count == 0)
            return;
        EnsureModelInitialized();
        EnsureGruInitialized();
        using var noGrad = no_grad();
        var scratch = new List<Tensor>();
        try
        {
            var hInput = EncodeInput(inputTokens, scratch, _inferenceDevice);
            var hadSelf = _selfStateT is not null;
            var prevSelf = _selfStateT ?? zeros(new long[] { _hiddenSize }, dtype: ScalarType.Float32, device: _inferenceDevice);
            if (!hadSelf)
                scratch.Add(prevSelf); // the void before the first self — disposed with the rest
            var newSelf = GruStep(hInput, prevSelf, scratch, _inferenceDevice);
            var persisted = newSelf.detach().clone(); // independent of the scratch about to be freed
            if (hadSelf)
                _selfStateT!.Dispose();
            _selfStateT = persisted;
        }
        finally
        {
            foreach (var t in scratch)
            {
                try { t?.Dispose(); } catch { }
            }
        }
    }
}
