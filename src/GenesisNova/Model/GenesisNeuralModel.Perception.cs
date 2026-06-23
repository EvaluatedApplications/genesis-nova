using GenesisNova.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Model;

public partial class GenesisNeuralModel
{
    // h: [..., hidden] → [..., ReasoningTrunkDim]. The parameterless form uses the training-device params; the
    // static form takes device-moved weights for the inference path.
    private Tensor ReasoningTrunk(Tensor h) => ReasoningTrunk(h, _trunkW!, _trunkB!);
    private static Tensor ReasoningTrunk(Tensor h, Tensor tw, Tensor tb) => nn.functional.relu(h.matmul(tw) + tb);

    private void EnsureReasoningTrunk()
    {
        if (_trunkW is not null)
            return;
        _trunkW = new TorchSharp.Modules.Parameter(
            ((rand(new long[] { _hiddenSize, ReasoningTrunkDim }, device: _trainingDevice) * 2.0) - 1.0) * 0.05, true);
        _trunkB = new TorchSharp.Modules.Parameter(
            zeros(new long[] { ReasoningTrunkDim }, dtype: ScalarType.Float32, device: _trainingDevice), true);
        RecreateOptimizer();
    }

    /// <summary>
    /// Predict how strongly the platonic space should be edited for this input context.
    /// Encodes the input tokens through the SHARED GRU to hInput — the same learned representation the
    /// router reads — passes it (detached) through the learned edit-head linear layer, and squashes
    /// with a sigmoid to a bounded magnitude in [0,1]. Deterministic given weights. Returns 0.5
    /// (neutral) when there is no signal: empty input or an uninitialized model/head.
    /// hInput is detached implicitly here (the whole call runs under no_grad), so the edit-head reads a
    /// fixed snapshot of the shared encoder and never backprops into it via this path.
    /// </summary>
    private static float[] PerceptionFloats(double[] perception)
    {
        var f = new float[EditPerceptionDim];
        for (var i = 0; i < EditPerceptionDim && i < perception.Length; i++)
            f[i] = (float)Math.Clamp(perception[i], -4.0, 4.0);
        return f;
    }
}
