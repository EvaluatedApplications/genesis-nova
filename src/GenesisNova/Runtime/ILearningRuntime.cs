using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using GenesisNova.Cognition;

namespace GenesisNova.Runtime;

/// <summary>
/// The NARROW runtime surface the training learning modules depend on (ISP + DIP): predict (for the correctness
/// gate), the three task→space mechanisms (credit / Rung-1 disruption / Rung-2 gradient), and the op-balance
/// telemetry. The modules depend on THIS — not on the 30-method <see cref="GenesisEvalAppRuntime"/> façade — so a
/// module can be unit-tested against a tiny fake and can't reach training/checkpoint/GPU concerns it has no business
/// touching. <see cref="GenesisEvalAppRuntime"/> implements it; this is the library seam for the learning layer.
/// </summary>
public interface ILearningRuntime
{
    Task<GenesisPredictTaskData?> TryPredictAsync(
        string input, int maxTokens = 48, int gateWaitMilliseconds = 150, CancellationToken cancellationToken = default);

    void ReinforceEvidence(IReadOnlyList<PlatonicEvidence> evidence, bool success);
    void DisruptWrongAnswer(string query, string output);
    void TrainRetrievalToward(string query, IReadOnlyList<string> allowedAnswers);

    IReadOnlyList<long> OpClassBalance { get; }
}
