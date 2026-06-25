using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// Runs a TRAINING curriculum while keeping extra units PROBE-ONLY — graded every cycle (so they appear in the train
/// list with a live accuracy + mastery state) but never DECODE-trained. The orchestrator drives training from
/// <see cref="NextTrainBatch"/> (the training curriculum alone) and probes every entry of <see cref="Units"/>; a
/// probe-only unit's <c>NextTrainBatch</c> is therefore never invoked.
///
/// This is for RETRIEVAL skills that are EXERCISED, not gradient-trained — the SEEDED conversational persona: its
/// reply CHUNKS are placed once (whole-reply relations) and in the conscious field the GRU decoder is bypassed, so
/// decode-training a reply would only pollute retrieval. Probing it each cycle keeps it visible + graded (and, with
/// the talk route correctly preserved across reloads, its probes score CORRECT so the credit-assignment module
/// reinforces the chunk edges rather than the disruption module repelling them). See [[nova-talk-by-chunk]].
/// </summary>
public sealed class ProbeAlongsideCurriculum : ITrainingCurriculum
{
    private readonly ITrainingCurriculum _train;
    private readonly IReadOnlyList<ITrainingCurriculum> _probeOnly;

    public ProbeAlongsideCurriculum(ITrainingCurriculum train, params ITrainingCurriculum[] probeOnly)
    {
        _train = train;
        _probeOnly = probeOnly ?? Array.Empty<ITrainingCurriculum>();
    }

    public string Name => _probeOnly.Count == 0 ? _train.Name
        : _train.Name + "+probe(" + string.Join(",", _probeOnly.Select(p => p.Name)) + ")";
    public int Difficulty => _train.Difficulty;
    public IReadOnlyList<string> OperationTokens => _train.OperationTokens;

    // TRAINING is the wrapped curriculum's alone — the probe-only units never contribute decode examples.
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch() => _train.NextTrainBatch();

    // Probe the training units AND the probe-only units (each gated independently by the orchestrator).
    public IReadOnlyList<ITrainingCurriculum> Units => _train.Units.Concat(_probeOnly).ToList();

    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>(); // grading is per-unit
    public void RecordCycle(CycleGrade grade) => _train.RecordCycle(grade);
}
