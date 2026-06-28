using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// The function-word FOUNDATION as a controlled BLEND of two sources, by example fraction per cycle. The majority is the
/// SYNTHETIC prebake — clean, structured frames where function words bridge typed clusters, so the glue-vs-content
/// separation actually FORMS (it SHOWS the pattern). A minority is REAL CORPUS text for vocabulary breadth + naturalness.
///
/// WHY the mix (not either alone): raw corpus ALONE collapses the signal — windowed adjacency makes nearly every word
/// co-occur with every other, so content words look as bridging as function words and the separation averages to ~0 (the
/// problem we kept hitting). Synthetic ALONE overfits to a toy vocabulary. ~80/20 lets the synthetic establish the geometry
/// while the corpus keeps it honest. Both sources warm the SAME space and grade by the SAME property, so SelfAssess reads
/// the blended result. (Two separate foundation curricula don't achieve this — the focuser would train one then the other,
/// never a real mix.)
/// </summary>
public sealed class FoundationBlendCurriculum : ITrainingCurriculum
{
    private readonly ITrainingCurriculum _synthetic; // shows the pattern (prebake)
    private readonly ITrainingCurriculum _corpus;    // breadth / anti-overfit
    private readonly double _syntheticFraction;
    private readonly int _trainPerCycle;
    private readonly Random _rng = new(7);

    public FoundationBlendCurriculum(ITrainingCurriculum synthetic, ITrainingCurriculum corpus,
        int trainPerCycle, double syntheticFraction = 0.8)
    {
        _synthetic = synthetic;
        _corpus = corpus;
        _trainPerCycle = Math.Max(8, trainPerCycle);
        _syntheticFraction = Math.Clamp(syntheticFraction, 0.0, 1.0);
    }

    public string Name => "warm:foundation";
    public int Difficulty => 1;                 // one job: warm the function-word foundation
    public bool IsMastered => _synthetic.IsMastered;

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var synthCount = (int)Math.Round(_trainPerCycle * _syntheticFraction);
        var corpusCount = _trainPerCycle - synthCount;
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        if (synthCount > 0) batch.AddRange(_synthetic.NextTrainBatch().Take(synthCount));
        if (corpusCount > 0) batch.AddRange(_corpus.NextTrainBatch().Take(corpusCount));
        // Interleave so the model never sees a long homogeneous run of one source within a cycle.
        for (var i = batch.Count - 1; i > 0; i--) { var j = _rng.Next(i + 1); (batch[i], batch[j]) = (batch[j], batch[i]); }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>(); // graded by SelfAssess (space property)

    // Both sources warm the SAME space property; read the separation off the synthetic's metric (it owns the clean signal),
    // falling back to the corpus's if the synthetic abstains.
    public double? SelfAssess(GenesisEvalAppRuntime runtime) => _synthetic.SelfAssess(runtime) ?? _corpus.SelfAssess(runtime);

    public void RecordCycle(CycleGrade grade) { _synthetic.RecordCycle(grade); _corpus.RecordCycle(grade); }
}
