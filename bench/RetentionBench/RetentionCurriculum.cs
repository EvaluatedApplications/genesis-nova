using GenesisNova.Runtime;
using GenesisNova.Train;

namespace RetentionBench;

// A continual-learning curriculum over the procedural fact universe. Trains WAVE by WAVE: it stays on the current wave
// (with light rehearsal of earlier waves) until that wave is mastered, then advances — introducing the next slice of
// facts. Difficulty == the highest wave introduced, so the space GROWS as the run proceeds. Facts are framed exactly
// like the gym's Category skill (GymTrainer.CategoryFrames + rotating cruft) so they train+retrieve on the proven path
// and no framing word becomes a constant correlate (the hub-avoidance lesson).
public sealed class RetentionCurriculum : ITrainingCurriculum
{
    private readonly ProceduralKnowledge _k;
    private readonly Random _rng;
    private readonly int _trainPerCycle;
    private readonly int _probeCount;
    private readonly double _rehearsalFraction;   // share of each train batch drawn from EARLIER waves (anti-forgetting interleave)
    private readonly double _bar;
    private int _wave;                              // highest wave introduced so far (== Difficulty)
    private int _streak;

    private static readonly string[] LeadIns = { "", "", "", "tell me, ", "quick one, ", "so, ", "i forget, ", "remind me, " };
    private static readonly string[] Trailers = { "", "", "", "", " please", " for me", " again", " you know" };

    public RetentionCurriculum(ProceduralKnowledge k, int trainPerCycle = 96, int probeCount = 48,
                               double rehearsalFraction = 0.30, double masteryBar = 0.80, int seed = 7)
    {
        _k = k; _trainPerCycle = trainPerCycle; _probeCount = probeCount;
        _rehearsalFraction = rehearsalFraction; _bar = masteryBar; _rng = new Random(seed);
    }

    public string Name => "retention";
    public int Difficulty => _wave;
    public int WavesIntroduced => _wave + 1;
    public int FactsIntroduced => Math.Min(_k.Facts.Length, WavesIntroduced * _k.WaveSize);

    private string Frame(string entity)
    {
        var f = GymTrainer.CategoryFrames[_rng.Next(GymTrainer.CategoryFrames.Length)];
        return LeadIns[_rng.Next(LeadIns.Length)] + string.Format(f, entity) + Trailers[_rng.Next(Trailers.Length)];
    }

    // A FIXED, cruft-free frame for held-out retention probing, so the same fact is queried identically over time.
    public static string ProbeFrame(string entity) => "what kind of thing is " + entity;

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string, string)>(_trainPerCycle);
        for (var i = 0; i < _trainPerCycle; i++)
        {
            // Mostly the current wave (new learning); a rehearsal share from a random earlier wave (resist forgetting).
            var wave = (_wave > 0 && _rng.NextDouble() < _rehearsalFraction) ? _rng.Next(_wave) : _wave;
            var seg = _k.Wave(wave);
            var (entity, cat) = seg[_rng.Next(seg.Count)];
            batch.Add((Frame(entity), cat));
        }
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
    {
        // Probe the CURRENT wave (the learning signal that drives mastery + wave advancement). AnswerVocabulary = all
        // category labels (the competing answers), so the grader scores selectivity. RequirePlatonic off: we grade on
        // correctness, not on which route fired.
        var seg = _k.Wave(_wave);
        var probes = new List<TrainingProbe>(_probeCount);
        for (var i = 0; i < _probeCount && seg.Count > 0; i++)
        {
            var (entity, cat) = seg[_rng.Next(seg.Count)];
            probes.Add(new TrainingProbe(Frame(entity), new[] { cat }, RequiredDepth: 1,
                AnswerVocabulary: _k.Categories, RequirePlatonic: false, SurfaceStrict: false));
        }
        return probes;
    }

    public void RecordCycle(CycleGrade grade)
    {
        if (grade.Accuracy >= _bar) _streak++; else _streak = 0;
        if (_streak >= 3 && _wave < _k.WaveCount - 1) { _wave++; _streak = 0; } // mastered → introduce the next wave
    }

    public double? SelfAssess(GenesisEvalAppRuntime runtime) => null; // use probe grading
    public int MasteryDepth => 1;
}
