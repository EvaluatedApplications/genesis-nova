using System.Collections.Generic;
using System.Linq;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// TRAINING-EXAMPLE CREATOR from the REPL chat: turns the durable <see cref="ConversationLog"/> into graded training
/// examples so a feedback conversation can REPLAY in the gym. Only POSITIVELY-reacted turns become (cue→reply)
/// reinforcement pairs — a 👍/❤️ turn is a confirmed good exchange; negative/neutral/un-reacted turns are skipped (a wrong
/// or unrated answer must not be reinforced). This is the bridge from "talk + react" to durable substrate learning.
/// </summary>
public sealed class ConversationReplayCurriculum : ITrainingCurriculum
{
    private readonly IReadOnlyList<(string Input, string Output)> _pairs;

    public ConversationReplayCurriculum(ConversationLog? log, double minReward = 0.5)
        : this(log?.Load() ?? System.Array.Empty<ConversationTurn>(), minReward) { }

    public ConversationReplayCurriculum(IEnumerable<ConversationTurn> turns, double minReward = 0.5)
        => _pairs = BuildPairs(turns, minReward);

    /// <summary>The positively-reacted (cue → reply) pairs — the confirmed-good exchanges worth reinforcing. The reply must
    /// be NATURAL LANGUAGE (a multi-word phrase): the talk/chunk route only ever utters multi-word chunks, so a single-word
    /// target is unlearnable AS SPEECH and must not become a training example — the creator obeys the SAME rule the REPL
    /// teach does (a taught reply must be speakable). Single-word "positive" turns are dropped, not reinforced.</summary>
    public static IReadOnlyList<(string Input, string Output)> BuildPairs(IEnumerable<ConversationTurn> turns, double minReward = 0.5)
        => turns.Where(t => t.Reaction is double r && r >= minReward
                            && !string.IsNullOrWhiteSpace(t.User) && IsNaturalLanguageResponse(t.Response))
                .Select(t => (t.User.Trim(), t.Response.Trim()))
                .ToList();

    /// <summary>A response the talk route can actually SPEAK = natural language = ≥2 words. Enforced on every example
    /// creator so training targets match what the model can produce (never a bare single token as a spoken reply).</summary>
    public static bool IsNaturalLanguageResponse(string? s)
        => !string.IsNullOrWhiteSpace(s) && s.Trim().Split((char[]?)null, System.StringSplitOptions.RemoveEmptyEntries).Length >= 2;

    public string Name => "conversation-replay";
    public int Difficulty => 1;
    public int PairCount => _pairs.Count;

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch() => _pairs;

    public IReadOnlyList<TrainingProbe> NextProbes()
        => _pairs.Select(p => new TrainingProbe(p.Input, new[] { p.Output }, 1)).ToList();

    public void RecordCycle(CycleGrade grade) { }
}
