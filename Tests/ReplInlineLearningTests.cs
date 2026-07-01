using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Core;
using GenesisNova.Runtime;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// REPL FEEDBACK-TRAINING — talk to a blank model, react, and it learns to SPEAK in real time. P1 proves a BLANK model,
/// taught a reply inline, speaks it on the very next query (the "make it speak" loop). P2/P3 prove the conversation log
/// (durable in the model folder) round-trips and converts to positive-only training pairs. The interactive talk-test is
/// the user's; these lock the LOGIC.
/// </summary>
public sealed class ReplInlineLearningTests
{
    private readonly ITestOutputHelper _out;
    public ReplInlineLearningTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public async Task P1_BlankModel_TaughtReply_SpeaksInline()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-repl-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var config = new GenesisNovaConfig(
                Backend: ComputeBackend.Cpu, HiddenSize: 256, FaceDimensionOverride: 256,
                AutoResume: false, AutoPersist: false, LocalStateDirectory: dir).WithProductionMechanisms();
            var runtime = new GenesisEvalAppRuntime(config);
            runtime.SetConversationalMode(true);

            // BLANK model → teach a reply → it speaks it immediately (no gym cycle).
            runtime.TeachReply("hello there", "well hello to you too");
            var r1 = (await runtime.PredictAsync("hello there", 16)).Result!;
            _out.WriteLine($"'hello there' → '{r1.Output?.Trim()}' [{r1.DecisionPath}]");
            Assert.Contains("well hello", (r1.Output ?? "").ToLowerInvariant());
            Assert.StartsWith("field-respond", r1.DecisionPath);

            // A SECOND cue speaks its OWN reply (not the first) — distinct cue→reply chunks.
            runtime.TeachReply("how are you", "i am doing just fine thanks");
            var r2 = (await runtime.PredictAsync("how are you", 16)).Result!;
            _out.WriteLine($"'how are you' → '{r2.Output?.Trim()}' [{r2.DecisionPath}]");
            Assert.Contains("just fine", (r2.Output ?? "").ToLowerInvariant());

            // Reaction loop: record the turn, react, and confirm it logged (to the model folder) without breaking speech.
            runtime.RecordConversationTurn("how are you", r2.Output ?? "", r2.DecisionPath, new DateTime(2026, 7, 1));
            runtime.ReactToLast(+1.0, new DateTime(2026, 7, 1));
            var r3 = (await runtime.PredictAsync("how are you", 16)).Result!;
            Assert.Contains("just fine", (r3.Output ?? "").ToLowerInvariant());   // positive reaction didn't break it

            var log = runtime.ConversationLog.Load();
            Assert.Contains(log, t => t.User == "how are you" && t.Reaction is > 0);
            _out.WriteLine($"conversation log @ {runtime.ConversationLog.Path}: {log.Count} turns");
            _out.WriteLine(">>> blank model SPEAKS a taught reply inline via field-respond; reaction logged to the model folder.");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public void P2_ConversationLog_RoundTrips_InModelFolder()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-convlog-" + Guid.NewGuid().ToString("N"));
        try
        {
            var ts = new DateTime(2026, 7, 1, 12, 0, 0, DateTimeKind.Utc);
            var log = new ConversationLog(dir);
            Assert.StartsWith(dir, log.Path);
            var s1 = log.AppendTurn("hello", "hi there friend", "field-respond", ts);
            var s2 = log.AppendTurn("what is 2 + 2", "4", "field-arith", ts);
            log.AppendReactionToLast(-1.0, ts);          // 👎 on the arithmetic turn
            log.AppendReaction(s1, +1.5, ts);            // ❤️ on the greeting turn

            var reloaded = new ConversationLog(dir).Load();  // reopen → parse from disk
            Assert.Equal(2, reloaded.Count);
            Assert.Equal("hello", reloaded[0].User);
            Assert.Equal("hi there friend", reloaded[0].Response);
            Assert.Equal(1.5, reloaded[0].Reaction);
            Assert.Equal(-1.0, reloaded[1].Reaction);
            Assert.Equal(s2, reloaded[1].Seq);
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public void P3_ReplayCurriculum_EmitsPositivePairsOnly()
    {
        var ts = new DateTime(2026, 7, 1);
        var turns = new[]
        {
            new ConversationTurn(1, ts, "hello", "hi there", "field-respond", 1.0),   // 👍 → keep
            new ConversationTurn(2, ts, "bad question", "wrong answer here", "field-relax", -1.0), // 👎 → drop
            new ConversationTurn(3, ts, "no reaction", "meh reply", "field-relax", null),          // unrated → drop
            new ConversationTurn(4, ts, "great one", "excellent reply", "field-respond", 0.5),     // weak+ (>=0.5) → keep
            new ConversationTurn(5, ts, "neutral", "neutral reply", "field-relax", 0.0),           // 0 → drop
        };
        var pairs = ConversationReplayCurriculum.BuildPairs(turns, minReward: 0.5);
        Assert.Equal(2, pairs.Count);
        Assert.Contains(("hello", "hi there"), pairs);
        Assert.Contains(("great one", "excellent reply"), pairs);
        Assert.DoesNotContain(("bad question", "wrong answer here"), pairs);
        Assert.DoesNotContain(("no reaction", "meh reply"), pairs);

        var cur = new ConversationReplayCurriculum(turns);
        Assert.Equal(2, cur.NextTrainBatch().Count);
        Assert.Equal("conversation-replay", cur.Name);
    }
}
