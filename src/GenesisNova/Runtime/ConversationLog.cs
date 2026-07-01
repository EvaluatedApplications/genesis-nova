using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace GenesisNova.Runtime;

/// <summary>One conversational turn from the REPL + its optional reaction (the emoji feedback reward).</summary>
public sealed record ConversationTurn(int Seq, DateTime Ts, string User, string Response, string DecisionPath, double? Reaction);

/// <summary>
/// APPEND-ONLY conversation log in the MODEL'S folder (next to the checkpoint) — the REPL chat becomes durable, replayable
/// TRAINING DATA (feedback-training loop: talk → react → learn). Two JSONL record kinds: a "turn" and a "react" keyed to a
/// turn's seq; <see cref="Load"/> reconciles them so a reaction that arrives after the turn is folded onto it. Append-only
/// (no rewrite) keeps it crash-safe and cheap. Timestamps are passed IN (the substrate forbids ambient clocks; the caller
/// stamps at the UI/runtime layer).
/// </summary>
public sealed class ConversationLog
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };
    private readonly object _lock = new();
    private int _lastSeq;

    public ConversationLog(string folder, string fileName = "conversation.jsonl")
    {
        Directory.CreateDirectory(folder);
        Path = System.IO.Path.Combine(folder, fileName);
        _lastSeq = File.Exists(Path) ? Load().Select(t => t.Seq).DefaultIfEmpty(0).Max() : 0;
    }

    /// <summary>The JSONL file path (in the model folder).</summary>
    public string Path { get; }

    /// <summary>The seq of the most recently appended turn (0 before any turn) — the target of a following reaction.</summary>
    public int LastSeq { get { lock (_lock) return _lastSeq; } }

    /// <summary>Append a conversational turn; returns its seq so a later reaction can be attached to it.</summary>
    public int AppendTurn(string user, string response, string decisionPath, DateTime ts)
    {
        lock (_lock)
        {
            var seq = ++_lastSeq;
            Write(new Record("turn", seq, ts, user ?? "", response ?? "", decisionPath ?? "", null));
            return seq;
        }
    }

    /// <summary>Attach a reaction (the emoji reward) to a specific turn.</summary>
    public void AppendReaction(int seq, double reward, DateTime ts)
    {
        lock (_lock) Write(new Record("react", seq, ts, null, null, null, reward));
    }

    /// <summary>React to the MOST RECENT turn; returns the seq reacted to (0 if there is no turn yet).</summary>
    public int AppendReactionToLast(double reward, DateTime ts)
    {
        lock (_lock)
        {
            if (_lastSeq > 0) Write(new Record("react", _lastSeq, ts, null, null, null, reward));
            return _lastSeq;
        }
    }

    /// <summary>Read the whole conversation, turns in order, each with its reaction reconciled from later "react" records.</summary>
    public IReadOnlyList<ConversationTurn> Load()
    {
        if (!File.Exists(Path)) return Array.Empty<ConversationTurn>();
        var turns = new Dictionary<int, ConversationTurn>();
        var order = new List<int>();
        foreach (var line in File.ReadAllLines(Path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            Record? rec;
            try { rec = JsonSerializer.Deserialize<Record>(line, Json); } catch { continue; }
            if (rec is null) continue;
            if (rec.Kind == "turn")
            {
                if (!turns.ContainsKey(rec.Seq)) order.Add(rec.Seq);
                turns[rec.Seq] = new ConversationTurn(rec.Seq, rec.Ts, rec.User ?? "", rec.Response ?? "", rec.DecisionPath ?? "", rec.Reaction);
            }
            else if (rec.Kind == "react" && turns.TryGetValue(rec.Seq, out var t))
            {
                turns[rec.Seq] = t with { Reaction = rec.Reaction };
            }
        }
        return order.Select(s => turns[s]).ToList();
    }

    private void Write(Record rec)
        => File.AppendAllText(Path, JsonSerializer.Serialize(rec, Json) + Environment.NewLine);

    private sealed record Record(string Kind, int Seq, DateTime Ts, string? User, string? Response, string? DecisionPath, double? Reaction);
}
