using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace GenesisNova.Train;

/// <summary>
/// Memory + code INDEX curriculum — the ClaudeMemory daemon's training, rewired into the unified orchestrator
/// with the SAME list-generation + grading parity. Scans a memory index file (MEMORY.md, with `[[links]]` from
/// each entry's body) and a code root into <c>find</c>/<c>contains</c>/<c>calls</c> facts; builds each cue's FULL
/// allowed-answer set (one-to-many held as MANY single-member edges — the space's native strength, never a
/// comma-list target); broad-trains a random whole-corpus sample every cycle (the proven anti-forgetting fix);
/// and ramps a GLOBAL difficulty (how many valid answers a cue must surface) as held-out quality holds. Grading
/// is <see cref="GenesisGrader"/> (fuzzy full-list, require-platonic) — identical behaviour to the daemon.
/// Op-tokens (find/contains/calls) are route triggers; the orchestrator registers them via <see cref="OperationTokens"/>.
/// </summary>
public sealed class MemoryCodeCurriculum : ITrainingCurriculum
{
    private const int MaxAnswersPerCue = 3;
    private const double RampBar = 0.85;
    private const int RampStable = 3;

    private static readonly Regex IndexLine = new(@"^\s*-\s*\[(?<name>[^\]]+)\]\([^)]+\)\s*(?:—|--|-)\s*(?<desc>.+)$", RegexOptions.Compiled);
    private static readonly Regex ReType = new(@"\b(?:class|record|struct|interface|enum)\s+([A-Za-z_]\w*)", RegexOptions.Compiled);
    private static readonly Regex ReMember = new(@"^\s*public\s+(?:static\s+|async\s+|virtual\s+|override\s+|sealed\s+|abstract\s+|readonly\s+|partial\s+|new\s+|unsafe\s+|extern\s+)*[\w<>\[\],\.\?]+\s+([A-Za-z_]\w*)\s*[\(\{]", RegexOptions.Compiled);
    private static readonly Regex ReCall = new(@"([A-Za-z_]\w*)\s*\(", RegexOptions.Compiled);
    private static readonly Regex ReCamel = new(@"(?<=[a-z0-9])(?=[A-Z])|[_\s]+", RegexOptions.Compiled);
    private static readonly HashSet<string> Stop = new(StringComparer.OrdinalIgnoreCase)
    {
        "the","and","via","not","with","into","that","from","for","its","now","was","are","but","this","than",
        "then","per","each","you","your","when","which","what","how","why","can","could","should","would","must",
        "over","under","out","all","any","more","most","they","them","their","get","use","used","using","run",
        "new","old","see","read","add","like","has","have","public","private","protected","internal","static",
        "void","string","int","bool","double","var","async","await","task","return","null","true","false","base",
        "class","struct","enum","interface","record","value","result","name","node","model","list","type","data",
    };

    private readonly string _indexPath;
    private readonly string _codeRoot;
    private readonly string? _bridgesPath;
    private readonly Random _rng = new();
    private readonly int _trainPerCycle;

    private readonly List<(string Input, string Output)> _corpus = new(); // single-member edges = the broad-training corpus
    private readonly List<TrainingProbe> _probes = new();
    private int _globalDifficulty = 1;
    private int _rampStreak;

    public string Name => "memory+code";
    public int Difficulty => _globalDifficulty;
    public IReadOnlyList<string> OperationTokens => new[] { "find", "contains", "calls" };

    /// <param name="indexPath">Memory index file (MEMORY.md). Its directory holds the per-entry .md bodies (for [[links]]).</param>
    /// <param name="codeRoot">Code directory to index (*.cs, excluding bin/obj); empty to skip code.</param>
    /// <param name="bridgesPath">Optional hand-authored Q=>key / key&lt;=&gt;key augmentation file.</param>
    public MemoryCodeCurriculum(string indexPath, string codeRoot, string? bridgesPath = null, int trainPerCycle = 96)
    {
        _indexPath = indexPath ?? string.Empty;
        _codeRoot = codeRoot ?? string.Empty;
        _bridgesPath = bridgesPath;
        _trainPerCycle = Math.Max(16, trainPerCycle);
        Build();
    }

    /// <summary>Number of cues and single-member edges (broad corpus) — for the host to display.</summary>
    public (int Cues, int Edges, int Probes) Stats => (_corpus.Select(e => e.Input).Distinct().Count(), _corpus.Count, _probes.Count);

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>();
        if (_corpus.Count == 0) return batch;
        var n = Math.Min(_trainPerCycle, _corpus.Count);
        for (var i = 0; i < n; i++) batch.Add(_corpus[_rng.Next(_corpus.Count)]); // broad rehearsal: random whole-corpus slice
        return batch;
    }

    public IReadOnlyList<TrainingProbe> NextProbes()
        => _probes.Select(p => new TrainingProbe(p.Query, p.Allowed, Math.Max(1, Math.Min(_globalDifficulty, p.Allowed.Count)))).ToList();

    public void RecordCycle(CycleGrade grade)
    {
        // GLOBAL difficulty ramp: require more valid answers per cue as held-out quality holds above the bar.
        if (grade.Accuracy >= RampBar)
        {
            if (++_rampStreak >= RampStable && _globalDifficulty < MaxAnswersPerCue) { _globalDifficulty++; _rampStreak = 0; }
        }
        else _rampStreak = 0;
    }

    // ── Faithful port of the daemon's fact + probe generation ──────────────────────────────────────────────
    private void Build()
    {
        var byCue = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        void Add(string cue, string resp)
        {
            cue = cue.Trim(); resp = resp.Trim();
            if (cue.Length == 0 || resp.Length == 0) return;
            if (!byCue.TryGetValue(cue, out var l)) byCue[cue] = l = new List<string>();
            if (!l.Contains(resp, StringComparer.OrdinalIgnoreCase)) l.Add(resp);
        }
        foreach (var (cue, response, relate) in GenerateTruth()) { Add(cue, response); if (relate) Add(response, cue); }

        _corpus.Clear();
        foreach (var (cue, all) in byCue)
            foreach (var ans in all.Take(MaxAnswersPerCue))   // a few clean single edges; the rest emerge as valid over-generation
                _corpus.Add((cue, ans));

        BuildProbes();
    }

    private List<(string Cue, string Response, bool Relate)> GenerateTruth()
    {
        var facts = new List<(string, string, bool)>();
        var memDir = File.Exists(_indexPath) ? Path.GetDirectoryName(_indexPath) : null;
        if (File.Exists(_indexPath))
            foreach (var raw in File.ReadLines(_indexPath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                var m = IndexLine.Match(line);
                if (m.Success)
                {
                    var name = m.Groups["name"].Value.Trim();
                    foreach (var kw in Keywords(m.Groups["desc"].Value))
                        facts.Add(($"find {kw}", Key(name), false));
                    if (memDir is not null)
                    {
                        var file = Path.Combine(memDir, name + ".md");
                        if (File.Exists(file))
                            foreach (Match lm in Regex.Matches(File.ReadAllText(file), @"\[\[([^\]]+)\]\]"))
                            {
                                var target = lm.Groups[1].Value.Trim();
                                if (target.Length > 0 && !target.Equals(name, StringComparison.OrdinalIgnoreCase))
                                {
                                    facts.Add(($"find {Key(name)}", Key(target), false));
                                    facts.Add(($"find {Key(target)}", Key(name), false));
                                }
                            }
                    }
                }
                else if (line.Contains("<=>")) { var p = line.Split("<=>", 2); facts.Add((Key(p[0].Trim()), Key(p[1].Trim()), true)); }
                else if (line.Contains("=>")) { var p = line.Split("=>", 2); facts.Add((Key(p[0].Trim()), Key(p[1].Trim()), false)); }
            }
        if (_bridgesPath is not null && File.Exists(_bridgesPath))
            foreach (var raw in File.ReadLines(_bridgesPath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                if (line.Contains("<=>")) { var p = line.Split("<=>", 2); facts.Add((Key(p[0].Trim()), Key(p[1].Trim()), true)); }
                else if (line.Contains("=>")) { var p = line.Split("=>", 2); facts.Add((Key(p[0].Trim()), Key(p[1].Trim()), false)); }
            }
        facts.AddRange(GenerateCodeFacts());
        return facts;
    }

    private List<(string, string, bool)> GenerateCodeFacts()
    {
        var facts = new List<(string, string, bool)>();
        if (string.IsNullOrEmpty(_codeRoot) || !Directory.Exists(_codeRoot)) return facts;
        var allLines = new List<string[]>();
        foreach (var f in Directory.EnumerateFiles(_codeRoot, "*.cs", SearchOption.AllDirectories))
            if (!f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") &&
                !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"))
                try { allLines.Add(File.ReadAllLines(f)); } catch { }

        var symbols = new HashSet<string>(StringComparer.Ordinal);
        var declared = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var lines in allLines)               // PASS 1 — declarations, keyword facts, type⟷member containment
        {
            var doc = new List<string>(); string? currentType = null;
            for (var i = 0; i < lines.Length; i++)
            {
                var t = lines[i].TrimStart();
                if (t.StartsWith("///")) { doc.Add(t.TrimStart('/').Trim()); continue; }
                string? sym = null; var isType = false;
                var tm = ReType.Match(lines[i]);
                if (tm.Success) { sym = tm.Groups[1].Value; isType = true; }
                else { var mm = ReMember.Match(lines[i]); if (mm.Success) sym = mm.Groups[1].Value; }
                if (sym is not null && sym.Length >= 3)
                {
                    symbols.Add(sym);
                    if (isType) currentType = sym;
                    if (declared.Add(sym))
                    {
                        facts.Add(($"find {sym.ToLowerInvariant()}", sym, false));
                        var nameKw = ReCamel.Split(sym).Where(s => s.Length >= 3).Select(s => s.ToLowerInvariant());
                        var docKw = doc.SelectMany(d => Regex.Matches(d.ToLowerInvariant(), "[a-z][a-z0-9]{2,}").Select(mm => mm.Value));
                        foreach (var kw in nameKw.Concat(docKw).Where(k => !Stop.Contains(k)).Distinct().Take(4))
                            facts.Add(($"find {kw}", sym, false));
                    }
                    if (!isType && currentType is not null && !currentType.Equals(sym, StringComparison.OrdinalIgnoreCase))
                        facts.Add(($"contains {currentType}", sym, false));
                }
                doc.Clear();
            }
        }

        var callPairs = new HashSet<string>(); const int MaxCalls = 2500;
        foreach (var lines in allLines)               // PASS 2 — call graph: caller ⟷ callee (both known symbols)
        {
            if (callPairs.Count >= MaxCalls) break;
            string? currentMember = null;
            foreach (var raw in lines)
            {
                var t = raw.TrimStart();
                if (t.StartsWith("//")) continue;
                if (ReType.IsMatch(raw)) { currentMember = null; continue; }
                var mm = ReMember.Match(raw);
                if (mm.Success) currentMember = mm.Groups[1].Value;
                if (currentMember is null) continue;
                foreach (Match cm in ReCall.Matches(raw))
                {
                    var callee = cm.Groups[1].Value;
                    if (callee.Length < 3 || callee.Equals(currentMember, StringComparison.Ordinal) || !symbols.Contains(callee)) continue;
                    if (callPairs.Add($"{currentMember}|{callee}")) facts.Add(($"calls {currentMember}", callee, false));
                    if (callPairs.Count >= MaxCalls) break;
                }
            }
        }
        return facts;
    }

    private void BuildProbes()
    {
        _probes.Clear();
        var entries = new List<(string Name, List<string> Kws)>();
        if (File.Exists(_indexPath))
            foreach (var raw in File.ReadLines(_indexPath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                var m = IndexLine.Match(line);
                if (!m.Success) continue;
                entries.Add((m.Groups["name"].Value.Trim(), Keywords(m.Groups["desc"].Value).Where(k => !k.Contains('-')).ToList()));
            }
        // FULL allowed-answer set per cue: a `find <kw>` is satisfied by ANY memory carrying <kw> (one-to-many).
        var kwToNames = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, kws) in entries)
            foreach (var kw in kws)
            {
                if (!kwToNames.TryGetValue(kw, out var ns)) kwToNames[kw] = ns = new List<string>();
                if (!ns.Contains(name, StringComparer.OrdinalIgnoreCase)) ns.Add(name);
            }
        IReadOnlyList<string> Allowed(string kw, string fallback)
            => kwToNames.TryGetValue(kw, out var ns) && ns.Count > 0 ? ns.Select(Key).ToList() : new List<string> { Key(fallback) };
        foreach (var (name, kws) in entries)
        {
            foreach (var ho in kws.Skip(3).Take(2)) _probes.Add(new TrainingProbe($"find {ho}", Allowed(ho, name), 1)); // held-out operand
            if (kws.Count > 0) _probes.Add(new TrainingProbe($"find {kws[0]}", Allowed(kws[0], name), 1));               // in-sample
        }
        if (_bridgesPath is not null && File.Exists(_bridgesPath))
            foreach (var raw in File.ReadLines(_bridgesPath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#') || line.Contains("<=>") || !line.Contains("=>")) continue;
                var p = line.Split("=>", 2);
                _probes.Add(new TrainingProbe(p[0].Trim(), new[] { Key(p[1].Trim()) }, 1));
            }
    }

    // A memory key is a hyphenated filename (nova-claude-memory) — trained as a UNIQUE SINGLE TOKEN (hyphens
    // stripped) so the decoder learns it cleanly; phrases (spaces) and paths/symbols are untouched.
    private static string Key(string v) => string.IsNullOrEmpty(v) || v.Contains(' ') ? v : v.Replace("-", string.Empty);

    private static IEnumerable<string> Keywords(string text)
    {
        var terms = new List<string>();
        void Consider(string t) { if (t.Length >= 3 && !Stop.Contains(t)) terms.Add(t); }
        foreach (Match m in Regex.Matches(text.ToLowerInvariant(), "[a-z0-9][a-z0-9-]*"))
        {
            var tok = m.Value.Trim('-');
            if (tok.Length == 0) continue;
            Consider(tok);
            if (tok.Contains('-'))
                foreach (var part in tok.Split('-', StringSplitOptions.RemoveEmptyEntries)) Consider(part);
        }
        return terms.Distinct(StringComparer.OrdinalIgnoreCase).Take(20);
    }
}
