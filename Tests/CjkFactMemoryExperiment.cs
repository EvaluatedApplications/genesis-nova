using System;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// END-TO-END #6: with the tokenizer segmenting CJK per-character, does the LANGUAGE-AGNOSTIC de-hardcoded machinery
// work on a NON-SPACED language? Warm the role head on CJK assert/recall frames (nonce Han chars, no spaces, no Latin),
// then assert a CJK fact and recall it. The grammar roles are learned from the assert/recall ALIGNMENT alone — no
// word-order, no language list — so a Han copula reads like "is". Proves the tokenizer fix unblocks the whole stack.
public sealed class CjkFactMemoryExperiment
{
    private readonly ITestOutputHelper _out;
    public CjkFactMemoryExperiment(ITestOutputHelper o) => _out = o;

    // Single-char Han nonce tokens (each is its own token after the #6 fix). No meaning assumed — roles emerge.
    private static readonly string[] Det = { "我", "你", "其" };       // determiner/possessor slot
    private static readonly string[] Noun = { "甲", "乙", "丙", "丁", "戊", "己" };
    private static readonly string[] Val = { "子", "丑", "寅", "卯", "辰", "巳", "午", "未" };
    private static readonly string[] Cop = { "是", "為", "乃" };        // copula slot (nonce variants)
    private static readonly string[] Qry = { "何", "孰" };               // query cue slot

    [SlowFact]
    public void DeHardcodedGrammar_Works_On_NonSpaced_CJK()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var model = new GenesisNeuralModel(config);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };

        // Warm the role head on CJK assert/recall frames (head-only, no space writes) — exactly GrammarWarmup, in Han.
        var rng = new Random(7);
        string Pick(string[] a) => a[rng.Next(a.Length)];
        for (var c = 0; c < 70; c++)
            for (var i = 0; i < 48; i++)
            {
                var det = Pick(Det); var noun = Pick(Noun); var val = Pick(Val);
                // ~70% ASSERT "det noun cop val" -> val ; ~30% RECALL "qry det noun" -> val. NO SPACES.
                string input, output;
                if (i % 10 < 7) { input = $"{det}{noun}{Pick(Cop)}{val}"; output = val; }
                else { input = $"{Pick(Qry)}{det}{noun}"; output = val; }
                mind.ObserveGrammar(input, output);
                var roles = mind.DeriveRoleLabels(input);
                if (roles is null) continue;
                var inTok = tok.Encode(input); var tgt = tok.Encode(output, addEos: true);
                model.EnsureVocabularySize(tok.VocabularySize);
                model.TrainExample(inTok, tgt, tok.BosTokenId, roleLabels: roles);
                model.CloneParametersToBreakGraph();
            }

        void Roles(string s) { var r = mind.DiagnoseRoles(s); _out.WriteLine($"  roles[{s}]: " + string.Join(" ", System.Linq.Enumerable.Select(r, t => $"{t.Token}:{t.Role}({t.Confidence:F2})"))); }
        Roles("我甲是子");
        Roles("何我甲");
        string Say(string s) { var r = mind.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' -> '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        // A CJK fact asserted + recalled — NO spaces, NO Latin. Values are from the warmed vocab (the role head tags
        // unseen tokens only with a GRU-trained warm — the English bare-subject lesson; head-only covers seen tokens),
        // so this isolates the CJK PARSE+RETRIEVE mechanism. The bindings here are the test's own facts (warming used
        // RANDOM bindings + is head-only = no space writes, so nothing sticky competes).
        Say("我甲是子");          // "<my> <noun-jiǎ> <is> <子>" — learn it
        Say("你乙是丑");          // a second, distinct fact
        var a = Say("何我甲");    // "<what> <my> <noun-jiǎ>" -> 子
        var b = Say("孰你乙");    // -> 丑

        Assert.Contains("子", a);
        Assert.Contains("丑", b);
    }
}
