using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// DIAGNOSTIC: why does "alice is doctor" / "what is alice" (bare single-word subject, real value) fail where
// "my name is stephen" works? Warm the role head, then watch the DECISION PATHS of assert + recall + role tags.
public sealed class BareSubjectDiagnostic
{
    private readonly ITestOutputHelper _out;
    public BareSubjectDiagnostic(ITestOutputHelper o) => _out = o;

    [SlowFact]
    public void Trace_BareSubject_AssertAndRecall()
    {
        var config = new GenesisNovaConfig(HiddenSize: 256, FaceDimensionOverride: 256);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var model = new GenesisNeuralModel(config);
        GrammarWarmup.WarmRoleHeadWithGym(tok, model, space, config);
        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };

        void Roles(string s)
        {
            var r = mind.DiagnoseRoles(s);
            _out.WriteLine($"  roles[{s}] = " + string.Join(" ", System.Linq.Enumerable.Select(r, t => $"{t.Token}:{t.Role}({t.Confidence:F2})")));
        }
        string Say(string s) { var r = mind.Generate(new GenerationRequest(s, 8)); _out.WriteLine($"  '{s}' -> '{r.Output?.Trim()}' [{r.DecisionPath}]"); return r.Output?.Trim() ?? ""; }

        Roles("alice is doctor");
        Roles("what is alice");
        Roles("my favorite color is indigo");
        Roles("what is my favorite color");
        Say("my favorite color is indigo");
        Say("what is my favorite color");
        Say("alice is doctor");
        Say("bob is teacher");
        Say("what is alice");
        Say("what is bob");
        // contrast: the known-good possessive case
        Say("my name is stephen");
        Say("what is my name");
        Assert.True(true);
    }
}
