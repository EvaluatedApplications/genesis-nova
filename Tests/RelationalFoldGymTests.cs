using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// Does the relational fold fire IN THE HOT PATH on the HARDER gym shape? The plain gym Category is 1-hop retrieval
/// (apple→fruit, a direct edge). The harder shape is a HELD-OUT member: shared attributes taught, the category
/// WITHHELD — answering it is the embedding mirror-fold (reasoning), not retrieval. This MEASURES routing: teach the
/// held-out substrate the way the trainer's ObservePlatonicSpace couples input→output (member↔category, member↔attr),
/// then probe the FULL field ladder (GenerateFromField, prod flags) and report each answer's DecisionPath — to see
/// whether relax greedily returns an attribute BEFORE the bridge infers the category, or the bridge actually wins.
/// </summary>
public sealed class RelationalFoldGymTests
{
    private readonly ITestOutputHelper _out;
    public RelationalFoldGymTests(ITestOutputHelper o) => _out = o;

    private static readonly (string cat, string[] attrs, string[] known, string[] heldOut)[] Families =
    {
        ("fruit",  new[]{"sweet","juicy","edible"},  new[]{"apple","banana","cherry","pear"}, new[]{"grape","plum"}),
        ("animal", new[]{"breathes","moves","eats"}, new[]{"dog","cat","horse","cow"},        new[]{"wolf","deer"}),
        ("tool",   new[]{"metal","held","works"},    new[]{"hammer","wrench","saw","drill"},  new[]{"pliers","chisel"}),
    };

    [Fact]
    public void HeldOutCategory_InHotPath_ReportsWhetherBridgeWinsOverRelax()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        // Prod field flags: think-by-field, learned cues, bridge (mirror-fold) + relational fold both ON — the hot path.
        var infer = new GenesisInferenceEngine(tok, model, space, null)
        { ConsciousField = true, LearnedCuesOnly = true, BridgeReasoning = true, RelationalFold = true };

        // TEACH as the trainer's ObservePlatonicSpace couples input→output: known members carry (category + attrs);
        // held-out members carry attrs ONLY (category withheld). Repeat so the attribute clouds settle and overlap.
        for (var pass = 0; pass < 25; pass++)
        {
            foreach (var (cat, attrs, known, _) in Families)
                foreach (var m in known)
                {
                    space.FineEditFromExample(new[] { m }, new[] { cat }, isNegativeExample: false);
                    foreach (var a in attrs) space.FineEditFromExample(new[] { m }, new[] { a }, isNegativeExample: false);
                }
            foreach (var (_, attrs, _, ho) in Families)
                foreach (var m in ho)
                    foreach (var a in attrs) space.FineEditFromExample(new[] { m }, new[] { a }, isNegativeExample: false);
        }
        space.FlushCloudBatch();

        (string outp, string path) Ask(string q)
        { var r = infer.Generate(new GenerationRequest(q, 16)); return ((r.Output ?? "").Trim(), r.DecisionPath ?? ""); }

        var heldOut = Families.SelectMany(f => f.heldOut.Select(m => (m, f.cat))).ToArray();
        int correct = 0, viaBridge = 0;
        foreach (var (m, cat) in heldOut)
        {
            var (outp, path) = Ask($"what kind of thing is {m}");
            var hit = string.Equals(outp, cat, StringComparison.OrdinalIgnoreCase);
            if (hit) correct++;
            if (path == "field-bridge") viaBridge++;
            _out.WriteLine($"HELD-OUT {m,-7} → '{outp,-10}' [{path,-14}] want {cat,-7} {(hit ? "HIT" : "miss")}");
        }
        _out.WriteLine($"\nheld-out category: {correct}/{heldOut.Length} correct, {viaBridge}/{heldOut.Length} via field-bridge");
        _out.WriteLine(viaBridge > 0
            ? ">>> the bridge FIRES in the hot path on held-out members — relational reasoning, not retrieval."
            : ">>> relax pre-empts the bridge (returns an attribute first). The bridge is wired but a routing yield is needed.");

        // The relational fold must WIN in the hot path: held-out members resolve to their category via the mirror-fold
        // (field-bridge), NOT a bare sibling from relax. This is the routing fix (bridge runs before relax) locked in.
        Assert.True(correct >= heldOut.Length - 1, $"held-out categories should resolve by reasoning, got {correct}/{heldOut.Length}");
        Assert.True(viaBridge >= heldOut.Length - 1, $"resolution must route through the mirror-fold (field-bridge), got {viaBridge}/{heldOut.Length}");
    }
}
