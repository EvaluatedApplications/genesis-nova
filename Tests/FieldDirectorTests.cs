using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// STAGE 2 — THE LEARNED DIRECTOR. Deciding WHEN to compose vs retrieve is not a hand-heuristic (settle-confidence
// arbitration failed). Here a tiny online controller LEARNS the decision, self-supervised from outcome, off substrate
// features (relation degree, concept count, settle confidences) — and gets held-out routing right where the
// confidence-only heuristic misroutes. Capability CAN emerge; the conductor is learned, not coded.
public sealed class FieldDirectorTests
{
    private readonly ITestOutputHelper _out;
    public FieldDirectorTests(ITestOutputHelper o) => _out = o;

    // Substrate features for the compose-vs-retrieve decision on a 2-concept query.
    private static double[] Features(DialecticalSpace s, string query)
    {
        var content = query.Split(' ', StringSplitOptions.RemoveEmptyEntries).Where(s.ContainsConcept).Distinct().ToList();
        var degs = content.Select(s.GetRelationDegree).ToList();
        var minDeg = degs.Min();
        var composeConf = s.Reason(content).Confidence;
        var subject = content[degs.IndexOf(minDeg)];          // discriminative subject = the most specific (lowest degree)
        var retrieveConf = s.Reason(new[] { subject }).Confidence;
        return new[] { minDeg / (minDeg + 4.0), content.Count / 4.0, composeConf, retrieveConf, composeConf - retrieveConf };
    }

    [Fact]
    public void Director_LearnsWhenToCompose_WhereConfidenceAloneFails()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string a, string b) { for (var i = 0; i < 3; i++) space.FineEditFromExample(new[] { a }, new[] { b }, false); }

        // CATEGORIES — shared by many members (high relation degree), and OVERLAPPING so every compose-pair has a
        // real member (apple is red+fruit+sweet; cherry red+fruit+sweet; ruby red+gem). Composing two narrows to it.
        foreach (var m in new[] { "apple", "cherry", "brick", "rose", "ruby" }) Rel("red", m);
        foreach (var m in new[] { "apple", "cherry", "lemon", "pear", "plum" }) Rel("fruit", m);
        foreach (var m in new[] { "apple", "cherry", "honey", "sugar", "jam" }) Rel("sweet", m);
        foreach (var m in new[] { "ruby", "opal", "jade", "pearl", "topaz" }) Rel("gem", m);
        // A HUB VERB ("describe") that relates to many entities — high degree, but it's not the subject.
        var entities = new[] { "otter", "badger", "heron", "lynx", "raven", "marten", "finch" };
        foreach (var e in entities) Rel("describe", e);
        // ENTITIES — specific (low degree): each has ONE attribute (+ the describe hub).
        var attr = new[] { "amber", "copper", "topaz", "onyx", "ivory", "coral", "slate" };
        for (var i = 0; i < entities.Length; i++) Rel(entities[i], attr[i]);

        // Compose-cases — each pair has a REAL shared member (apple/cherry/ruby), so composing genuinely settles.
        var composeQ = new[] { "red fruit", "sweet fruit", "red sweet", "red gem" }.Where(q => q.Split(' ').All(space.ContainsConcept)).ToArray();
        var retrieveQ = entities.Select(e => $"describe {e}").ToArray();

        // TRAIN on a subset of each kind (held-out queries are NOT trained).
        var trainC = composeQ.Take(2).ToArray();
        var trainR = retrieveQ.Take(4).ToArray();
        var heldC = composeQ.Skip(2).ToArray();
        var heldR = retrieveQ.Skip(4).ToArray();

        var dir = new FieldDirector(featureCount: 5);
        for (var epoch = 0; epoch < 300; epoch++)
        {
            foreach (var q in trainC) dir.Observe(Features(space, q), composeWasCorrect: true);
            foreach (var q in trainR) dir.Observe(Features(space, q), composeWasCorrect: false);
        }

        // The learned director must route HELD-OUT queries correctly.
        var dirRight = 0; var total = 0;
        var heurRight = 0;
        foreach (var q in heldC.Concat(heldR))
        {
            var want = heldC.Contains(q);                 // ground truth: compose-case?
            var f = Features(space, q);
            var dPick = dir.ShouldCompose(f);
            var hPick = f[2] > f[3];                       // the confidence-only heuristic: composeConf > retrieveConf
            _out.WriteLine($"'{q}' want={(want ? "compose" : "retrieve")}  director={(dPick ? "compose" : "retrieve")}  heuristic={(hPick ? "compose" : "retrieve")}");
            if (dPick == want) dirRight++;
            if (hPick == want) heurRight++;
            total++;
        }
        _out.WriteLine($"director {dirRight}/{total}   confidence-heuristic {heurRight}/{total}");

        Assert.Equal(total, dirRight);            // the LEARNED director routes every held-out query correctly
        Assert.True(dirRight > heurRight, "the learned director beats the confidence-only heuristic it replaces");
    }
}
