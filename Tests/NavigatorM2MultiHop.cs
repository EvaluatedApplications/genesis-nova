using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using TorchSharp;
using Xunit;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace GenesisNova.Tests;

/// <summary>
/// M2 — MULTI-HOP via CROSS-RELATION COMPOSITION + LOOKAHEAD (PROJECT_PLAN.md M2). The honest acceptance: a LIVE
/// head-to-head over the REAL <see cref="GenesisEvalAppRuntime"/> with <see cref="GenesisNovaConfig.WithProductionMechanisms"/>
/// (navigator ON by default), where the navigator beats the one-shot/old-ladder baseline on HELD-OUT multi-hop queries
/// the 1-hop heuristic provably cannot answer:
///   • CROSS-RELATION COMPOSITION — "what COUNTRY does X live in" is answered by composing person→city→country (a
///     different relation type each hop). The 1-hop baseline reaches the wrong intermediate (the city) or a face-near
///     distractor country; the navigator walks 2 hops to the country and halts THERE because it is conditioned on the
///     TARGET KIND (the "country" hub face — learned as a high-degree category, no relation-name word list).
///   • LOOKAHEAD TRAP — each person also has a face-near distractor country one hop away (so the query reaches the
///     ambiguous branch, AND a greedy-nearest step is wrong). The flow-field oracle routes AROUND it through the city;
///     the navigator learned that lookahead, the 1-hop baseline falls for the trap.
///   • KIND SELECTION — the SAME held-out anchor under DIFFERENT kinds ("what country" vs "what industry") composes to
///     DIFFERENT answers along DIFFERENT chains — which the M1 abstraction-LEVEL cue cannot express (both are 2-hop).
/// The policy is trained by <see cref="GenesisEvalAppRuntime.TrainNavigatorCycle"/> ALONE (the deepened sampler emits the
/// composition + trap pairs); the queried instances are HELD OUT (planted only AFTER nav training). HONEST: the reported
/// numbers ARE the result; loose floors only. Gated [SlowFact] behind RUN_SLOW=1.
/// </summary>
public sealed class NavigatorM2MultiHop
{
    private readonly ITestOutputHelper _out;
    public NavigatorM2MultiHop(ITestOutputHelper o) => _out = o;

    private const int ObservePerEdge = 5;
    private const string CountryHub = "country";
    private const string IndustryHub = "industry";

    private static GenesisNovaConfig BuildConfig(string dir) =>
        new GenesisNovaConfig(
            HiddenSize: 64,                 // tiny GRU — ConsciousField bypasses the neural decoder; the navigator
                                            // (fixed face-dim/2048) + the full-size substrate are what is under test
            Backend: ComputeBackend.Cpu,
            AutoPersist: false,
            AutoResume: false,
            LocalStateDirectory: dir)
        .WithProductionMechanisms() with
        {
            FieldTicks = false,
            MeaningOps = false,
        };

    // A person record: the person, their city, their country (the country-query answer), an optional company + industry
    // (the industry-query answer, for the same-anchor kind-selection cases), and a distractor country one hop away (the
    // lookahead trap — face-near, wrong, makes the query ambiguous and a greedy step wrong).
    private readonly record struct Person(string Name, string City, string Country, string? Company, string? Industry, string Distractor);

    private static readonly string[] TrainCountries = { "france", "germany", "italy", "japan", "brazil", "egypt" };
    private static readonly string[] TrainCities    = { "paris", "berlin", "rome", "tokyo", "rio", "cairo" };
    private static readonly string[] TrainPersons   = { "alice", "bob", "carl", "dan", "emma", "finn" };
    // four of the six train people also work (company→industry), so the kind-selection signal is trained.
    private static readonly string[] TrainCompanies  = { "acme", "globex", "initech", "umbrella" };
    private static readonly string[] TrainIndustries = { "tech", "finance", "retail", "energy" };

    private static List<Person> BuildTrainPeople()
    {
        var people = new List<Person>();
        for (var i = 0; i < TrainPersons.Length; i++)
        {
            var distractor = TrainCountries[(i + 1) % TrainCountries.Length]; // a WRONG country, one hop away (the trap)
            string? company = i < TrainCompanies.Length ? TrainCompanies[i] : null;
            string? industry = i < TrainIndustries.Length ? TrainIndustries[i] : null;
            people.Add(new Person(TrainPersons[i], TrainCities[i], TrainCountries[i], company, industry, distractor));
        }
        return people;
    }

    // HELD-OUT people (planted only AFTER nav training): all-new cities/countries/industries → a true generalisation test.
    private static List<Person> BuildHeldOutPeople() => new()
    {
        new Person("dave", "madrid",  "spain",   null,    null,      "italy"),
        new Person("erin", "athens",  "greece",  null,    null,      "japan"),
        new Person("fred", "lima",    "peru",    null,    null,      "france"),
        new Person("gina", "oslo",    "norway",  null,    null,      "germany"),
        new Person("hank", "quito",   "ecuador", "zeta",  "biotech", "brazil"),  // two-kind: country AND industry
        new Person("iris", "accra",   "ghana",   "omega", "aero",    "egypt"),   // two-kind
    };

    private static void Edge(DialecticalSpace ds, string a, string b)
    {
        for (var i = 0; i < ObservePerEdge; i++) ds.ObserveContradiction(a, b, 0.0);
    }

    private static void PlantPerson(DialecticalSpace ds, Person p)
    {
        Edge(ds, p.City, p.Country);        // city  --inCountry--> country   (hop 2 of the composition)
        Edge(ds, p.Country, CountryHub);    // country is-a country           (the kind the answer belongs to)
        Edge(ds, p.Name, p.City);           // person --livesIn--> city        (hop 1)
        Edge(ds, p.Name, p.Distractor);     // person --near--> wrong country  (the LOOKAHEAD TRAP, one hop, face-near)
        if (p.Company is { } co && p.Industry is { } ind)
        {
            Edge(ds, co, ind);              // company --partOf--> industry
            Edge(ds, ind, IndustryHub);     // industry is-a industry
            Edge(ds, p.Name, co);           // person --worksAt--> company
        }
    }

    private static void PlantTrainingGraph(DialecticalSpace ds, List<Person> people)
    {
        foreach (var p in people) PlantPerson(ds, p);
        // Pad the two category hubs well above any country/industry so they are unambiguously the kinds (degree gradient,
        // mirrors M1's root padding). Filler members are degree-1 leaves — they enlarge the hub, never compete as a kind.
        for (var i = 0; i < 6; i++) { Edge(ds, $"cfill{i}", CountryHub); Edge(ds, $"ifill{i}", IndustryHub); }
        // A CLEAR 1-hop case with a single dominant relation — owned by the relation-first heuristic BEFORE the navigator
        // branch, so the hook on/off must be byte-identical (no regression to the ballpark cases).
        Edge(ds, "sol", "helios");
        ds.FlushCloudBatch();
    }

    [SlowFact]
    public async Task Navigator_MultiHopComposition_AndLookaheadTrap_BeatsOneShot_Live()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navm2-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            var eng = runtime.State.Inference;
            var trainPeople = BuildTrainPeople();
            PlantTrainingGraph(ds, trainPeople);

            // CUTOVER PROOF — WithProductionMechanisms turned NavigatorDisambiguation on, so the runtime wired the hook.
            var navHook = eng.NavigatorDisambiguator;
            Assert.NotNull(navHook);
            var net = runtime.State.Navigator;
            _out.WriteLine($"=== M2 MULTI-HOP LIVE: {ds.NodeCount} concepts | navigator {net.Dim}/{net.Hidden} | " +
                           $"country-hub deg={ds.GetRelationDegree(CountryHub)} industry-hub deg={ds.GetRelationDegree(IndustryHub)} | nav ON via WithProductionMechanisms ===");

            async Task<(string Out, string Path)> Ask(string query)
            {
                var r = (await runtime.PredictAsync(query, 8)).Result;
                return (r?.Output?.Trim() ?? string.Empty, r?.DecisionPath ?? string.Empty);
            }
            static string CountryQ(string p) => $"what country does {p} live in";
            static string IndustryQ(string p) => $"what industry does {p} work in";

            // ── (0) COLD-NAVIGATOR FALL-THROUGH (cutover safety): BEFORE any nav training the policy is random; an
            //    ambiguous multi-hop query must NOT be confidently mis-resolved — it must FALL THROUGH (no navigator-walk). ──
            var coldRows = new List<(string Q, string Out, string Path)>();
            foreach (var p in trainPeople) { var (o, pa) = await Ask(CountryQ(p.Name)); coldRows.Add((CountryQ(p.Name), o, pa)); }
            var coldWalks = coldRows.Count(r => r.Path == "navigator-walk");
            _out.WriteLine($"--- COLD navigator: {coldRows.Count} ambiguous multi-hop queries | navigator-walk emits = {coldWalks} ---");
            foreach (var r in coldRows.Take(4)) _out.WriteLine($"    cold '{r.Q}' -> '{r.Out}' [{r.Path}]");

            // ── (1) TRAIN THE NAVIGATOR VIA TrainNavigatorCycle ALONE — proving the DEEPENED sampler emits the
            //    kind-conditioned composition + lookahead-trap pairs (no hand-fed BC). ──
            for (var c = 0; c < 45; c++)
            {
                var r = runtime.TrainNavigatorCycle(maxMembers: 64, epochs: 7);
                if (c % 9 == 0 || c == 44)
                    _out.WriteLine($"    TrainNavigatorCycle {c}: loss={r.Loss:F4} queries={r.Queries} resolve={r.ResolvePct:P0}");
            }

            // ── (2) PLANT THE HELD-OUT PEOPLE (new cities/countries/industries) — never seen by nav training. ──
            var heldOut = BuildHeldOutPeople();
            foreach (var p in heldOut) PlantPerson(ds, p);
            ds.FlushCloudBatch();

            // M3: gym nav training no longer writes the shared engine self (the self is now shaped only by INFERENCE
            // resolutions). This clear is therefore a no-op belt-and-braces — kept so an isolated M2 reading of the
            // navigator's COMPOSITION is provably self-free (purely kind-conditioned, matching the diagnostic walk).
            eng.RestoreSelfField(null);

            var device = net.parameters().FirstOrDefault()?.device ?? CPU;

            // DIAGNOSTIC — the trained policy composes a HELD-OUT 2-hop walk under the KIND face (proves >1 hop traversed,
            // not a lucky 1-hop), with NO answer supplied (goal-emergent: only anchor + kind face).
            (bool Reached, int Steps, string Final, IReadOnlyList<string> Traj) KindWalk(string member, string hub, string expect)
            {
                if (!ds.TryGetConceptFace(member, out var anchorFace) || !ds.TryGetConceptFace(hub, out var kindFace))
                    return (false, 0, member, new[] { member });
                using var policy = new QueryNavPolicy(net, ds, anchorFace, (int)NavCue.Genus, device, NavQueryDaggerTrainer.DefaultK,
                    0.0, 0.5, selfVec: null, kindFace: kindFace);
                var res = new NavigatorWalk().Walk(ds, member, anchorFace, null, policy, new NavWalkOptions(MaxSteps: 8));
                return (policy.LastHalt && string.Equals(res.FinalSymbol, expect, StringComparison.Ordinal), res.Steps, res.FinalSymbol, res.Trajectory);
            }
            _out.WriteLine("=== HELD-OUT KIND-CONDITIONED WALK (goal-emergent; trajectory proves the hop count) ===");
            foreach (var p in heldOut)
            {
                var w = KindWalk(p.Name, CountryHub, p.Country);
                _out.WriteLine($"    [{p.Name}|country ] reached={w.Reached} hops={w.Steps} final='{w.Final}' traj={string.Join("->", w.Traj)}");
                if (p.Industry is { } ind)
                {
                    var wi = KindWalk(p.Name, IndustryHub, ind);
                    _out.WriteLine($"    [{p.Name}|industry] reached={wi.Reached} hops={wi.Steps} final='{wi.Final}' traj={string.Join("->", wi.Traj)}");
                }
            }

            // ── THE LIVE HEAD-TO-HEAD on the HELD-OUT multi-hop subset, through the REAL PredictAsync. ──
            async Task<(double Acc, int Walks, List<(string Q, string Out, string Path, string Exp, bool Hit)> Rows)> RunArm(bool navOn, IEnumerable<(string Q, string Exp)> qs)
            {
                eng.NavigatorDisambiguator = navOn ? navHook : null;
                var rows = new List<(string, string, string, string, bool)>();
                foreach (var (q, exp) in qs)
                {
                    var (o, pa) = await Ask(q);
                    rows.Add((q, o, pa, exp, string.Equals(o, exp, StringComparison.OrdinalIgnoreCase)));
                }
                var acc = 100.0 * rows.Count(r => r.Item5) / Math.Max(1, rows.Count);
                return (acc, rows.Count(r => r.Item3 == "navigator-walk"), rows);
            }

            var countryQs = heldOut.Select(p => (CountryQ(p.Name), p.Country)).ToList();
            var industryQs = heldOut.Where(p => p.Industry is not null).Select(p => (IndustryQ(p.Name), p.Industry!)).ToList();

            var baseArm = await RunArm(navOn: false, countryQs);
            var navArm = await RunArm(navOn: true, countryQs);
            var baseInd = await RunArm(navOn: false, industryQs);
            var navInd = await RunArm(navOn: true, industryQs);
            eng.NavigatorDisambiguator = navHook; // restore production wiring

            _out.WriteLine("=== HELD-OUT 'what country does X live in' → expect the 2-hop COMPOSED country ===");
            _out.WriteLine($"    {"person",-8} | {"expect",-9} | {"BASELINE (1-hop)",-28} | NAVIGATOR (kind=country)");
            for (var i = 0; i < navArm.Rows.Count; i++)
            {
                var b = baseArm.Rows[i]; var n = navArm.Rows[i];
                _out.WriteLine($"    {heldOut[i].Name,-8} | {b.Item4,-9} | {b.Item2 + " [" + b.Item3 + "]",-28} {(b.Item5 ? "OK" : "..")} | " +
                               $"{n.Item2 + " [" + n.Item3 + "]"} {(n.Item5 ? "OK" : "..")}");
            }
            _out.WriteLine("=== SAME-ANCHOR KIND SELECTION: 'what industry does X work in' → expect the 2-hop COMPOSED industry ===");
            for (var i = 0; i < navInd.Rows.Count; i++)
            {
                var n = navInd.Rows[i]; var b = baseInd.Rows[i];
                _out.WriteLine($"    {n.Item1,-34} | base='{b.Item2}'[{b.Item3}] {(b.Item5 ? "OK" : "..")} | nav='{n.Item2}'[{n.Item3}] {(n.Item5 ? "OK" : "..")}");
            }

            // ── LOOKAHEAD TRAP — EVERY country query is a trap: each person has a face-near distractor country ONE hop
            //    away (same KIND as the goal — the hardest form). The baseline always lands on a 1-hop neighbour (the city
            //    or the distractor), NEVER the 2-hop goal; the navigator routes around the trap through the city. We
            //    report the per-query landings and a headline case where the navigator clears the trap. ──
            var navReach = navArm.Rows.Count(r => r.Item5);
            var baseReach = baseArm.Rows.Count(r => r.Item5);
            var headline = navArm.Rows.Select((r, i) => (r, i)).FirstOrDefault(x => x.r.Item5 && !baseArm.Rows[x.i].Item5);
            _out.WriteLine("=== LOOKAHEAD TRAP (each person has a 1-hop SAME-KIND distractor country off the path) ===");
            for (var i = 0; i < navArm.Rows.Count; i++)
            {
                var p = heldOut[i]; var b = baseArm.Rows[i]; var n = navArm.Rows[i];
                var baseKind = b.Item2 == p.City ? "city(1-hop)" : b.Item2 == p.Distractor ? "DISTRACTOR(1-hop)" : "other";
                _out.WriteLine($"    {p.Name,-6} goal={p.Country,-8} distractor={p.Distractor,-8} | base='{b.Item2}' [{baseKind}] | nav='{n.Item2}' {(n.Item5 ? "→GOAL" : "..")}");
            }

            // ── CLEAR-CASE NO-REGRESSION: a single dominant relation ("sol"→"helios") is owned by relation-first BEFORE
            //    the navigator branch — hook on vs off must be byte-identical. ──
            eng.NavigatorDisambiguator = null; var clearOff = await Ask("sol");
            eng.NavigatorDisambiguator = navHook; var clearOn = await Ask("sol");

            _out.WriteLine("=== SUMMARY (HONEST) ===");
            _out.WriteLine($"    COUNTRY (2-hop compose): BASELINE {baseArm.Acc:F1}% ({baseArm.Rows.Count(r => r.Item5)}/{baseArm.Rows.Count}) walks={baseArm.Walks}  |  NAVIGATOR {navArm.Acc:F1}% ({navArm.Rows.Count(r => r.Item5)}/{navArm.Rows.Count}) walks={navArm.Walks}");
            _out.WriteLine($"    INDUSTRY (same anchor, kind=industry): BASELINE {baseInd.Acc:F1}%  |  NAVIGATOR {navInd.Acc:F1}% ({navInd.Rows.Count(r => r.Item5)}/{navInd.Rows.Count})");
            _out.WriteLine($"    COLD navigator-walk emits = {coldWalks}/{coldRows.Count} (0 ⇒ cutover-safe fall-through)");
            _out.WriteLine($"    TRAP (aggregate): navigator reaches the 2-hop GOAL {navReach}/{navArm.Rows.Count}; baseline {baseReach}/{baseArm.Rows.Count} (1-hop ceiling)");
            if (headline.r.Item1 is not null)
                _out.WriteLine($"    TRAP headline: '{headline.r.Item1}' baseline='{baseArm.Rows[headline.i].Item2}'(1-hop) vs navigator='{headline.r.Item2}'(2-hop GOAL)");
            _out.WriteLine($"    CLEAR 'sol': off='{clearOff.Out}'[{clearOff.Path}] on='{clearOn.Out}'[{clearOn.Path}] identical={clearOff.Out == clearOn.Out && clearOff.Path == clearOn.Path}");
            _out.WriteLine($"    VERDICT: navigator {(navArm.Acc > baseArm.Acc ? "BEATS" : navArm.Acc == baseArm.Acc ? "TIES" : "LOSES TO")} one-shot on held-out multi-hop ({navArm.Acc:F1}% vs {baseArm.Acc:F1}%)");

            // Loose floors only — the DATA above is the result:
            Assert.NotNull(navHook);                              // the cutover wired the hook by default
            Assert.Equal(0, coldWalks);                           // COLD navigator falls through (no confident mis-emit)
            Assert.Equal(clearOff.Out, clearOn.Out);              // clear 1-hop case untouched by the hook
            Assert.Equal(clearOff.Path, clearOn.Path);
            Assert.Equal(0, baseReach);                           // the 1-hop baseline NEVER reaches the 2-hop composed goal
            Assert.True(navArm.Acc > baseArm.Acc,                 // the kind-conditioned navigator BEATS one-shot on multi-hop
                $"navigator ({navArm.Acc:F1}%) did not beat one-shot ({baseArm.Acc:F1}%) on held-out 2-hop composition");
            Assert.True(navReach >= 3,                            // the navigator clears the lookahead trap on the majority
                $"navigator reached the 2-hop goal on only {navReach}/{navArm.Rows.Count} held-out trap queries");
            Assert.NotNull(headline.r.Item1);                     // at least one query where nav clears the trap and baseline does not
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
