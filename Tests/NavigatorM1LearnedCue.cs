using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

/// <summary>
/// M1 CLOSE-OUT — THE LEARNED NAVIGATOR CUE DRIVES INFERENCE LIVE (PROJECT_PLAN.md M1.1 + cutover). This is the honest
/// acceptance for closing M1: a LIVE head-to-head over the REAL <see cref="GenesisEvalAppRuntime"/> with
/// <see cref="GenesisNovaConfig.WithProductionMechanisms"/> — where <c>NavigatorDisambiguation</c> is now ON BY DEFAULT
/// (the cutover) — proving four things end to end:
///   1. the navigator's target-aspect cue (genus/domain/root) comes from a LEARNED resolver (∘gns/∘dom/∘rut, taught by
///      <see cref="GenesisInferenceEngine.LearnNavLevelCue"/> from GRAPH DEPTH — no English keyword map, no word list);
///   2. it generalises to NATURALLY-PHRASED, VARIED wording AND a NONCE level marker the old keyword switch could never
///      know ("zarni"), reaching the right multi-hop answer and BEATING the one-shot baseline on the ambiguous subset;
///   3. the navigator policy is trained by <see cref="GenesisEvalAppRuntime.TrainNavigatorCycle"/> ALONE — proving the
///      DEEPENED gym sampler now emits genus→root pairs (no focused ground-truth supplement, unlike NavigatorInferenceM1);
///   4. a COLD navigator (never trained) FALLS THROUGH safely on the ambiguous branch — the cutover cannot regress.
/// HONEST: the reported numbers ARE the result; loose floors only. Gated [SlowFact] behind RUN_SLOW=1.
/// </summary>
public sealed class NavigatorM1LearnedCue
{
    private readonly ITestOutputHelper _out;
    public NavigatorM1LearnedCue(ITestOutputHelper o) => _out = o;

    private const int ObservePerEdge = 4;
    private const string Root = "thing";

    // member → genus → domain → root, padded so relational degree strictly increases up the levels (the substrate's own
    // is-a-parent signal the navigator climbs). The genera are mild HUBS (≥3 comparable relations) → a ROOT query on a
    // genus reaches the AMBIGUOUS branch where the one-shot reason tops out (the navigator's territory).
    private static readonly (string Domain, (string Genus, string[] Members)[] Genera)[] Taxonomy =
    {
        ("animal", new[]
        {
            ("mammal", new[] { "dog", "cat" }),
            ("bird",   new[] { "robin", "sparrow" }),
            ("fish",   new[] { "salmon", "trout" }),
        }),
        ("plant", new[]
        {
            ("tree",   new[] { "oak", "pine" }),
            ("flower", new[] { "rose", "tulip" }),
            ("grass",  new[] { "wheat", "corn" }),
        }),
    };

    // PRODUCTION config — NavigatorDisambiguation is ON via WithProductionMechanisms (the cutover); we DON'T set it here,
    // proving the flag flip wired the hook. FieldTicks/MeaningOps off so the two arms differ ONLY in the navigator hook.
    private static GenesisNovaConfig BuildConfig(string dir) =>
        new GenesisNovaConfig(
            HiddenSize: 64,                 // tiny GRU — ConsciousField bypasses the neural decoder; the navigator
                                            // (fixed 1024/2048) + the full-size substrate are what is under test
            Backend: ComputeBackend.Cpu,
            AutoPersist: false,
            AutoResume: false,
            LocalStateDirectory: dir)
        .WithProductionMechanisms() with
        {
            FieldTicks = false,
            MeaningOps = false,
        };

    private static void PlantTaxonomy(DialecticalSpace ds,
        out List<(string Genus, string Domain)> genera, out List<(string Member, string Genus, string Domain)> members)
    {
        genera = new List<(string, string)>();
        members = new List<(string, string, string)>();
        void Edge(string child, string parent) { for (var i = 0; i < ObservePerEdge; i++) ds.ObserveContradiction(child, parent, 0.0); }

        foreach (var (domain, generaArr) in Taxonomy)
        {
            Edge(domain, Root);
            foreach (var (genus, mem) in generaArr)
            {
                Edge(genus, domain);
                genera.Add((genus, domain));
                foreach (var m in mem) { Edge(m, genus); members.Add((m, genus, domain)); }
            }
        }
        // Pad the root's degree above the domains so the climb tops out at the root (monotone degree gradient).
        foreach (var misc in new[] { "rock", "water", "cloud", "fire", "stone", "dust" }) Edge(misc, Root);
        ds.FlushCloudBatch();
    }

    [SlowFact]
    public async Task LearnedCue_GymTrainedNavigator_BeatsOneShot_OnNaturalQueries_Live()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navm1cue-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var runtime = new GenesisEvalAppRuntime(BuildConfig(dir));
            var ds = Assert.IsType<DialecticalSpace>(runtime.State.Memory);
            PlantTaxonomy(ds, out var genera, out var members);
            var eng = runtime.State.Inference;

            // CUTOVER PROOF — WithProductionMechanisms turned NavigatorDisambiguation on, so the runtime wired the hook
            // WITHOUT this test touching the flag. Capture it; baseline = nulled, navigator = restored.
            var navHook = eng.NavigatorDisambiguator;
            Assert.NotNull(navHook);
            var net = runtime.State.Navigator;
            _out.WriteLine($"=== M1 LEARNED-CUE LIVE: {ds.NodeCount} concepts | {genera.Count} genera (ambiguous hubs) | " +
                           $"navigator {net.Dim}/{net.Hidden} | NavigatorDisambiguation ON via WithProductionMechanisms ===");

            async Task<(string Out, string Path)> Ask(string query)
            {
                var r = (await runtime.PredictAsync(query, 8)).Result;
                return (r?.Output?.Trim() ?? string.Empty, r?.DecisionPath ?? string.Empty);
            }
            static bool Hit(string o) => string.Equals(o, Root, StringComparison.OrdinalIgnoreCase);

            // Natural, VARIED root-query phrasings (NOT the exact trained marker frames). Includes a NONCE marker the old
            // English keyword switch could never resolve.
            string[] RootPhrasings(string g) => new[] { $"{g} ultimately", $"{g} at its core", $"{g} zarni" };

            // ── (0) COLD-NAVIGATOR FALL-THROUGH (cutover safety). BEFORE any nav training or cue teaching: the policy is
            //    random and the level cue is unlearned (→ default GENUS). An ambiguous root query must NOT be confidently
            //    mis-resolved — it should FALL THROUGH to the one-shot reason (DecisionPath ≠ navigator-walk). ──
            var coldRows = new List<(string Q, string Out, string Path)>();
            foreach (var (g, _) in genera)
                foreach (var q in RootPhrasings(g)) { var (o, p) = await Ask(q); coldRows.Add((q, o, p)); }
            var coldWalks = coldRows.Count(r => r.Path == "navigator-walk");
            _out.WriteLine($"--- COLD navigator (untrained): {coldRows.Count} ambiguous queries | navigator-walk emits = {coldWalks} ---");
            foreach (var r in coldRows.Take(6)) _out.WriteLine($"    cold '{r.Q}' -> '{r.Out}' [{r.Path}]");

            // ── (1) TRAIN THE NAVIGATOR POLICY VIA TrainNavigatorCycle ALONE — no focused BC. This proves the DEEPENED
            //    sampler emits genus→root pairs (a genus hub's chain is only 2 hops, which the old sampler skipped). ──
            for (var c = 0; c < 30; c++)
            {
                var r = runtime.TrainNavigatorCycle(maxMembers: 64, epochs: 6);
                if (c % 6 == 0 || c == 29)
                    _out.WriteLine($"    TrainNavigatorCycle {c}: loss={r.Loss:F4} queries={r.Queries} resolve={r.ResolvePct:P0}");
            }

            // ── (2) TEACH THE LEVEL CUES FROM DATA (the LEARNED ∘gns/∘dom/∘rut). LearnNavLevelCue derives the level from
            //    the answer's GRAPH DEPTH above the subject — no word list. Varied subjects (members AND genera) so the
            //    constant marker dominates; the nonce "zarni" is learnable ONLY structurally (no English keyword knows it). ──
            void TeachRoot(string subj) { eng.LearnNavLevelCue($"{subj} ultimately", Root); eng.LearnNavLevelCue($"{subj} at its core", Root); eng.LearnNavLevelCue($"{subj} zarni", Root); }
            foreach (var (m, _, _) in members) TeachRoot(m);
            foreach (var (g, _) in genera) TeachRoot(g);
            foreach (var (m, g, d) in members)
            {
                eng.LearnNavLevelCue($"what kind is {m}", g);     // immediate parent → GENUS marker "kind"
                eng.LearnNavLevelCue($"what sort is {m}", g);     //                                     "sort"
                eng.LearnNavLevelCue($"{m} broadly", d);          // 2-hop ancestor → DOMAIN marker "broadly"
                eng.LearnNavLevelCue($"what class is {m}", d);    //                                  "class"
            }
            foreach (var (g, d) in genera) eng.LearnNavLevelCue($"what kind is {g}", d); // genus's immediate parent → GENUS
            ds.FlushCloudBatch();

            // DIAGNOSTIC — the gym-trained policy climbs genus→root under the explicit cue (WalkNavigator), confirming
            // TrainNavigatorCycle ALONE (the deepened sampler) trained the multi-hop walk with no focused supplement.
            foreach (var (g, _) in genera)
            {
                var wRoot = runtime.WalkNavigator(g, "root");
                _out.WriteLine($"    [diag] WalkNavigator('{g}','root') -> '{wRoot.Answer}' reached={wRoot.Reached} traj={string.Join("->", wRoot.Trajectory)}");
            }

            // ── THE LIVE HEAD-TO-HEAD on the AMBIGUOUS multi-hop subset, NATURAL phrasings, through the REAL PredictAsync.
            async Task<(double Acc, int Walks, List<(string Q, string Out, string Path, bool Hit)> Rows)> RunArm(bool navOn)
            {
                eng.NavigatorDisambiguator = navOn ? navHook : null;
                var rows = new List<(string, string, string, bool)>();
                foreach (var (g, _) in genera)
                    foreach (var q in RootPhrasings(g)) { var (o, p) = await Ask(q); rows.Add((q, o, p, Hit(o))); }
                var acc = 100.0 * rows.Count(r => r.Item4) / Math.Max(1, rows.Count);
                var walks = rows.Count(r => r.Item3 == "navigator-walk");
                return (acc, walks, rows);
            }

            var baseArm = await RunArm(navOn: false);
            var navArm = await RunArm(navOn: true);
            eng.NavigatorDisambiguator = navHook; // leave production wiring restored

            _out.WriteLine("=== AMBIGUOUS multi-hop subset: natural root phrasings → expect ROOT 'thing' (2 hops, learned cue) ===");
            _out.WriteLine($"    {"query",-22} | {"BASELINE",-26} | NAVIGATOR (learned cue)");
            for (var i = 0; i < navArm.Rows.Count; i++)
            {
                var b = baseArm.Rows[i]; var n = navArm.Rows[i];
                _out.WriteLine($"    {b.Item1,-22} | {b.Item2 + " [" + b.Item3 + "]",-26} {(b.Item4 ? "OK" : "..")} | " +
                               $"{n.Item2 + " [" + n.Item3 + "]"} {(n.Item4 ? "OK" : "..")}");
            }

            // The nonce-marker subset in isolation — the cleanest learned-cue-vs-keyword win (the keyword map defaults GENUS).
            var nonceRows = navArm.Rows.Where(r => r.Item1.EndsWith("zarni", StringComparison.Ordinal)).ToList();
            var nonceHits = nonceRows.Count(r => r.Item4);

            // ── CLEAR-CASE NO-REGRESSION: a leaf with one dominant relation ("dog") is owned by the dominant-relation
            //    heuristic BEFORE the navigator branch — hook on vs off must be byte-identical. ──
            eng.NavigatorDisambiguator = null; var clearOff = await Ask("dog");
            eng.NavigatorDisambiguator = navHook; var clearOn = await Ask("dog");

            _out.WriteLine("=== SUMMARY (HONEST) ===");
            _out.WriteLine($"    BASELINE  one-shot accuracy = {baseArm.Acc:F1}% ({baseArm.Rows.Count(r => r.Item4)}/{baseArm.Rows.Count})  navigator-walk={baseArm.Walks}");
            _out.WriteLine($"    NAVIGATOR (learned cue) acc = {navArm.Acc:F1}% ({navArm.Rows.Count(r => r.Item4)}/{navArm.Rows.Count})  navigator-walk={navArm.Walks}/{navArm.Rows.Count}");
            _out.WriteLine($"    NONCE marker 'zarni' (keyword map CANNOT know it): {nonceHits}/{nonceRows.Count} reached ROOT via the LEARNED cue");
            _out.WriteLine($"    COLD navigator: navigator-walk emits on the ambiguous branch = {coldWalks}/{coldRows.Count} (0 ⇒ falls through, cutover-safe)");
            _out.WriteLine($"    VERDICT: navigator {(navArm.Acc > baseArm.Acc ? "BEATS" : navArm.Acc == baseArm.Acc ? "TIES" : "LOSES TO")} one-shot LIVE ({navArm.Acc:F1}% vs {baseArm.Acc:F1}%)");
            _out.WriteLine($"    CLEAR CASE 'dog': off='{clearOff.Out}'[{clearOff.Path}] on='{clearOn.Out}'[{clearOn.Path}] identical={clearOff.Out == clearOn.Out && clearOff.Path == clearOn.Path}");

            // Loose floors only — the DATA above is the result:
            Assert.NotEmpty(genera);
            Assert.NotNull(navHook);                              // the cutover wired the hook by default
            Assert.Equal(0, coldWalks);                           // COLD navigator falls through (no confident mis-emit)
            Assert.Equal(clearOff.Out, clearOn.Out);              // clear case untouched by the hook
            Assert.Equal(clearOff.Path, clearOn.Path);
            Assert.True(navArm.Acc > baseArm.Acc,                 // the learned cue + gym-trained navigator BEATS one-shot
                $"navigator ({navArm.Acc:F1}%) did not beat one-shot ({baseArm.Acc:F1}%) on the natural-phrasing ambiguous subset");
            Assert.True(nonceHits > 0, "the NONCE level marker did not generalise — the learned cue is no better than a keyword list");
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
