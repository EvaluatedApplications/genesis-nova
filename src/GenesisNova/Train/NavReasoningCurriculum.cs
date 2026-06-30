using System;
using System.Collections.Generic;
using System.Linq;
using GenesisNova.Cognition.Navigator;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

/// <summary>
/// NAV-REASONING curriculum (M4) — the gym path that actually self-teaches the NAVIGATOR's reasoning and exposes a
/// HELD-OUT generalization curve. It does two things through the NORMAL gym observe path:
///
///   (a) GROWS a multi-hop is-a TAXONOMY (member → genus → domain → root) the navigator can distil into trajectories.
///       The clean ADJACENCY (each child → its IMMEDIATE parent only, degree strictly increasing up) is planted ONCE in
///       the constructor (<see cref="GenesisEvalAppRuntime.PlantNavigatorTaxonomy"/>, adjacency-only, no shortcuts) so
///       <c>SampleNavigatorQueries</c>/<c>ClimbAncestors</c> derive a correct genus→domain→root chain. The same edges are
///       then REINFORCED every cycle by the genus FRAMES below (observe couples child→immediate-parent = the same edge).
///
///   (b) EMITS the level-cue + composition TEXT FRAMES as DATA (varied natural phrasings). The cue stays LEARNED — the
///       trainer's <c>LearnNavLevelCue</c> derives the level from the answer's GRAPH DEPTH above the subject (immediate
///       parent ⇒ Genus, 2-hop ⇒ Domain, top ⇒ Root), NOT from a hardcoded cue-word list. The MARKER words ("kind",
///       "broadly", "ultimately", …) live only in the DATA here; the field relates whichever discriminative marker the
///       frame carries to the level the GRAPH says it is, so a marker NEVER seen in code still resolves at inference.
///       GENUS markers are taught on the (sampled) training members where the answer is the true immediate parent —
///       harmless reinforcement of the adjacency. DOMAIN/ROOT markers are taught on dedicated CUE-TEACH members whose
///       answer is a 2-hop / top ancestor: those frames DO write a child→ancestor shortcut, so the cue-teach members are
///       registered as TRAINING-EXCLUSIONS (their corrupted climb is never sampled into the policy).
///
/// THE M4 SIGNAL: a fixed HELD-OUT member set (one extra leaf per genus) is planted into the graph but registered as
/// held-out, so <see cref="GenesisEvalAppRuntime.EvaluateNavigatorHeldOut"/> measures whether the policy GENERALIZES the
/// walk to members it never trained on. <see cref="SelfAssess"/> runs that eval each cycle and returns its accuracy, so
/// the unit's displayed score IS the held-out curve and the warm-history climbs (or honestly doesn't).
///
/// This is the op-cue/grammar analogue for reasoning: a focused, reliable source of taxonomy + cue frames, instead of
/// hoping the Category muscle incidentally grows a deep enough graph. No hardcoded taxonomy table drives inference — the
/// words are DATA; the navigator learns the walk and the cue learner learns the markers from the graph's own shape.
/// </summary>
public sealed class NavReasoningCurriculum : ITrainingCurriculum
{
    // ── THE TAXONOMY (DATA, not a dispatch table). member roles per genus: [train, train, HELD-OUT, CUE-TEACH]. Real
    //    English words so the tokenizer is happy; their MEANING is irrelevant (it is built distributionally from the
    //    planted edges). Degrees come out monotone: member 1 < genus 5 < domain 6 < root 10 (see the planting below). ──
    private const string Root = "entity";
    private static readonly string[] Misc = { "rock", "water", "cloud", "fire", "dust", "metal", "glass", "stone" };

    private static readonly (string Domain, (string Genus, string[] Members)[] Genera)[] Taxonomy =
    {
        ("creature", new[]
        {
            ("mammal",  new[] { "dog", "cat", "wolf", "otter" }),
            ("bird",    new[] { "robin", "sparrow", "eagle", "finch" }),
            ("fish",    new[] { "salmon", "trout", "tuna", "perch" }),
            ("reptile", new[] { "lizard", "snake", "gecko", "iguana" }),
            ("insect",  new[] { "ant", "bee", "wasp", "beetle" }),
        }),
        ("plant", new[]
        {
            ("tree",   new[] { "oak", "pine", "birch", "maple" }),
            ("flower", new[] { "rose", "tulip", "daisy", "lily" }),
            ("herb",   new[] { "basil", "mint", "sage", "thyme" }),
            ("grass",  new[] { "wheat", "corn", "rye", "oat" }),
            ("vine",   new[] { "grape", "ivy", "hop", "pea" }),
        }),
    };

    // index 0,1 = TRAIN (sampled), 2 = HELD-OUT (eval only), 3 = CUE-TEACH (domain/root marker frames; excluded).
    private const int HeldOutIdx = 2;
    private const int CueTeachIdx = 3;

    // ── MARKER WORDS (DATA). Distinct per level so they don't collide; the LEVEL each implies is decided by the answer's
    //    graph depth in LearnNavLevelCue, NOT by these strings. Many phrasings so the marker — not the framing glue —
    //    accumulates the relation. No code anywhere maps these words to a level. ──
    private static readonly string[] GenusFrames  = { "what kind is {0}", "what type is {0}", "{0} is a sort of", "what is {0} a kind of" };
    private static readonly string[] DomainFrames = { "{0} broadly", "{0} in the wider family of", "what category covers {0}", "{0} more generally" };
    private static readonly string[] RootFrames   = { "{0} ultimately", "{0} at its essence", "{0} fundamentally", "what is {0} at bottom" };

    private readonly Random _rng;
    private readonly int _trainPerCycle;
    private GenesisEvalAppRuntime? _runtime;

    // Derived role lists (filled in BuildRoles).
    private readonly List<(string Member, string Genus, string Domain)> _trainMembers = new();
    private readonly List<(string Member, string Genus, string Domain)> _heldOutMembers = new();
    private readonly List<(string Member, string Genus, string Domain)> _cueMembers = new();
    // GENUS frames are emitted for training AND held-out members: this DEVELOPS the held-out members' meaning-clouds the
    // same way the live gym's is-a frames would, so the held-out eval tests the NAVIGATOR POLICY's generalization (it
    // never distils these members' TRAJECTORIES — they stay out of SampleNavigatorQueries) rather than penalising an
    // impoverished substrate representation. (Genus frames only reinforce the already-planted child→genus adjacency.)
    private readonly List<(string Member, string Genus, string Domain)> _genusFrameMembers = new();
    private readonly List<(string Genus, string Domain)> _genera = new();
    private readonly List<string> _domains = new();
    private readonly List<(string Child, string Parent)> _edges = new();

    public NavReasoningCurriculum(GenesisEvalAppRuntime runtime, int trainPerCycle = 48, int? seed = null)
    {
        _rng = seed is { } s ? new Random(s) : new Random();
        _trainPerCycle = Math.Max(12, trainPerCycle);
        BuildRoles();
        Attach(runtime);
    }

    /// <summary>Test/host seam: build the role lists WITHOUT a runtime (for inspecting the planted set / held-out set).</summary>
    public NavReasoningCurriculum(int trainPerCycle = 48, int? seed = null)
    {
        _rng = seed is { } s ? new Random(s) : new Random();
        _trainPerCycle = Math.Max(12, trainPerCycle);
        BuildRoles();
    }

    private void BuildRoles()
    {
        foreach (var misc in Misc) _edges.Add((misc, Root));
        foreach (var (domain, genera) in Taxonomy)
        {
            _domains.Add(domain);
            _edges.Add((domain, Root));
            foreach (var (genus, members) in genera)
            {
                _genera.Add((genus, domain));
                _edges.Add((genus, domain));
                for (var i = 0; i < members.Length; i++)
                {
                    var m = members[i];
                    _edges.Add((m, genus)); // ADJACENCY ONLY — every member's immediate parent is its genus
                    var tuple = (m, genus, domain);
                    if (i == HeldOutIdx) { _heldOutMembers.Add(tuple); _genusFrameMembers.Add(tuple); }
                    else if (i == CueTeachIdx) _cueMembers.Add(tuple);
                    else { _trainMembers.Add(tuple); _genusFrameMembers.Add(tuple); }
                }
            }
        }
    }

    /// <summary>Plant the clean taxonomy, register the held-out / excluded sets, AND self-teach the level cue from the
    /// frames as DATA — all idempotent and NON-corrupting (direct edges + the cue learner, no distractor-repulsion). Safe
    /// to call at construction AND every cycle (re-asserts the clean is-a structure against any observe-path drift) so the
    /// navigator always samples a correct genus→domain→root chain and the cue stays learned. The cue level is still
    /// derived from the answer's GRAPH DEPTH (no hardcoded cue-word list); only the marker→∘level relation is written.</summary>
    public void Attach(GenesisEvalAppRuntime runtime)
    {
        _runtime = runtime;
        try { runtime.PlantNavigatorTaxonomy(_edges, reinforce: 5); } catch { }
        try { runtime.RegisterNavigatorHeldOut(HeldOutQueries(), _cueMembers.Select(c => c.Member)); } catch { }
        TeachCueFrames(runtime);
    }

    /// <summary>Self-teach the level cue from the curriculum's frames as DATA, via the SAME LearnNavLevelCue the gym
    /// observe path uses (level from graph depth, no word list) — decoupled from the structural-edit corruption.</summary>
    public void TeachCueFrames(GenesisEvalAppRuntime runtime)
    {
        try
        {
            foreach (var (m, g, _) in _genusFrameMembers)
                foreach (var f in GenusFrames) runtime.TeachNavLevelCue(string.Format(f, m), g);   // immediate parent ⇒ GENUS
            foreach (var (g, d) in _genera)
                foreach (var f in GenusFrames) runtime.TeachNavLevelCue(string.Format(f, g), d);    // genus's parent ⇒ GENUS
            foreach (var (m, _, d) in _cueMembers)
                foreach (var f in DomainFrames) runtime.TeachNavLevelCue(string.Format(f, m), d);   // 2-hop ancestor ⇒ DOMAIN
            foreach (var (m, _, _) in _cueMembers)
                foreach (var f in RootFrames) runtime.TeachNavLevelCue(string.Format(f, m), Root);  // top ancestor ⇒ ROOT
        }
        catch { }
    }

    /// <summary>The fixed HELD-OUT query set (member, cue, expected-ancestor) — the M4 generalization probe. One GENUS
    /// (1-hop), one DOMAIN (2-hop) and one ROOT (top) query per held-out member, so the curve reflects the WHOLE
    /// abstraction ladder, not just the easy immediate-kind hop.</summary>
    /// <summary>The held-out members (member, genus, domain) — for a test's per-level diagnostic.</summary>
    public IReadOnlyList<(string Member, string Genus, string Domain)> HeldOutMembersView => _heldOutMembers.ToArray();

    /// <summary>The training members (member, genus, domain) — for a test's per-level diagnostic (the policy's ceiling).</summary>
    public IReadOnlyList<(string Member, string Genus, string Domain)> TrainMembersView => _trainMembers.ToArray();

    public IReadOnlyList<(string Member, NavCue Cue, string Ancestor)> HeldOutQueries()
    {
        var qs = new List<(string, NavCue, string)>();
        foreach (var (m, g, d) in _heldOutMembers)
        {
            qs.Add((m, NavCue.Genus, g));
            qs.Add((m, NavCue.Domain, d));
            qs.Add((m, NavCue.Root, Root));
        }
        return qs;
    }

    public string Name => "nav-reasoning";
    public int Difficulty => 1;          // open-ended foundation unit (the navigator's level lives in the held-out curve)
    public int MasteryDepth => 1;
    public bool IsMastered => _lastHeldOutAcc >= 0.85;
    private double _lastHeldOutAcc;

    /// <summary>One cycle's frames: GENUS markers on training members + genera + domains (answer = immediate parent, so
    /// LearnNavLevelCue reads Genus and the observe coupling only reinforces adjacency), and DOMAIN/ROOT markers on the
    /// excluded cue-teach members (answer = 2-hop / top ancestor). All natural phrasings; the marker is DATA, the level
    /// is the graph's.</summary>
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        var batch = new List<(string Input, string Output)>(_trainPerCycle);
        while (batch.Count < _trainPerCycle)
        {
            // Bias toward the genus frames (they grow + reinforce the sampled structure); sprinkle domain/root markers.
            var roll = _rng.Next(10);
            if (roll < 6 && _genusFrameMembers.Count > 0)
            {
                var (m, g, _) = _genusFrameMembers[_rng.Next(_genusFrameMembers.Count)]; // train ∪ held-out (develops held-out clouds)
                batch.Add((string.Format(GenusFrames[_rng.Next(GenusFrames.Length)], m), g));
            }
            else if (roll < 7 && _genera.Count > 0)
            {
                var (g, d) = _genera[_rng.Next(_genera.Count)];
                batch.Add((string.Format(GenusFrames[_rng.Next(GenusFrames.Length)], g), d)); // genus's immediate parent = domain
            }
            else if (roll < 8 && _domains.Count > 0)
            {
                var d = _domains[_rng.Next(_domains.Count)];
                batch.Add((string.Format(GenusFrames[_rng.Next(GenusFrames.Length)], d), Root)); // domain's immediate parent = root
            }
            else if (roll < 9 && _cueMembers.Count > 0)
            {
                var (m, _, d) = _cueMembers[_rng.Next(_cueMembers.Count)];
                batch.Add((string.Format(DomainFrames[_rng.Next(DomainFrames.Length)], m), d)); // 2-hop ancestor ⇒ DOMAIN
            }
            else if (_cueMembers.Count > 0)
            {
                var (m, _, _) = _cueMembers[_rng.Next(_cueMembers.Count)];
                batch.Add((string.Format(RootFrames[_rng.Next(RootFrames.Length)], m), Root)); // top ancestor ⇒ ROOT
            }
            else break;
        }
        return batch;
    }

    /// <summary>No surface probes — this unit self-assesses by the HELD-OUT navigator curve (a property of the space +
    /// trained policy, like the prebake's function-word separation). Returning empty makes the grader use SelfAssess.</summary>
    public IReadOnlyList<TrainingProbe> NextProbes() => Array.Empty<TrainingProbe>();

    public void RecordCycle(CycleGrade grade) { /* level is open-ended; the held-out curve is the progress signal */ }

    /// <summary>Run the HELD-OUT navigator eval on the LIVE trained policy and return its ACCURACY (fraction landing on
    /// the cued ancestor) as this unit's cycle score — so the gym's per-lesson view shows the M4 curve climbing. Side
    /// effect: appends a point to the runtime's NavHeldOutHistory (the warm-history the Inspect tab / a log line reads).
    /// Cheap, gated, GPU-OOM-safe (read-only walk of the resting net).</summary>
    public double? SelfAssess(GenesisEvalAppRuntime runtime)
    {
        try
        {
            // Re-assert the clean is-a structure + cue EACH cycle (cheap, gated, no repulsion) so the gym's own
            // observe-path edits (distractor repulsion / cloud drift) can't erode the taxonomy the navigator samples.
            try { runtime.PlantNavigatorTaxonomy(_edges, reinforce: 2); } catch { }
            TeachCueFrames(runtime);
            var p = runtime.EvaluateNavigatorHeldOut();
            if (p.Count == 0) return null; // not registered yet → fall back to (empty) probe grading
            _lastHeldOutAcc = p.AccuracyPct;
            return p.AccuracyPct;
        }
        catch { return null; }
    }
}
