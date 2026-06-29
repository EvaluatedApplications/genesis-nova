using System;
using System.IO;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Persistence;
using GenesisNova.Runtime;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace GenesisNova.Tests;

/// <summary>
/// THE NAVIGATOR + PERSISTENT-SELF CHECKPOINT ROUND-TRIP (foundation for overnight gym training). The shared
/// <see cref="GenesisRuntimeState.Navigator"/> policy net and the engine's persistent <c>SelfField</c> are written into
/// the SAME sharded checkpoint as the model NN (navigator weights as one concatenated f64 shard; self in meta JSON) and
/// restored on load via <see cref="GenesisRuntimeState.Replace"/>. These tests pin: (1) a trained navigator + a known
/// self survive a save→reload BIT-EXACT, and (2) an OLD checkpoint without any navigator data still loads — the
/// navigator stays freshly initialised and the engine resumes self-less, no exception. If this breaks, an overnight run
/// silently resets its trained navigator/self on the next reload, so it is asserted hard.
///
/// Production dims (FaceDimension 1024 ⇒ self length 608, navigator hidden 2048) — pure save/load, no training, so a
/// plain [Fact]. The model NN is kept tiny (HiddenSize 64) since only the navigator + self are under test here.
/// </summary>
public sealed class NavigatorPersistenceTests
{
    private static GenesisNovaConfig Config(string dir) => new(
        HiddenSize: 64,
        Backend: ComputeBackend.Cpu,
        AutoPersist: false,
        AutoResume: false,
        LocalStateDirectory: dir);
    // FaceDimensionOverride defaults to the production 1024, so SemanticLength == 608 (the self width).

    [Fact]
    public void NavigatorWeightsAndSelf_SurviveSaveReload_BitExact()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navpersist-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "checkpoint.json");
        var cfg = Config(dir);
        try
        {
            var s1 = new GenesisRuntimeState(cfg);

            // The self width the engine lives in == the navigator's W_s input width == 608 at production dim 1024.
            var selfLen = FaceCodec.SemanticLength(cfg.FaceDimension);
            Assert.Equal(608, selfLen);
            Assert.Equal(selfLen, s1.Navigator.SelfLength);

            // TRAIN the navigator (deterministically perturb every weight so it differs from any fresh random net),
            // and set a KNOWN persistent self — the two things the checkpoint must carry.
            using (no_grad())
                foreach (var p in s1.Navigator.parameters())
                    p.add_(ones_like(p));

            var knownSelf = new double[selfLen];
            for (var i = 0; i < selfLen; i++) knownSelf[i] = Math.Sin(i * 0.013) * 0.1 - 0.02;
            s1.Inference.RestoreSelfField(knownSelf);

            var navBefore = s1.Navigator.ExportWeights();
            var selfBefore = s1.Inference.SelfField.ToArray();
            Assert.Equal(selfLen, selfBefore.Length);
            Assert.True(navBefore.Parameters.Length > 0, "navigator must export its parameter tensors");

            // SAVE the navigator + self into the checkpoint (exactly what the persister does).
            GenesisCheckpointStore.Save(
                path, cfg, s1.Tokenizer, s1.Model,
                platonicSpace: s1.Memory.ExportSnapshot(),
                navigator: navBefore,
                navigatorSelfField: selfBefore);

            // RELOAD from the same path.
            var loaded = GenesisCheckpointStore.LoadForRuntime(path, cfg);
            Assert.NotNull(loaded.Navigator);
            Assert.NotNull(loaded.NavigatorSelfField);
            Assert.Equal(selfLen, loaded.NavigatorSelfField!.Length);

            var s2 = new GenesisRuntimeState(cfg);
            s2.Replace(
                loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation,
                loaded.TrainerLearningStateJson, loaded.GrammarRoles, loaded.Navigator, loaded.NavigatorSelfField);

            // ── ASSERT weights identical (bit-exact: f32 weights round-trip f32→f64→shard→f64→f32 losslessly). ──
            var navAfter = s2.Navigator.ExportWeights();
            Assert.Equal(navBefore.Dim, navAfter.Dim);
            Assert.Equal(navBefore.Hidden, navAfter.Hidden);
            Assert.Equal(navBefore.CueCount, navAfter.CueCount);
            Assert.Equal(navBefore.SelfLength, navAfter.SelfLength);
            Assert.Equal(navBefore.Parameters.Length, navAfter.Parameters.Length);
            for (var i = 0; i < navBefore.Parameters.Length; i++)
            {
                var a = navBefore.Parameters[i];
                var b = navAfter.Parameters[i];
                Assert.Equal(a.Name, b.Name);
                Assert.Equal(a.Shape, b.Shape);
                Assert.Equal(a.Values.Length, b.Values.Length);
                Assert.True(a.Values.AsSpan().SequenceEqual(b.Values), $"weights for {a.Name} must round-trip exactly");
            }

            // ── ASSERT self identical (elementwise exact). ──
            var selfAfter = s2.Inference.SelfField.ToArray();
            Assert.Equal(knownSelf.Length, selfAfter.Length);
            for (var i = 0; i < knownSelf.Length; i++)
                Assert.Equal(knownSelf[i], selfAfter[i]);

            // ── ASSERT a forward pass on a FIXED input is identical (weights truly loaded, not just the export blob). ──
            var anchorArr = new float[cfg.FaceDimension];
            for (var i = 0; i < anchorArr.Length; i++) anchorArr[i] = (float)Math.Cos(i * 0.001);
            var selfF = knownSelf.Select(v => (float)v).ToArray();
            using (no_grad())
            {
                s1.Navigator.eval();
                s2.Navigator.eval();
                using var anchorT = tensor(anchorArr, new long[] { 1, cfg.FaceDimension });
                using var cueT = tensor(new long[] { 0 }, new long[] { 1 });
                using var selfT = tensor(selfF, new long[] { 1, selfLen });
                using var h1 = s1.Navigator.SeedHidden(anchorT, cueT, selfT);
                using var h2 = s2.Navigator.SeedHidden(anchorT, cueT, selfT);
                var v1 = h1.to(float64).cpu().data<double>().ToArray();
                var v2 = h2.to(float64).cpu().data<double>().ToArray();
                Assert.True(v1.AsSpan().SequenceEqual(v2), "seeded hidden state must be identical after reload");
            }

            s1.Navigator.Dispose();
            s2.Navigator.Dispose();
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }

    [Fact]
    public void OldCheckpoint_WithoutNavigatorData_LoadsFresh_NoThrow()
    {
        var dir = Path.Combine(Path.GetTempPath(), "gn-navpersist-old-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "checkpoint.json");
        var cfg = Config(dir);
        try
        {
            var s1 = new GenesisRuntimeState(cfg);

            // Save WITHOUT navigator / self — exactly what a pre-navigator (≤ v6) checkpoint looks like (the sections
            // are simply absent).
            GenesisCheckpointStore.Save(
                path, cfg, s1.Tokenizer, s1.Model,
                platonicSpace: s1.Memory.ExportSnapshot());

            var loaded = GenesisCheckpointStore.LoadForRuntime(path, cfg);
            Assert.Null(loaded.Navigator);
            Assert.Null(loaded.NavigatorSelfField);

            // Replace must NOT throw; the navigator stays freshly initialised and the engine resumes self-less.
            var s2 = new GenesisRuntimeState(cfg);
            var ex = Record.Exception(() => s2.Replace(
                loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation,
                loaded.TrainerLearningStateJson, loaded.GrammarRoles, loaded.Navigator, loaded.NavigatorSelfField));
            Assert.Null(ex);

            Assert.NotNull(s2.Navigator);
            Assert.Equal(cfg.FaceDimension, s2.Navigator.Dim);
            Assert.Empty(s2.Inference.SelfField); // self-less (the fresh, pre-first-perception state)

            s1.Navigator.Dispose();
            s2.Navigator.Dispose();
        }
        finally { try { Directory.Delete(dir, true); } catch { } }
    }
}
