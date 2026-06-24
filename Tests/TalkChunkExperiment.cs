using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using GenesisNova.Train;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (make-or-break for talk-by-chunk): can the conscious field RETRIEVE a multi-word reply CHUNK as a
// response? The word face stores multi-token strings as composite concepts; if a cue can relate to a reply chunk and
// the field retrieves it whole, then "talking" = NN-directed retrieval/sequencing of reusable chunks is on the table.
// If it can't, the chunk-retrieval itself is the first thing to fix. No training — just relations + retrieval.
public sealed class TalkChunkExperiment
{
    private readonly ITestOutputHelper _out;
    public TalkChunkExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Field_RetrievesAMultiWordChunk_AsAResponse()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var model = new GenesisNeuralModel(config);
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        void Rel(string cue, string reply) { for (var i = 0; i < 4; i++) space.FineEditFromExample(new[] { cue }, new[] { reply }, false); }

        // A rude persona's repertoire — cue → a reply CHUNK (multi-word). The chunk becomes a composite concept.
        var repertoire = new[]
        {
            ("bye", "get lost"), ("hello", "what now"), ("thanks", "whatever"),
            ("help", "figure it out"), ("sorry", "too late"), ("why", "who cares"),
        };
        foreach (var (cue, reply) in repertoire) Rel(cue, reply);

        var mind = new GenesisInferenceEngine(tok, model, space, null) { ConsciousField = true };
        var hits = 0;
        foreach (var (cue, reply) in repertoire)
        {
            var r = mind.Generate(new GenerationRequest(cue, 8));
            var got = r.Output?.Trim() ?? "";
            var hit = got.Equals(reply, StringComparison.OrdinalIgnoreCase) || got.Contains(reply.Split(' ')[^1], StringComparison.OrdinalIgnoreCase);
            if (hit) hits++;
            _out.WriteLine($"  '{cue}' -> '{got}' [{r.DecisionPath}]  (want '{reply}') {(hit ? "OK" : "")}");
        }
        _out.WriteLine($"chunk replies retrieved: {hits}/{repertoire.Length}");
        Assert.True(hits >= repertoire.Length - 1, $"the field must retrieve reply chunks; {hits}/{repertoire.Length}");
    }

    [Fact]
    public void Field_TalksTheRudePersona_FromTheCurriculum()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);

        // SEED the persona's repertoire from the actual curriculum (cue -> reply, fanned out — each cue relates to
        // its intent's many replies). This is the chunk repertoire the field will draw from.
        var persona = new PersonalityCurriculum(trainPerCycle: 400);
        foreach (var (cue, reply) in persona.NextTrainBatch())
            space.FineEditFromExample(new[] { cue }, new[] { reply }, isNegativeExample: false);

        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true, TalkEnabled = true };

        // CONVERSE: feed each probe cue, grade with the curriculum's own grader (any rude marker / valid reply = a
        // hit; polite words are the competing vocabulary). This is "does it talk IN CHARACTER".
        var probes = persona.NextProbes();
        var hits = 0; var shown = 0;
        foreach (var p in probes)
        {
            var res = mind.Generate(new GenerationRequest(p.Query, 12));
            var q = GenesisGrader.Quality(res.Output ?? string.Empty, p.Allowed, p.RequiredDepth,
                res.UsedNeuralFallback, requirePlatonic: false, p.AnswerVocabulary, p.SurfaceStrict);
            if (q >= 0.5) hits++;
            if (shown++ < 8) _out.WriteLine($"  '{p.Query}' -> '{res.Output?.Trim()}' [{res.DecisionPath}] q={q:F2}");
        }
        var rate = hits / (double)probes.Count;
        _out.WriteLine($"in-character rate: {rate:P0} ({hits}/{probes.Count})");
        Assert.True(rate >= 0.7, $"the field must answer in the persona's voice; {rate:P0}");
    }
}
