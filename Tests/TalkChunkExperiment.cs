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

    [Fact]
    public void Persona_SeededChunks_TalkInCharacter_AtProductionDims()
    {
        // THE GYM DESIGN, isolated: in the conscious field the GRU DECODER is bypassed — a persona is retrieved as a
        // CHUNK by TryFieldRespond, never decoded token-by-token. So the gym does NOT decode-train the persona (that
        // builds stray cue→WORD edges that crowd the chunk out of the top-N neighbours — measured); it SEEDS the
        // reply chunks (whole-reply relations) once. This proves seeded chunks alone talk in-character, and that the
        // chunk-PREFERENCE in TryFieldRespond holds at production dims.
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var persona = new PersonalityCurriculum(trainPerCycle: 600);

        // SEED the persona's chunk repertoire (cue → WHOLE reply), exactly as the gym does at start.
        foreach (var (cue, reply) in persona.Repertoire)
            space.FineEditFromExample(new[] { cue }, new[] { reply }, isNegativeExample: false);

        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true, TalkEnabled = true };
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
        _out.WriteLine($"seeded-chunk persona in-character: {rate:P0} ({hits}/{probes.Count})");
        Assert.True(rate >= 0.7, $"seeded chunks talk in-character; {rate:P0}");
    }

    [Fact]
    public void Persona_GeneralizesRudeness_ToUnseenCues()
    {
        var config = new GenesisNovaConfig(HiddenSize: ProductionDims.HiddenSize);
        var tok = new WhitespaceGenesisTokenizer();
        var space = new DialecticalSpace(config.FaceDimension, seed: 7);
        var persona = new PersonalityCurriculum(trainPerCycle: 400);
        var personaReplies = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (cue, reply) in persona.NextTrainBatch())
        {
            space.FineEditFromExample(new[] { cue }, new[] { reply }, isNegativeExample: false);
            personaReplies.Add(reply);
        }

        var mind = new GenesisInferenceEngine(tok, new GenesisNeuralModel(config), space, null) { ConsciousField = true, TalkEnabled = true };

        // WARM UP the self — the mind says rude things to seen cues, so its SELF becomes the asshole (it folds its own
        // replies in). A personality is who you've BEEN, not a table.
        for (var round = 0; round < 4; round++)
            foreach (var cue in new[] { "hello", "thanks", "help", "bye", "sorry", "hi", "you good" })
                mind.Generate(new GenerationRequest(cue, 12));

        // UNSEEN inputs — NEVER in the persona's cue list (and not copula assertions). A lookup abstains; a personality
        // stays in character. In-character = it said something the persona would say (a known rude line or a rude marker).
        bool InCharacter(string s) => personaReplies.Contains(s) || PersonalityCurriculum.RudeMarkers.Any(m => s.Contains(m, StringComparison.OrdinalIgnoreCase));
        var unseen = new[] { "can you do my taxes", "do you know python", "lets be friends", "tell me a joke", "make me a sandwich", "give me directions" };
        var rude = 0; var distinct = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var u in unseen)
        {
            var r = mind.Generate(new GenerationRequest(u, 12));
            var got = r.Output?.Trim() ?? "";
            var inChar = InCharacter(got);
            if (inChar) rude++;
            distinct.Add(got);
            _out.WriteLine($"  '{u}' -> '{got}' [{r.DecisionPath}] {(inChar ? "in-character" : "")}");
        }
        _out.WriteLine($"in-character on {rude}/{unseen.Length} UNSEEN inputs, {distinct.Count} distinct replies");
        Assert.True(rude >= unseen.Length - 1, $"the asshole stays an asshole to unseen input; {rude}/{unseen.Length}");
        Assert.True(distinct.Count >= 3, $"it varies its insults, not one looped line; {distinct.Count} distinct");
    }
}
