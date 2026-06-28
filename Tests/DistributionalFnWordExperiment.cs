using System;
using System.Linq;
using GenesisNova.Cognition.Platonic;
using Xunit;
using Xunit.Abstractions;

namespace GenesisNova.Tests;

// EXPERIMENT (theory #2): function-likeness as a DISTRIBUTIONAL property — PMI computed from the co-occurrence COUNTS we
// already keep — vs the graph-clustering metric that tops out (~36% on the live model; possessives stuck). A function word
// co-occurs with everything BY FREQUENCY (weak associations, PMI≈0); content co-occurs SELECTIVELY with its cluster (strong
// PMI). PMI is degree-robust, where the graph-clustering estimator gets noisy on LOW-degree function words (possessives).
// This pits both metrics on the SAME warmed space and measures whether PMI separates the words the graph misses.
public sealed class DistributionalFnWordExperiment
{
    private readonly ITestOutputHelper _out;
    public DistributionalFnWordExperiment(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Pmi_separates_function_words_the_graph_metric_misses()
    {
        var space = new DialecticalSpace(faceDimension: 64);
        string[][] clusters =
        {
            new[]{"cat","dog","cow","pig","hen","fox","owl","bat","ant","elk"},
            new[]{"red","blue","green","pink","gray","black","white","brown","gold","teal"},
            new[]{"rome","paris","tokyo","cairo","lima","oslo","delhi","perth","kyoto","nice"},
            new[]{"bread","cake","soup","rice","meat","fish","egg","jam","ham","pie"},
            new[]{"car","bus","van","ship","bike","jet","cab","tram","boat","sled"},
        };
        var content = clusters.SelectMany(c => c).ToArray();
        string[] highFn = { "the", "of" };     // co-occur with EVERYTHING (high degree) — both metrics should get these
        string[] possLike = { "my", "your" };  // co-occur with a SMALL diverse subset (low degree) — the case the graph misses
        var rng = new Random(7);
        var possSub = possLike.ToDictionary(p => p, _ => content.OrderBy(_ => rng.Next()).Take(12).ToArray()); // each ~12 diverse words

        for (var step = 0; step < 16000; step++)
        {
            var cl = clusters[rng.Next(clusters.Length)];
            var a = cl[rng.Next(cl.Length)]; var b = cl[rng.Next(cl.Length)];
            if (a != b) space.ObserveContradiction(a, b, 0.1);                                  // content ↔ its cluster (SELECTIVE → high PMI)
            space.ObserveContradiction(highFn[rng.Next(highFn.Length)], content[rng.Next(content.Length)], 0.2); // high-freq fn ↔ ANY content
            foreach (var p in possLike) space.ObserveContradiction(p, possSub[p][rng.Next(possSub[p].Length)], 0.2); // possessive ↔ its small subset
        }

        bool GraphFn(string w) => space.IsFunctionLike(w);
        space.DistributionalFnWord = true;
        bool PmiFn(string w) => space.IsFunctionLike(w);

        space.DistributionalFnWord = false;
        _out.WriteLine("word      graph-coh  graphFn   assoc   pmiFn");
        foreach (var w in highFn.Concat(possLike).Concat(new[] { "cat", "red", "bread", "car" }))
        {
            var coh = space.FunctionStats(w).Centrality; var assoc = space.AssociationStrength(w);
            _out.WriteLine($"{w,-9} {coh,8:F2}  {GraphFn(w),-7}  {assoc,6:F2}  {PmiFn(w)}");
        }

        var fnWords = highFn.Concat(possLike).ToArray();
        var graphSep = fnWords.Count(GraphFn) / (double)fnWords.Length - content.Count(GraphFn) / (double)content.Length;
        var pmiSep   = fnWords.Count(PmiFn)   / (double)fnWords.Length - content.Count(PmiFn)   / (double)content.Length;
        _out.WriteLine($"\nSEPARATION  graph {graphSep:P0}  |  PMI {pmiSep:P0}   (fn flagged: graph {fnWords.Count(GraphFn)}/{fnWords.Length}, PMI {fnWords.Count(PmiFn)}/{fnWords.Length})");

        Assert.True(content.Count(PmiFn) == 0, "PMI wrongly flagged content as function");
        Assert.True(pmiSep >= graphSep, $"PMI ({pmiSep:P0}) did not beat the graph metric ({graphSep:P0})");
        Assert.True(PmiFn("my") && PmiFn("your"), "PMI failed to flag the low-degree possessives (the case the graph misses)");
    }
}
