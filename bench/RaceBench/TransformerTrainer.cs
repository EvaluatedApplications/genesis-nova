using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace RaceBench;

/// <summary>
/// Best-effort training/eval wrapper around <see cref="TinyTransformer"/> for "input → answer" tasks.
/// Trains next-token cross-entropy with the loss MASKED TO THE ANSWER SPAN (standard instruction tuning —
/// this HELPS the baseline focus on producing the answer), Adam optimizer, greedy decode for eval. Uses the
/// SAME nova tokenizer instance, so both competitors see an identical token stream.
/// </summary>
public sealed class TransformerTrainer
{
    private readonly TinyTransformer _net;
    private readonly torch.optim.Optimizer _opt;
    private readonly IGenesisTokenizer _tok;
    private readonly Device _dev;
    private readonly int _vocab;
    private readonly int _pad;
    private readonly int _sep;
    private readonly int _eos;
    private readonly int _maxLen;
    private readonly int _maxAnswer;

    public long ParameterCount { get; }

    public TransformerTrainer(IGenesisTokenizer tok, int vocabSize, int dModel, int heads, int layers,
        int ffMult, int maxLen, int maxAnswer, Device dev, double lr)
    {
        _tok = tok;
        _dev = dev;
        _maxLen = maxLen;
        _maxAnswer = maxAnswer;
        _eos = tok.EosTokenId;
        _sep = tok.BosTokenId;          // reuse BOS as the input/answer separator
        _pad = vocabSize;               // pad id = one past the real vocab
        _vocab = vocabSize + 1;
        _net = new TinyTransformer(_vocab, dModel, heads, layers, ffMult, maxLen, dev);
        _opt = optim.Adam(_net.parameters(), lr: lr);
        long n = 0;
        foreach (var p in _net.parameters()) n += p.numel();
        ParameterCount = n;
    }

    private (int[] Seq, int SepIndex) BuildSeq(string input, string output)
    {
        var inp = _tok.Encode(input);
        var outp = _tok.Encode(output);
        var seq = new List<int>(inp.Length + outp.Length + 2);
        seq.AddRange(inp);
        var sepIndex = seq.Count;       // position of SEP
        seq.Add(_sep);
        seq.AddRange(outp);
        seq.Add(_eos);
        return (seq.ToArray(), sepIndex);
    }

    /// <summary>Train on one batch; returns mean answer-span loss.</summary>
    public double TrainBatch(IReadOnlyList<(string Input, string Output)> batch)
    {
        _net.train();
        using var scope = NewDisposeScope();

        var built = batch.Select(e => BuildSeq(e.Input, e.Output))
            .Where(s => s.Seq.Length <= _maxLen)
            .ToList();
        if (built.Count == 0) return 0.0;

        var len = built.Max(s => s.Seq.Length);
        var b = built.Count;
        var input = new long[b * (len - 1)];
        var target = new long[b * (len - 1)];
        Array.Fill(input, _pad);
        Array.Fill(target, _pad);  // pad = ignore_index for CE

        for (var i = 0; i < b; i++)
        {
            var (seq, sepIdx) = built[i];
            for (var t = 0; t < seq.Length - 1; t++)
            {
                input[i * (len - 1) + t] = seq[t];
                // Loss only where we PREDICT an answer-span token (the token at t+1 sits past the SEP).
                target[i * (len - 1) + t] = (t + 1) > sepIdx ? seq[t + 1] : _pad;
            }
        }

        var idx = tensor(input, new long[] { b, len - 1 }, dtype: ScalarType.Int64, device: _dev);
        var tgt = tensor(target, new long[] { b, len - 1 }, dtype: ScalarType.Int64, device: _dev);
        var logits = _net.forward(idx);
        var loss = nn.functional.cross_entropy(logits.reshape(-1, _vocab), tgt.reshape(-1), ignore_index: _pad);
        _opt.zero_grad();
        loss.backward();
        _opt.step();
        return loss.item<float>();
    }

    /// <summary>Greedy-decode the answer for an input; returns the decoded answer string.</summary>
    public string Generate(string input)
    {
        _net.eval();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        var seq = new List<int>(_tok.Encode(input)) { _sep };
        var produced = new List<int>();
        for (var step = 0; step < _maxAnswer && seq.Count < _maxLen; step++)
        {
            var idx = tensor(seq.ToArray(), new long[] { 1, seq.Count }, dtype: ScalarType.Int64, device: _dev);
            var logits = _net.forward(idx);                       // [1, T, vocab]
            var last = logits.select(1, seq.Count - 1).reshape(_vocab); // [vocab]
            var next = (int)last.argmax().item<long>();
            if (next == _eos || next == _pad) break;
            produced.Add(next);
            seq.Add(next);
        }
        return _tok.Decode(produced.ToArray()).Trim();
    }

    public double Accuracy(IReadOnlyList<(string Input, string Output)> data)
    {
        if (data.Count == 0) return 0.0;
        var correct = 0;
        foreach (var (inp, outp) in data)
            if (Generate(inp) == outp.Trim()) correct++;
        return correct / (double)data.Count;
    }
}
