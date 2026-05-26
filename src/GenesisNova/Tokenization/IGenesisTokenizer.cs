namespace GenesisNova.Tokenization;

public interface IGenesisTokenizer
{
    int PadTokenId { get; }
    int BosTokenId { get; }
    int EosTokenId { get; }
    int VocabularySize { get; }
    IReadOnlyList<string> Vocabulary { get; }
    int[] Encode(string text, bool addBos = false, bool addEos = false);
    string Decode(IReadOnlyList<int> tokens);
}

