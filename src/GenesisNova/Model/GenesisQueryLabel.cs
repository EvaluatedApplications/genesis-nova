namespace GenesisNova.Model;

/// <summary>
/// Supervision for the GRU's platonic-query construction heads, derived from a training example's
/// own numeric structure (NOT from any surface grammar): the operands are the digit-run tokens in
/// the input and the operation is whichever single face op maps them onto the example's output.
///
/// OperationId indexes the face-derived op vocabulary: 0 = none/abstain, 1 = add, 2 = sub,
/// 3 = mul, 4 = div — i.e. {poly, log} × {+, −} plus abstention, the structure of the platonic
/// faces themselves rather than a training convenience.
/// </summary>
public readonly record struct GenesisQueryLabel(int OperationId, bool[] OperandMask);
