# The Nucleus & the Cloud: how an element stores data across its dimensions

> Every element is a single vector, but its dimensions are not equal. Some are a **frozen nucleus**, the element's
> exact, immutable *identity*. The rest are a **free electron cloud**, where *meaning* lives and moves. Identity is
> re-derived from the symbol via the codec rather than stored, so it cannot drift; the only mutable region is the
> word face `[202,dim)`. An element can drift, cluster, and relate in that free region all it likes, and its
> identity never moves.
>
> This is the genesis dual-face (`research/03-SYMMETRY-BRIDGE` Corr. 7): the **arithmetic face is *like* the
> proton/nucleus** (crisp, conserved, exact), and the **semantic face is *like* the electron orbital** (distributed,
> probabilistic, mobile).
> Companions: `PLATONIC_THEORY.md` (the formal model), `PLATONIC_CONSCIOUSNESS.md`.

> **On the physics language (read this first).** "Nucleus", "orbital", "proton/electron", "conservation" are an
> **inspiration, a generative metaphor**, not a claim that the system models physics. A platonic space is a space
> of *ideas*; we are free to choose its rules as long as they satisfy the axioms (G1-G6, `PLATONIC_THEORY.md`). The
> literal content is mundane and checkable: some vector dimensions are an exact, never-mutated *identity* region;
> the rest are a learnable *meaning* region. Read the physics words as a vivid name for that split, nothing more.

---

## ⚠️ A note on the names: each face is named for its SLOTS, but holds the COMPOSITE one level up

The face names describe what each **slot holds** (the components), not what the face **represents**. Read them one
level up:

| face (named for slots) | slots hold… | …so the face actually represents | composition |
|---|---|---|---|
| numeric (poly/log) | digit place-values | a **NUMBER** (its value) | digits → number |
| **"char"** `[42,202)` | characters | a **WORD** (its identity) | chars → **word** |
| **"word"** `[202,dim)` | words | a **SENTENCE / phrase** | words → **sentence** |

So: the char face stores a **word**, and the word face stores a **sentence** (a whole phrase of words). The names
are off by one level. Each is named for its atoms, but holds the thing those atoms compose.

```mermaid
flowchart LR
  ch["characters"] -->|compose| wd["a WORD<br/>(lives in the 'char' face [42,202))"]
  wd -->|compose| se["a SENTENCE / phrase<br/>(lives in the 'word' face [202,dim))"]
  dg["digit place-values"] -->|compose| nu["a NUMBER's value<br/>(lives in the numeric face [0,42))"]
  classDef comp fill:#1a5276,color:#ffffff,stroke:#85c1e9,stroke-width:2px;
  classDef atom fill:#566573,color:#ffffff,stroke:#aeb6bf,stroke-width:1px;
  class wd,se,nu comp;
  class ch,dg atom;
```

---

## 1. One vector, two kinds of dimension

A concept is a vector of width `dim` (production 512). Reading left→right is reading from the crisp **nucleus** to
the diffuse **cloud**. The boundaries are fixed in `Core/FaceLayout.cs` (`PolyFaceMax = 42`,
`FaceLayout.cs:29`; `CharFaceStart = 42`, `FaceLayout.cs:58`; `WordFaceStart = 202`, `FaceLayout.cs:64`):

```mermaid
flowchart LR
  subgraph V["one element = one vector (dim 512)"]
    direction LR
    P["poly · [0,21)<br/>value · 10⁻ⁱ&nbsp;(add/sub)"]
    L["log · [21,42)<br/>ln│v│ · 10⁻ⁱ&nbsp;(mul/div)"]
    C["'char' face · [42,202)<br/><b>a WORD</b><br/>(slots = characters)"]
    W["'word' face · [202,512)<br/><b>a SENTENCE / phrase</b><br/>(slots = words): the big face"]
    P --- L --- C --- W
  end
  classDef nuc fill:#922b21,color:#ffffff,stroke:#f1948a,stroke-width:2px;
  classDef free fill:#1a5276,color:#ffffff,stroke:#85c1e9,stroke-width:2px;
  class P,L,C nuc;
  class W free;
```

- **Low end: small, crisp, structured.** 42 dims of pure algebra: a number's value, encoded so that
  `embed(a)+embed(b) = embed(a+b)`. Exact, generalizes to unseen operands, zero stored facts.
- **Middle: a word's identity.** The "char" face holds a word (composed from its characters): its fixed lexical
  fingerprint.
- **High end: large, diffuse, relational.** 310 dims (≈60% of the vector). The "word" face holds a **sentence /
  phrase of words**, and this is where a concept's meaning lives as a *cloud*: a superposition of the words and
  phrases it appears with. Ambiguity lives here, since a two-sense word is near *both* senses at once.

The **poly + log + char faces are the identity nucleus** (frozen, codec-derived); the **word face `[202,dim)` is
the single mutable region** for every element.

---

## 2. Identity is codec-derived, never stored

An element stores **only** its learnable orbital (the word face `[202,dim)`) plus its structural part-of edges
(`Element.cs:25-29`). The whole identity nucleus (the arithmetic poly/log faces and the char face) is **recomputed
from the symbol via the codec on demand** (`FaceCodec.AssemblePositiveFace`, `FaceCodec.cs:79-99`), so it *cannot*
drift. The mutable region begins at `FaceCodec.SemanticStart = FaceLayout.WordFaceStart` (`FaceCodec.cs:18-19`),
the same offset for **both** numbers and words.

So a number's identity is its *value* (read off the poly/log face), and a word's identity is the *word itself* (its
char-composed form). Both are reconstructed from the symbol, never mutated. Only the word/sentence face `[202,dim)`
actually moves:

```mermaid
flowchart TB
  subgraph NUM["a NUMBER: e.g. 5"]
    direction LR
    n1["🔒 identity&nbsp; [0,202)<br/><b>EXACT VALUE</b> (poly/log) + char form<br/>codec-derived: never stored"]
    n2["☁️ word face&nbsp; [202,512)<br/>FREE: settles near 'five', into phrases & meaning"]
    n1 --- n2
  end
  subgraph TXT["a WORD: e.g. five, cat"]
    direction LR
    t1["🔒 identity&nbsp; [0,202)<br/><b>THE WORD ITSELF</b> (char face)<br/>codec-derived: never stored"]
    t2["☁️ word face&nbsp; [202,512)<br/>FREE: the phrases/sentences it lives in (meaning)"]
    t1 --- t2
  end
  classDef nuc fill:#922b21,color:#ffffff,stroke:#f1948a,stroke-width:2px;
  classDef free fill:#1a5276,color:#ffffff,stroke:#85c1e9,stroke-width:2px;
  class n1,t1 nuc;
  class n2,t2 free;
```

| element | frozen nucleus (identity, `[0,202)`) | free cloud (meaning, `[202,dim)`) |
|---|---|---|
| number `5` | numeric `[0,42)`, the exact value | settles near `five`, gains meaning |
| word `five` / `cat` | the "char" face `[42,202)`, **the word itself** | the phrases / sentences it appears in |
| sentence / phrase | composed from its word-slots in the "word" face `[202,dim)` | (its meaning cloud) |

A number is routed through the homomorphism and carries no stored orbital for arithmetic, but it still gets a
semantic position written into the word face (so `5` can settle near `five`) *without* touching its exact
arithmetic face `[0,42)` (`FaceCodec.cs:91-98`).

---

## 3. The nucleus is the pivot: identity holds still while meaning moves

This is the whole point. Two elements can have **completely different nuclei** yet let their **clouds overlap**, and
that overlap *is* "they're related / they mean the same." The fixed nuclei are the reference frame the relational
geometry pivots off of; the clouds (the sentence-level "word" face) are where all the moving happens.

```mermaid
flowchart LR
  subgraph A["atom: 5"]
    A_n["🔒 nucleus<br/>value = 5<br/>(frozen, exact)"]
    A_c["☁️ cloud<br/>(free, mobile)"]
    A_n -. anchors .- A_c
  end
  subgraph B["atom: five"]
    B_n["🔒 nucleus<br/>the word 'five'<br/>(frozen)"]
    B_c["☁️ cloud<br/>(free, mobile)"]
    B_n -. anchors .- B_c
  end
  A_c <==>|"clouds overlap → related<br/>(5 means five)"| B_c
  classDef nuc fill:#922b21,color:#ffffff,stroke:#f1948a,stroke-width:2px;
  classDef free fill:#1a5276,color:#ffffff,stroke:#85c1e9,stroke-width:2px;
  class A_n,B_n nuc;
  class A_c,B_c free;
```

`5` and `five` will never be confused: their nuclei (exact value vs the exact word) are fixed and distinct forever.
But their *meaning clouds* drift together until they overlap, so retrieval treats them as the same thing. Identity
and meaning are stored in **different dimensions**, so you get both: permanent identity *and* fluid, shared,
ambiguous meaning.

### Why the cloud can move without ever corrupting identity

Every relational update touches only the free word-face dimensions; the nucleus is never *stored*, so it cannot
drift (it is re-derived from the symbol via the codec). Learning is *all* in the cloud, and the complement
(`¬e = −e`, G4) is re-enforced on the assembled face (`FaceCodec.Negate`, `FaceCodec.cs:101-108`). See below:

```mermaid
sequenceDiagram
  participant E as element
  participant Obs as observe(a, b, κ)
  Obs->>E: move the FREE word-face orbital toward (agree) / away (contradict) the neighbour
  Obs->>E: identity stays exact (codec-derived: never stored)
  Obs->>E: enforce ¬e = −e (G4 conservation)
  Note over E: only meaning moved: value & the word itself are untouched
```

---

## 4. Why this structure is the right way to store data

- **Identity is permanent and free of charge.** Identity is codec-derived and never stored, so it can't drift: an
  element can be pushed anywhere in the cloud and still decode to exactly itself (its value, or the word it is). No
  amount of relational learning can corrupt what a thing *is*.
- **The nucleus is a fixed coordinate frame.** Relations don't float in a vacuum; they pivot off the stable
  nuclei. You always know *what* two elements are; learning only settles *how they relate*.
- **Exact computation rides the frozen nucleus.** Arithmetic is read straight off the numeric nucleus
  (`embed(a)+embed(b)=embed(a+b)`), exact and generalizing, regardless of whatever the cloud is doing.
- **Meaning is large, distributed, and ambiguous, on purpose.** The big "word" face holds meaning as a
  *superposition of the phrases a concept appears in* (a sentence is literally a phrase of words), so related
  concepts cluster, unrelated ones go orthogonal, and a word with two senses sits near both at once. A single point
  couldn't do that; a cloud can.
- **The whole thing is a composition ladder.** digits → number, characters → word, words → sentence: each face
  holds the composite of the level below, and growth is *reuse* of the shared parts beneath it.
- **The two ends mirror each other.** Crisp quantity at the low end (the metaphorical "proton"), diffuse meaning at
  the high end (the metaphorical "orbital"), bound by the G4 conservation rule. Quantity is exact and singular;
  meaning is rich and plural. The layout puts each where it belongs.
