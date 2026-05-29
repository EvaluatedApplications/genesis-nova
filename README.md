# Genesis Nova

Genesis Nova is an experimental ML system for learning how text, symbols, and conceptual structure map to one another. It is not a productized chatbot; it is a research runtime for training a small neural model, observing what it learns, and checking whether learned structure is reusable at inference time.

## What it is trying to do

At a high level, the app learns from paired examples like:

```text
input -> output
```

Over time it tries to discover:

- direct token patterns
- arithmetic transforms
- concept relationships in memory
- when a request should be answered by the neural model vs. by learned structure

The goal is to keep the core model simple and general-purpose, while letting learned representations carry the useful structure instead of hardcoded rules.

## What is inside

- **Training pipeline**: reads examples, tokenizes them, and updates the model
- **Neural model**: predicts next tokens and supports inference-time routing
- **Platonic memory**: stores concepts, relations, and discovered transforms
- **Inference engine**: decides how to answer a request
- **REPL and UI**: tools for training, inspection, and hyperparameter control
- **Checkpointing**: saves and restores model state locally

## How training works

1. Load an example such as `say hello -> hello`
2. Convert the text into tokens
3. Train the neural model on the target sequence
4. Update concept memory and transform discovery from the same example
5. Repeat across many examples so the model generalizes

The current default keeps L2 compression off (`0.0`) so the model can learn freely. Compression is still available as an option if the model starts to grow too large later.

## How inference works

When you ask the model something, it:

1. Encodes the input
2. Checks whether a learned transform or concept chain can answer directly
3. Falls back to neural token generation when needed
4. Optionally uses memory or checkpoint context to bias the output

So the system is trying to combine two things:

- **neural prediction**
- **structured learned memory**

## Main code areas

- `src/GenesisNova/Data/` — example definitions and data generators
- `src/GenesisNova/Train/` — training orchestration
- `src/GenesisNova/Model/` — neural model and optimization step
- `src/GenesisNova/Cognition/` — concepts, transforms, and memory structures
- `src/GenesisNova/Infer/` — routing and generation logic
- `src/GenesisNova/Runtime/` — CLI, conversation memory, checkpoint flow
- `src/GenesisNova/UI/` — training controls in the desktop UI

## Controls

### REPL

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-repl
```

Useful commands:

- `train <input> => <output>`
- `trainfile <path> [epochs]`
- `config l2`
- `config l2 <preset>`
- `context`
- `compact`
- `reset`

### UI

The Training tab includes L2 presets:

- `off`
- `mild`
- `balanced`
- `aggressive`
- `extreme`

## Build and test

```bash
dotnet build GenesisNova.slnx
dotnet test GenesisNova.slnx
```

## Data

Training examples live in `data/`. The main corpus is:

```text
data\genesis-nova-train-expanded.txt
```
