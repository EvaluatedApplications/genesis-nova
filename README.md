# Genesis Nova

Genesis Nova is a research prototype for building a learned reasoning substrate around text, symbols, and reusable concept structure. It is not a finished assistant or a productized chatbot; it is an experimental ML app that was built to show what the system can learn, retain, and reuse.

## What we built

At the user level, this project gives you a small but complete research environment:

- a training loop for example-driven learning
- a model that can generate outputs from prompts
- a structured memory layer for concepts and discovered transforms
- a REPL for experimentation
- a desktop UI for training controls
- checkpoint save/load support
- a test suite focused on arithmetic, language, routing, and memory behavior

In plain terms, we built a hybrid ML research runtime where the model is expected to learn useful structure from examples instead of relying on hand-written rules.

## What it demonstrates

This release shows that the system can:

- learn simple conversation patterns
- learn compact symbolic forms like arithmetic
- store and reuse concept relationships
- route between direct generation and learned structure
- train repeatedly on generated curricula
- persist state across sessions

## Current shape of the app

- **Training**: consumes paired examples and updates the model
- **Inference**: produces text and can use learned structure when available
- **Memory**: tracks concepts, relations, and discovered transform-like behavior
- **Controls**: REPL and UI expose training, configuration, and inspection
- **Persistence**: checkpoints and local conversation state are saved on disk

## Important choices in this iteration

- Route labels were removed from the training data contract
- Compression exists, but the default is `L2 = 0.0`
- The old supervised route head path is no longer part of training
- Legacy introspection code was removed so the release surface is smaller

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
