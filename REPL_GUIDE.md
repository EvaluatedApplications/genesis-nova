# Genesis Nova - Interactive REPL Guide

## Quick Start

Start the REPL with:
```bash
cd src/bin/Release/net8.0
./GenesisNova.exe --genesis-repl
```

## Core Features

### 1. **Idle Introspection** (Background Processing)

Start background introspection cycles that run continuously:
```
genesis> introspect-idle
idle introspection started (runs continuously in background)
```

This processes the training queue in the background without blocking user input. Introspection happens every 100ms (10 cycles per second).

Stop it anytime with:
```
genesis> introspect-stop
idle introspection stopped
```

Enable verbose output to see progress:
```
genesis> verbose
verbose mode: ON
[idle] introspected=10 queue=1024
[idle] introspected=20 queue=1000
```

### 2. **Training with Feedback**

Train from a file with real-time loss reporting:
```
genesis> trainfile examples-50.jsonl 1
trained examples=50 epochs=1 loss=0.5661 time=6.64s
```

Or train individual examples:
```
genesis> train hello => hi
loss=0.5234 token=0.5234 route=0.0000
```

The REPL shows:
- `loss` - Total loss (combination of token and route losses)
- `token` - Token-level prediction error
- `route` - Route classification error
- `time` - Wall-clock training time

### 3. **Interactive Queries** (During Idle Introspection)

While idle introspection runs in the background, you can run queries:
```
genesis> predict hello
output=hello!

genesis> stats
vocab=100 hidden=48 queue=1024

genesis> queue
queue=1024
```

## Training Workflow

### Example 1: Minimal Setup
```
genesis> stats
vocab=100 hidden=48 queue=0

genesis> trainfile examples-50.jsonl 1
trained examples=50 epochs=1 loss=0.5661 time=6.64s

genesis> predict hello
output=hello!
```

### Example 2: Background Processing + Training
```
genesis> introspect-idle
idle introspection started (runs continuously in background)

genesis> verbose
verbose mode: ON

genesis> trainfile examples-50.jsonl 2
trained examples=50 epochs=2 loss=0.4123 time=12.45s
[idle] introspected=120 queue=512
[idle] introspected=130 queue=256

genesis> predict hello
output=hello!

genesis> introspect-stop
idle introspection stopped
```

### Example 3: Multi-Domain Training
```
genesis> trainfile arithmetic-examples.jsonl 1
trained examples=30 epochs=1 loss=0.3245 time=3.21s

genesis> trainfile language-examples.jsonl 1
trained examples=50 epochs=1 loss=0.5123 time=6.64s

genesis> predict "2 + 3"
output=5

genesis> predict "hello"
output=hi there
```

## Architecture Insights

### Computation Graph Management
- **Per-epoch cloning**: Parameters are cloned at epoch boundaries to break PyTorch's computation graph
- **Batch processing**: Examples processed in batches (~16 examples per batch)
- **First batch overhead**: ~380ms (graph growth), subsequent batches ~150ms (stable)

### Memory Management
- `queue` - Training queue depth (examples waiting for introspection)
- `introspect-idle` - Processes queue in background (10 cycles/sec)
- Higher queue depth = more introspection work pending

### Performance Characteristics
- 50 examples × 1 epoch: ~6.6 seconds
- 100 examples × 1 epoch: ~12-13 seconds (linear scaling)
- GPU utilization: ~80% (after optimization)

## Advanced Commands

### State Management
```
genesis> save checkpoint-v1.bin
saved checkpoint-v1.bin

genesis> load checkpoint-v1.bin
loaded checkpoint-v1.bin

genesis> compact
compacted=42 turns=15
```

### Memory & Reasoning
```
genesis> context
[conversation history summary]

genesis> concept learning
[concept description from memory]

genesis> relate concept1 concept2 0.8
relation updated
```

### Monitoring
```
genesis> stats
vocab=100 hidden=48 queue=1024

genesis> queue
queue=512
```

## Troubleshooting

### REPL freezes
- Idle introspection shouldn't block input
- If frozen, press Ctrl+C to interrupt

### Training is slow
- Check `queue` - high queue indicates introspection backlog
- Use `introspect-idle` to process in background
- Monitor `verbose` mode to see idle progress

### Loss not improving
- Generate more diverse examples with `--genesis-gen-examples`
- Train for more epochs: `trainfile examples-50.jsonl 3`
- Use `concept` command to inspect learned representations

## Architecture

```
GenesisRepl
├── Idle Introspection Thread
│   ├── 100ms cycle
│   ├── Processes 1 cycle per iteration
│   └── Non-blocking (doesn't interfere with user input)
├── Training Pipeline
│   ├── Per-batch processing (16 examples/batch)
│   ├── Per-epoch parameter cloning
│   └── Live feedback logging
└── User Input Handler
    ├── Async command processing
    └── Real-time query support
```

## Key Improvements Over Initial Design

1. **Background Introspection**: No longer blocks on long training jobs
2. **Real-time Feedback**: See loss values during training
3. **Interactive Queries**: Can predict/inspect while training
4. **Verbose Logging**: Optional detailed idle progress
5. **Optimized Graph Management**: Per-epoch cloning (not per-example)

## Next Steps

- [ ] Web UI with real-time graphs (training loss, accuracy over time)
- [ ] REST API for external training submissions
- [ ] Multi-GPU support for distributed training
- [ ] Automated hyperparameter tuning
- [ ] Export to ONNX for deployment
