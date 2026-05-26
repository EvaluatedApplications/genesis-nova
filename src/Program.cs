using GenesisNova.Runtime;
if (await GenesisCli.TryHandleAsync(args))
    return;

Console.WriteLine("Genesis Nova CLI");
Console.WriteLine("  --genesis-train --file <path> [--epochs <n>] [--introspect-cycles <n>] [--eval-samples <n>] [--threads <n>] [--deterministic] [--backend cpu|gpu] [--baseline-checkpoint <path>] [--max-exact-drop <ratio>] [--no-parallel-math] [--no-auto-scale-vram] [--save <checkpoint>] [--log <logfile>]");
Console.WriteLine("  --genesis-repl");
Console.WriteLine("  REPL: train, trainfile, predict, introspect, concept, relate, queue, save, load, stats, context, compact, reset, help, exit");
