using GenesisNova.Cognition.Platonic;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Runtime;
using GenesisNova.Train;
using System.Diagnostics;
using System.Net;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace GenesisNova.UI;

public class MainWindow : Form
{
    private const int MaxTrainingLogChars = 240_000;
    private const int TargetTrainingLogChars = 180_000;
    private const int MaxReplLogChars = 200_000;
    private const int TargetReplLogChars = 150_000;
    private const int AutoScrollSlackChars = 256;

    private readonly GenesisEvalAppRuntime _runtime;
    private readonly string _gymStateDir;
    private readonly bool _seededFromStarter; // true if this fresh start was seeded from the committed repo starter
    private CancellationTokenSource? _autonomousTrainingCts;
    private CancellationTokenSource? _gymCts;
    private Task? _gymLoopTask;            // the in-flight GymLoopAsync (tracked so a stop/restart can AWAIT it drain)
    private readonly SemaphoreSlim _gymLifecycleGate = new(1, 1); // serialize start/stop/restart — no overlapping loops, no start-before-save
    private GymTrainer? _gym;
    private int _gymCycle;
    private int _lastDifficulty;
    private double _lastLoss, _lastAcc, _lastRoute, _lastConf;
    private double _gymBar = 0.90;
    private int _gymTrainPerCycle = 256;
    private int _gymThrottlePct = 0;  // 0..500 — rest this % of the last cycle's time between cycles (0 = full speed)
    private HttpListener? _control;
    // Bounded in-memory mirror of the UI output log so the headless /log endpoint can serve the tail (gym/nav/train
    // lines) without the WinForms TextBox. Newest at the back; lock-guarded since AppendOutput runs off many threads.
    private readonly Queue<string> _logRing = new();
    private const int LogRingCapacity = 1000;
    private string _exampleFolder;
    private TabControl _tabControl = null!;
    private readonly List<(int Cycle, double Acc, double Loss)> _warmHistory = new(); // Inspect-tab warming curve
    private GenesisNova.Runtime.NavTrainCycleResult _lastNavTrain; // last navigator-training cycle (for the inspect tab)
    private System.Windows.Forms.Timer? _inspectTimer;
    private volatile bool _inspectReading;
    private string[]? _inspectGlue, _inspectContent; // probe words pulled from the LIVE foundation vocabulary (so the table shows trained words)
    private bool _replTraceEnabled;
    private PlatonicActivationView? _latestActivation;
    private int _activeTrainingOperations;
    private readonly SemaphoreSlim _replCommandGate = new(1, 1);
    private SplitContainer? _replVisualizerSplit;
    private DateTime _lastTrimTime = DateTime.MinValue;
    private const int TrimDebounceMs = 500;  // Only trim if >500ms since last trim

    public MainWindow()
    {
        _gymStateDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova",
            "gym");

        // FRESH-START SEED: if this machine has no gym checkpoint yet, copy the committed repo STARTER
        // (models/genesis-nova — a warmed shared baseline) into the gym state dir, so a clone / fresh start begins from
        // that model instead of an empty brain. Runs BEFORE the runtime (AutoResume reads the seeded pointer). No-op once
        // a local checkpoint exists — the local fork then evolves independently and is never overwritten. See MODEL_STORAGE.md.
        try
        {
            var repoRoot = GuessRepoRoot();
            if (!string.IsNullOrEmpty(repoRoot))
            {
                var starter = Path.Combine(repoRoot, "models", "genesis-nova.json");
                var local = Path.Combine(_gymStateDir, "genesis-nova.autosave.checkpoint.json");
                _seededFromStarter = GenesisNova.Persistence.GenesisShardedCheckpointStore.SeedFromStarter(starter, local);
            }
        }
        catch { /* seeding is best-effort — a missing/torn starter just means a fresh empty brain */ }

        // Production gym standard: 2048 GRU controller + decoupled 1024 substrate face-dim, fixed (no autoscale).
        // The 1024 face gives the learned-meaning orbital [416,1024) = 608 dims (the frozen address bands ≤416 are fixed).
        // The ARCHITECTURE (conscious field, dialectical core, keep-core, the meaning-space self, function gradient)
        // is defined ONCE in GenesisNovaConfig.WithProductionMechanisms() — shared with the RaceBench benchmark so the
        // race always runs the same brain. Here we set only the deployment INFRA (dims, dirs, backend, persistence).
        _runtime = new GenesisEvalAppRuntime(new GenesisNovaConfig
        {
            Backend = ComputeBackend.Gpu,
            HiddenSize = 2048,
            FaceDimensionOverride = 1024,
            AutoPersist = true,
            AutoResume = true,
            LocalStateDirectory = _gymStateDir,
            AutoScaleVram = false,
        }.WithProductionMechanisms());
        _exampleFolder = ResolveDefaultExampleFolder();
        
        Text = "Genesis Nova - CPU Training / VRAM Inference";
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(1200, 800);
        StartPosition = FormStartPosition.CenterScreen;

        LoadUI();

        // Ensure the main window is made visible/foreground on startup.
        Shown += (_, _) =>
        {
            if (WindowState == FormWindowState.Minimized)
                WindowState = FormWindowState.Normal;
            ShowInTaskbar = true;
            Activate();
            BringToFront();
            if (_seededFromStarter)
                AppendOutput("[seed] fresh start — seeded the gym from the committed repo starter (models/genesis-nova)");
            // Gym no longer AUTO-STARTS on launch (user preference) — the model loads idle; press "▶ Start Gym" to train.
            AppendOutput("[gym] ready — auto-start disabled. Press ▶ Start Gym to begin training.");
            StartControlServer(); // headless control endpoint so tools/Claude can drive the live model
        };
        
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _gymCts?.Cancel();
        try { _control?.Stop(); } catch { }
        _autonomousTrainingCts?.Cancel();
        _autonomousTrainingCts?.Dispose();
        _autonomousTrainingCts = null;
        // Persist SYNCHRONOUSLY on close. The gym's final save runs on a background task we don't await, so the
        // process can exit mid-write → a half-written / model-vs-substrate-inconsistent checkpoint that reloads weird
        // next launch. Block here (bounded) for one clean save after the gym has been told to stop. Run off the UI
        // thread so the gate wait can't deadlock against the close.
        try { Task.Run(() => _runtime.SaveAsync(_runtime.AutoCheckpointPath)).Wait(TimeSpan.FromSeconds(30)); } catch { }
        _replCommandGate.Dispose();
        base.OnFormClosed(e);
    }

    private void LoadUI()
    {
        _tabControl = new TabControl
        {
            Dock = DockStyle.Fill,
            Padding = new Point(8, 8)
        };

        // Tab 1: Autonomous Training
        _tabControl.TabPages.Add(CreateAutonomousTrainingTab());

        // Tab 2: REPL
        _tabControl.TabPages.Add(CreateReplTab());

        // Tab 3: Inspect — live view of the platonic space (the foundation-warming debugger)
        _tabControl.TabPages.Add(CreateInspectTab());

        Controls.Add(_tabControl);
    }

    private TabPage CreateAutonomousTrainingTab()
    {
        var tab = new TabPage("Gym");

        var mainLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            Padding = new Padding(12)
        };
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 70));

        mainLayout.Controls.Add(CreateGymControlPanel(), 0, 0);          // lean gym controls (replaces the old autonomous config)
        mainLayout.Controls.Add(CreateAutonomousResultsPanel(), 1, 0);   // reused: live log (AutoOutputBox) + stats (AutoStatsText)

        tab.Controls.Add(mainLayout);
        return tab;
    }

    // Lean control panel for the GYM (replaces the irrelevant autonomous dataset/numeric config). Creates the
    // Start/Pause buttons (named AutoTrainBtn/AutoStopBtn so the gym wiring finds them) + two relevant knobs.
    private Panel CreateGymControlPanel()
    {
        var panel = new Panel { Dock = DockStyle.Fill, Padding = new Padding(8) };
        var flow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoScroll = true };

        flow.Controls.Add(new Label { Text = "GYM — procedural skill trainer", AutoSize = true, Font = new Font("Segoe UI", 13, FontStyle.Bold), Margin = new Padding(0, 0, 0, 4) });
        flow.Controls.Add(new Label { Text = "Trains the GRU's core skills on synthetic data.\nDifficulty is an unbounded, mastery-gated LEVEL.\nRuns automatically on startup.", AutoSize = true, Margin = new Padding(0, 0, 0, 12) });

        var startBtn = new Button { Text = "▶ Start Gym", Width = 280, Height = 38, BackColor = Color.FromArgb(51, 153, 102), ForeColor = Color.White, Font = new Font("Segoe UI", 11, FontStyle.Bold), Name = "AutoTrainBtn" };
        startBtn.Click += (s, e) => _ = StartGymAsync();
        flow.Controls.Add(startBtn);

        var stopBtn = new Button { Text = "⏸ Pause Gym", Width = 280, Height = 38, BackColor = Color.FromArgb(204, 51, 51), ForeColor = Color.White, Font = new Font("Segoe UI", 11, FontStyle.Bold), Name = "AutoStopBtn", Enabled = false };
        stopBtn.Click += (s, e) => _ = StopGymAsync();
        flow.Controls.Add(stopBtn);

        flow.Controls.Add(new Label { Text = "Mastery bar (level-up threshold)", AutoSize = true, Margin = new Padding(0, 16, 0, 2) });
        flow.Controls.Add(new NumericUpDown { Name = "GymBar", Width = 120, DecimalPlaces = 2, Increment = 0.05M, Minimum = 0.50M, Maximum = 0.99M, Value = 0.90M });

        flow.Controls.Add(new Label { Text = "Examples per cycle", AutoSize = true, Margin = new Padding(0, 12, 0, 2) });
        flow.Controls.Add(new NumericUpDown { Name = "GymTrainPerCycle", Width = 120, Minimum = 16, Maximum = 256, Increment = 16, Value = 256 });

        flow.Controls.Add(new Label { Text = "Throttle (% backoff of last cycle time)", AutoSize = true, Margin = new Padding(0, 12, 0, 2) });
        var throttleNum = new NumericUpDown { Name = "GymThrottle", Width = 120, Minimum = 0, Maximum = 500, Increment = 25, Value = 0 };
        throttleNum.ValueChanged += (s, e) => _gymThrottlePct = (int)throttleNum.Value; // live — applies next cycle, no restart
        flow.Controls.Add(throttleNum);

        // CURRICULA — FLAT + standardized. Every trainable task is a PEER checkbox: no tiers, no nesting, no groups.
        // Read TOP-TO-BOTTOM = the order to train — PREREQUISITE (prebake) → FUNDAMENTALS → GYM skills → applied.
        // Only the PREBAKE is auto-checked: it's the prerequisite that warms the learned function-word signal the
        // Merge parse / fact memory / retrieval all read (see [[nova-merge-substrate-plan]]). Tick more as each is ready.
        flow.Controls.Add(new Label { Text = "Curricula — run top → bottom (only the prebake is checked; train it first):", AutoSize = true, Margin = new Padding(0, 18, 0, 4), Font = new Font("Segoe UI", 10, FontStyle.Bold) });

        // Uniform peer checkbox — same indent for every task. The `startChecked` ones are the PRODUCTION hot path that runs
        // by default (run order is set by BootstrapRank below, NOT by this flag): the prebake prerequisite + nav-reasoning
        // (so the navigator trains on a taxonomy and its HELD-OUT curve populates without the user ticking it). Corpus
        // prediction (masked-cloze base-model objective) is OPT-IN — its production learn path is unexercised, so the user
        // enables it deliberately and watches its first run, rather than mixing an untested objective into cycle 1.
        static CheckBox Cur(string name, string text, bool startChecked = false) =>
            new() { Name = name, Text = text, Checked = startChecked, AutoSize = true, Margin = new Padding(20, 1, 0, 1) };

        flow.Controls.Add(Cur("CurPrebakeLanguage", "Prebake — warm function-word recognition from BOTH a real text corpus (Wikipedia) + synthetic L1", startChecked: true));
        flow.Controls.Add(Cur("CurCorpusPrediction", "Corpus prediction (masked cloze) — self-supervised: predict a held-out CONTENT word from context; corpus BUILDS knowledge into the substrate", startChecked: false));
        flow.Controls.Add(Cur("CurOpCues", "Op-cue words — sum / difference / product / quotient → operator"));
        flow.Controls.Add(Cur("CurNumberWords", "Number words — digit ↔ word lexicon"));
        flow.Controls.Add(Cur("CurNavReasoning", "Nav-reasoning — grow an is-a taxonomy + level cues; trains the navigator's walk (HELD-OUT curve in [nav-heldout])", startChecked: true));
        foreach (var skill in Enum.GetValues<GenesisNova.Train.GymSkill>())
            flow.Controls.Add(Cur("GymSkill_" + skill, "Gym — " + GymSkillLabel(skill)));
        flow.Controls.Add(Cur("CurMemCode", "Memory + Code index"));
        flow.Controls.Add(Cur("CurCreators", "Creators — skill ladder"));
        flow.Controls.Add(Cur("CurPersonality", "Personality — rude chatbot (seeded reply chunks)"));

        flow.Controls.Add(new Label { Text = "Scheduler:", AutoSize = true, Margin = new Padding(0, 10, 0, 0), ForeColor = Color.DimGray });
        flow.Controls.Add(new CheckBox { Name = "CurFocused", Text = "Focused + rehearsal (unticked = all tasks every cycle)", Checked = true, AutoSize = true, Margin = new Padding(20, 0, 0, 0) });
        // The platonic NAVIGATOR (query-conditioned walker) trains a LIGHT step each gym cycle on the live space, so it
        // accumulates overnight alongside the model. Default ON so an unattended run trains it; untick to leave it frozen.
        flow.Controls.Add(new CheckBox { Name = "CurTrainNavigator", Text = "Train Navigator — the platonic walker learns the live space each cycle (accumulates overnight)", Checked = true, AutoSize = true, Margin = new Padding(20, 0, 0, 0) });
        flow.Controls.Add(new Label { Text = "Memory index file (MEMORY.md):", AutoSize = true, Margin = new Padding(0, 8, 0, 2) });
        flow.Controls.Add(new TextBox { Name = "MemPath", Width = 270, Text = DefaultMemoryIndexPath() });
        flow.Controls.Add(new Label { Text = "Code root directory:", AutoSize = true, Margin = new Padding(0, 6, 0, 2) });
        flow.Controls.Add(new TextBox { Name = "CodePath", Width = 270, Text = DefaultCodeRoot() });
        flow.Controls.Add(new Label { Text = "(curriculum changes take effect on next Start)", AutoSize = true, ForeColor = Color.DimGray, Margin = new Padding(0, 2, 0, 0) });

        flow.Controls.Add(new Label
        {
            Text = "Control endpoint (headless):\nhttp://127.0.0.1:8787/\n  /status\n  /recall?q=...   /query?q=...\n  /probe?q=CONCEPT\n  /log?tail=N   /nav\n  /throttle?pct=0..500\n  /gym/start   /gym/stop  (= resume/pause)\n  /gym/restart  (clean stop+save → start)\n  /gym/trainers[?set=|enable=|disable=]",
            AutoSize = true,
            ForeColor = Color.DimGray,
            Margin = new Padding(0, 20, 0, 0)
        });

        panel.Controls.Add(flow);
        return panel;
    }

    private static string GuessRepoRoot()
    {
        var d = AppContext.BaseDirectory;
        for (var i = 0; i < 8 && !string.IsNullOrEmpty(d); i++)
        {
            if (File.Exists(Path.Combine(d, "GenesisNova.slnx"))) return d;
            d = Path.GetDirectoryName(d);
        }
        return string.Empty;
    }

    private static string DefaultCodeRoot()
    {
        var root = GuessRepoRoot();
        var src = string.IsNullOrEmpty(root) ? string.Empty : Path.Combine(root, "src");
        return Directory.Exists(src) ? src : root;
    }

    private static string DefaultMemoryIndexPath()
    {
        var p = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".claude", "projects", "C--Users-dongy", "memory", "MEMORY.md");
        return File.Exists(p) ? p : string.Empty;
    }

    private TabPage CreateReplTab()
    {
        var tab = new TabPage("REPL");

        _replVisualizerSplit = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterWidth = 6
        };
        _replVisualizerSplit.SplitterDistance = (int)(ClientSize.Width * 0.6);

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(12)
        };
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 80));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 50));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));

        // Output display
        var outputBox = new ListBox
        {
            Dock = DockStyle.Fill,
            Font = new Font("Consolas", 10),
            BackColor = Color.Black,
            ForeColor = Color.Lime,
            Name = "ReplOutput",
            SelectionMode = SelectionMode.MultiExtended
        };
        layout.Controls.Add(outputBox, 0, 0);

        // Input field
        var inputBox = new TextBox
        {
            Dock = DockStyle.Fill,
            Font = new Font("Consolas", 10),
            Name = "ReplInput",
            Multiline = false,
            Height = 30
        };
        inputBox.KeyDown += (s, e) =>
        {
            if (e.KeyCode != Keys.Enter)
                return;
            e.Handled = true;
            e.SuppressKeyPress = true;
            ExecuteReplCommand();
        };
        layout.Controls.Add(inputBox, 0, 1);

        // Button panel
        var buttonPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            Name = "ReplButtonPanel"
        };
        var visualizerLayoutLabel = new Label
        {
            Text = "Visualizer:",
            AutoSize = true,
            Margin = new Padding(0, 8, 6, 0)
        };
        buttonPanel.Controls.Add(visualizerLayoutLabel);

        var layoutMode = new ComboBox
        {
            Width = 170,
            Height = 32,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Name = "ReplVizLayoutMode"
        };
        layoutMode.Items.Add("Side by side");
        layoutMode.Items.Add("Under REPL");
        layoutMode.SelectedIndex = 0;
        layoutMode.SelectedIndexChanged += (_, _) => ApplyReplVisualizerLayout(layoutMode.SelectedIndex == 1);
        buttonPanel.Controls.Add(layoutMode);

        var executeBtn = new Button { Text = "Send", Width = 100, Height = 32 };
        executeBtn.Click += (s, e) => ExecuteReplCommand();
        buttonPanel.Controls.Add(executeBtn);

        layout.Controls.Add(buttonPanel, 0, 2);
        _replVisualizerSplit.Panel1.Controls.Add(layout);
        _replVisualizerSplit.Panel2.Controls.Add(CreateVisualizerPanel());
        tab.Controls.Add(_replVisualizerSplit);
        tab.Resize += (_, _) => ApplyReplVisualizerLayout(layoutMode.SelectedIndex == 1);
        ApplyReplVisualizerLayout(false);
        
        // Welcome message
        AppendToRepl("Genesis Nova Chat REPL\nType prompts directly (no context wrapping).\nUse /help for commands.\n> ", Color.Cyan);
        
        return tab;
    }

    private Control CreateVisualizerPanel()
    {
        // A readable STEP-BY-STEP reasoning trace (replaces the old node/edge graph): how the model reached the
        // answer + which platonic elements it used. Populated on every REPL query by UpdateVisualizer.
        return new RichTextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            Font = new Font("Consolas", 9.5f),
            BackColor = Color.FromArgb(18, 18, 22),
            ForeColor = Color.FromArgb(222, 222, 222),
            WordWrap = false,
            Name = "ReplTrace",
            Text =
                "Step-by-step reasoning trace.\r\n\r\n" +
                "Run a REPL query — this shows HOW the model reached the answer:\r\n" +
                "  tokenize -> route chosen -> platonic elements used (concepts, relations,\r\n" +
                "  faces / homomorphism, learned transforms, evidence hops) -> decode -> verdict."
        };
    }

    private void ApplyReplVisualizerLayout(bool underRepl)
    {
        if (_replVisualizerSplit is null)
            return;

        _replVisualizerSplit.Orientation = underRepl ? Orientation.Horizontal : Orientation.Vertical;
        _replVisualizerSplit.SplitterDistance = underRepl
            ? Math.Max(180, (int)(_replVisualizerSplit.Height * 0.52))
            : Math.Max(260, (int)(_replVisualizerSplit.Width * 0.55));
    }

    private void ExecuteReplCommand()
    {
        var replInput = GetControl<TextBox>("ReplInput");
        if (replInput == null || string.IsNullOrWhiteSpace(replInput.Text)) return;

        var command = replInput.Text.Trim();
        AppendToRepl($"{command}\n", Color.White);
        replInput.Clear();

        Task.Run(async () =>
        {
            await _replCommandGate.WaitAsync();
            try
            {
                var result = await ExecuteReplPromptAsync(command);
                AppendToRepl($"{result}\n> ", Color.Cyan);
            }
            catch (Exception ex)
            {
                AppendToRepl($"Error: {ex.Message}\n> ", Color.Red);
            }
            finally
            {
                _replCommandGate.Release();
            }
        });
    }

    private async Task<string> ExecuteReplPromptAsync(string prompt)
    {
        var trimmed = prompt.Trim();
        if (trimmed.Length == 0)
            return string.Empty;

        if (trimmed.StartsWith("/", StringComparison.Ordinal))
        {
            if (trimmed.Equals("/help", StringComparison.OrdinalIgnoreCase))
                return GetReplHelp();
            if (trimmed.StartsWith("/trace", StringComparison.OrdinalIgnoreCase))
            {
                var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (parts.Length == 1)
                {
                    _replTraceEnabled = !_replTraceEnabled;
                }
                else
                {
                    _replTraceEnabled = parts[1].Equals("on", StringComparison.OrdinalIgnoreCase);
                    if (parts[1].Equals("off", StringComparison.OrdinalIgnoreCase))
                        _replTraceEnabled = false;
                }

                return $"Trace mode: {(_replTraceEnabled ? "ON" : "OFF")}";
            }
            if (trimmed.StartsWith("/nav", StringComparison.OrdinalIgnoreCase))
            {
                var parts = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (parts.Length < 2)
                    return "Usage: /nav <concept> [genus|domain|root]   (walk the platonic navigator from <concept>)";
                // An optional trailing cue keyword; everything between is the concept anchor.
                var last = parts[^1].ToLowerInvariant();
                var hasCue = last is "genus" or "domain" or "root";
                var cue = hasCue ? last : null;
                var concept = string.Join(' ', parts.Skip(1).Take(parts.Length - 1 - (hasCue ? 1 : 0)));
                if (concept.Length == 0)
                    return "Usage: /nav <concept> [genus|domain|root]   (walk the platonic navigator from <concept>)";
                // GATED like a predict (the walk runs torch under the model gate); never touches torch on this thread.
                var obs = await Task.Run(() => _runtime.WalkNavigator(concept, cue));
                return FormatNavWalk(obs);
            }
            if (trimmed.Equals("/stats", StringComparison.OrdinalIgnoreCase))
            {
                var checkpointPath = _runtime.AutoCheckpointPath;
                var checkpointInfo = File.Exists(checkpointPath)
                    ? $"{new FileInfo(checkpointPath).Length / (1024.0 * 1024.0):F2} MB"
                    : "missing";
                return
                    $"Vocabulary Size: {_runtime.VocabularySize}\n" +
                    $"Hidden Size: {_runtime.HiddenSize}\n" +
                    $"AutoResume: {_runtime.AutoResumeEnabled}\n" +
                    $"Checkpoint: {checkpointPath}\n" +
                    $"Checkpoint Size: {checkpointInfo}";
            }
            if (trimmed.Equals("/context", StringComparison.OrdinalIgnoreCase))
                return "Conversation context is disabled in REPL mode.";
            if (trimmed.Equals("/reset", StringComparison.OrdinalIgnoreCase))
                return "Conversation context is disabled in REPL mode.";
            if (trimmed.Equals("/wrong", StringComparison.OrdinalIgnoreCase))
                return "Conversation context is disabled in REPL mode.";
            if (trimmed.StartsWith("/trainfile ", StringComparison.OrdinalIgnoreCase))
            {
                var payload = trimmed["/trainfile ".Length..].Trim();
                var (path, epochs) = ParsePathAndEpochs(payload);
                var resolvedPath = ResolveTrainingPath(path);
                var report = await ExecuteTraining(resolvedPath, epochs);
                return
                    $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4} " +
                    $"success={report.ExampleSuccessRate:P1}";
            }

            return $"Unknown command: {trimmed}";
        }

        var modelInput = trimmed;
        try
        {
            GenesisPredictTaskData? predict = null;
            var waitingNoticeShown = false;
            while (predict is null)
            {
                predict = await _runtime.TryPredictAsync(modelInput, gateWaitMilliseconds: 200);
                if (predict is not null)
                    break;

                if (!waitingNoticeShown)
                {
                    waitingNoticeShown = true;
                    AppendToRepl("[repl] waiting for a safe model slot...\n", Color.DarkGray);
                }

                await Task.Delay(120);
            }
            var output = predict.Result?.Output?.Trim();
            if (string.IsNullOrWhiteSpace(output))
                output = "(no response)";

            if (predict.Result is not null)
                UpdateVisualizer(trimmed, predict.Result);
            if (_replTraceEnabled && predict.Result is not null)
                return BuildReplTrace(trimmed, predict.Result, output);

            return output;
        }
        catch (Exception ex)
        {
            AppendToRepl($"[debug] Full error: {ex}\n", Color.Red);
            return $"Prediction error: {ex.InnerException?.Message ?? ex.Message}";
        }
    }

    private string GetReplHelp()
    {
        return @"Available commands:
  <plain text>               - Query model directly (no context)
  /reset                     - No-op (context disabled)
  /wrong                     - No-op (context disabled)
  /context                   - Shows context disabled status
  /stats                     - Show model stats
  /trace [on|off]            - Toggle stage-by-stage inference trace
  /nav <concept> [cue]       - Walk the platonic navigator from <concept> (cue=genus|domain|root, default genus)
  /trainfile <path> [epochs] - Train from file
  /help                      - Show this message";
    }

    // Render an observational navigator walk (the `/nav` command) — the emergent answer, reached-vs-abstained, the
    // trajectory of concepts, the cue, and whether the live engine self conditioned the walk.
    private static string FormatNavWalk(GenesisNova.Runtime.NavWalkObservation o)
    {
        var traj = o.Trajectory is { Count: > 0 } ? string.Join(" → ", o.Trajectory) : "(none)";
        var status = o.Reached ? "reached" : "abstained";
        var selfNote = o.SelfConditioned ? "active (conditioned the walk)" : "empty (no dwelling context yet)";
        return
            $"[nav] '{o.Anchor}'  cue={o.Cue}\n" +
            $"  answer:     {o.Answer}  ({status})\n" +
            $"  trajectory: {traj}\n" +
            $"  self:       {selfNote}\n" +
            $"  note:       {o.Message}";
    }

    private string BuildReplTrace(string input, GenesisNova.Infer.GenerationResult result, string output)
    {
        var inputTokenIds = _runtime.EncodeTokens(input);
        var generatedTokenIds = result.GeneratedTokens ?? [];
        // Batch-decode under one model-gate acquisition instead of one per token (see TokenTexts).
        var inputTokenTexts = _runtime.TokenTexts(inputTokenIds);
        var outputTokenTexts = _runtime.TokenTexts(generatedTokenIds);
        var inputTokens = inputTokenIds.Select((id, i) => $"{id}:{inputTokenTexts[i]}");
        var outputTokens = generatedTokenIds.Select((id, i) => $"{id}:{outputTokenTexts[i]}");
        var decodedFromTokens = generatedTokenIds.Length == 0
            ? string.Empty
            : _runtime.DecodeTokens(generatedTokenIds);

        return
            $"[trace]\n" +
            $"stage 1 tokenize.input: [{string.Join(", ", inputTokens)}]\n" +
            $"stage 2 route: decision={result.DecisionPath}, platonic={result.UsedPlatonicQuery}, neuralFallback={result.UsedNeuralFallback}, confidence={result.PlatonicConfidence:F3}\n" +
            $"stage 3 bias: applied={result.AppliedBiasCount}, avgMagnitude={result.AverageBiasMagnitude:F4}, hops={result.PlatonicHopCount}, chunks={result.ChunksGenerated}\n" +
            $"stage 4 tokens.output: [{string.Join(", ", outputTokens)}]\n" +
            $"stage 5 decode: \"{decodedFromTokens}\"\n" +
            $"answer: {output}";
    }

    private void UpdateVisualizer(string input, GenesisNova.Infer.GenerationResult result)
    {
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => UpdateVisualizer(input, result)));
            return;
        }

        _latestActivation = _runtime.AnalyzePlatonicActivation(input);
        var box = GetControl<RichTextBox>("ReplTrace");
        if (box is null) return;
        var act = _latestActivation;
        var platonic = !result.UsedNeuralFallback;
        var (mech, why) = InferenceTraceFormatter.MechanismOf(result);
        var sb = new System.Text.StringBuilder();
        void H(string s) => sb.Append(s).Append("\r\n");

        H("══════════ HOW IT ANSWERED ══════════");
        H($"Q: {input}");
        H($"A: {result.Output}");
        H("");
        H($"MECHANISM    {mech} — {why}");
        H($"ROUTE        {result.DecisionPath}   (confidence {result.PlatonicConfidence:F2}, {(platonic ? "platonic" : "neural fallback")})");
        H($"TOKENIZE     {(act.InputTokens.Length == 0 ? "(none)" : string.Join("  |  ", act.InputTokens))}");
        H("");

        var dp = (result.DecisionPath ?? string.Empty).ToLowerInvariant();
        if (mech == "CALCULATED")
        {
            if (dp.Contains("expression-chain"))
            {
                H("HOW IT CHAINED IT  (a multi-operator expression — chained compute-elements, NOT one lookup):");
                H("   1. the plan head recognised a MULTI-operator expression from context");
                H("   2. each operator was classified from its OWN local context by the op head (no fixed symbol→op map)");
                H("   3. evaluated with precedence (×/÷ before +/−); each binary step = ONE substrate R2 compose + homomorphic decode");
                H($"   4. chained {result.PlatonicHopCount} compute-element step(s)  ->  {result.Output}");
                H("   => GENERALIZES: the chain computes for any operands, never memorised.");
            }
            else if (InferenceTraceFormatter.ParseArith(input) is { } a)
            {
                var rule = a.Face == "poly" ? "poly(a)+poly(b) = poly(a+b)" : "log(a)+log(b) = log(a*b)";
                H("HOW IT COMPUTED IT  (calculated, NOT remembered — numbers never form stored edges):");
                H($"   1. classified  op = {a.Op},  operands = [{string.Join(", ", a.Operands)}]  (op inferred from context)");
                H($"   2. routed to the {a.Face.ToUpperInvariant()} face — the homomorphic one:  {rule}");
                H($"   3. composed the operand faces in the geometry and decoded  ->  {result.Output}");
                H("   => GENERALIZES: the SAME homomorphism computes operands never trained (e.g. 5000 + 2).");
            }
            else if (result.RoutedTransform is not null)
            {
                H("HOW IT COMPUTED IT  (a learned function applied as a transform vector — computed, not recalled):");
                H($"   1. selected the learned function '{result.RoutedTransform}' from the space (chosen by relation)");
                H($"   2. applied it by COMPOSITION (embed(x) + T(f)) and decoded  ->  {result.Output}");
            }
            else
                H("HOW: computed on the substrate (a homomorphism / transform), not recalled from a stored fact.");
        }
        else if (mech == "REMEMBERED" && dp.Contains("geometric"))
        {
            H("HOW IT RECALLED IT  (geometric content-addressing — POSITION is identity, NOT a stored edge):");
            H($"   anchors (your input found in the space): {(act.Anchors.Length == 0 ? "(none)" : string.Join(", ", act.Anchors))}");
            H("   1. read your input concept's POSITION in the semantic face");
            H("   2. the lattice VP-Tree returned the NEAREST stored concept by face distance");
            H($"   3. nearest concept (closeness {result.PlatonicConfidence:F2})  ->  {result.Output}");
            H("   => positioned by training (message-passing pulled related concepts together; no edge needed).");
        }
        else if (mech == "REMEMBERED")
        {
            H("HOW IT RECALLED IT  (remembered a learned association — a stored relation edge, NOT computed):");
            H($"   anchors (your input found in the space): {(act.Anchors.Length == 0 ? "(none)" : string.Join(", ", act.Anchors))}");
            var rel = act.Edges
                .Where(e => act.Anchors.Contains(e.Left, StringComparer.OrdinalIgnoreCase) || act.Anchors.Contains(e.Right, StringComparer.OrdinalIgnoreCase))
                .Take(6).ToList();
            if (rel.Count > 0)
            {
                H("   learned edges from your input (cue <-> answer):");
                foreach (var e in rel) H($"     - {e.Left} <-> {e.Right}   trust {e.Score:F2}  seen {e.ObservationCount}");
            }
            if (result.Evidence is { Count: > 0 })
            {
                H($"   graph walk ({result.PlatonicHopCount} hop(s)):");
                foreach (var ev in result.Evidence.OrderBy(x => x.Hop).Take(6))
                    H($"     hop {ev.Hop}: {ev.Concept}{(ev.RelatedConcept is not null ? $" -> {ev.RelatedConcept}" : "")}");
            }
        }
        else if (mech == "COMPOSED")
        {
            H($"HOW IT COMPOSED IT  ({InferenceTraceFormatter.ShapeOf(dp)}):");
            H("   the GRU plan head SELECTED the shape; the substrate EXECUTED it (the GRU only chooses):");
            if (result.Evidence is { Count: > 0 })
                foreach (var ev in result.Evidence.OrderBy(x => x.Hop).Take(6))
                    H($"   step {ev.Hop}: {ev.Concept}{(ev.RelatedConcept is not null ? $" -> {ev.RelatedConcept}" : "")}");
            else
                H($"   executed on the platonic blocks  ->  {result.Output}");
        }
        else
        {
            H("HOW: the NEURAL decoder produced this token-by-token — no platonic computation or retrieval drove it.");
            if (act.Anchors.Length > 0)
                H($"     (your input touched concepts [{string.Join(", ", act.Anchors)}] but the route head didn't trust the platonic path.)");
        }

        H("");
        // Batch-decode under one model-gate acquisition instead of one per token (see TokenTexts).
        var outToks = _runtime.TokenTexts(result.GeneratedTokens ?? []);
        H($"DECODE       emitted: {string.Join("  ", outToks)}");
        H($"             platonic bias on {result.AppliedBiasCount} token(s) (avg {result.AverageBiasMagnitude:F3}); hops {result.PlatonicHopCount}; chunks {result.ChunksGenerated}" +
          (result.PlatonicAssistFired > 0 ? $"; mid-gen assists {result.PlatonicAssistFired}/{result.PlatonicAssistInvocations}" : ""));
        H($"VERDICT      {mech} — {InferenceTraceFormatter.DescribeConfidence(result.PlatonicConfidence)}");

        box.Text = sb.ToString();
    }

    private void AppendToRepl(string text, Color color)
    {
        var box = GetControl<ListBox>("ReplOutput");
        if (box == null || box.IsDisposed)
            return;

        if (box.InvokeRequired)
        {
            box.Invoke(() => AppendToRepl(text, color));
            return;
        }

        // ListBox doesn't support per-item colors, but we can add the text as items
        var lines = text.Split(new[] { "\n", "\r\n" }, StringSplitOptions.None);
        foreach (var line in lines)
        {
            if (!string.IsNullOrEmpty(line))
            {
                box.Items.Add(line);
            }
        }

        // Auto-scroll to last item
        if (box.Items.Count > 0)
        {
            box.TopIndex = box.Items.Count - 1;
        }

        // Trim if needed
        if (box.Items.Count > MaxReplLogChars / 50)
        {
            var itemsToRemove = Math.Max(1, box.Items.Count - (TargetReplLogChars / 50));
            for (var i = 0; i < itemsToRemove; i++)
            {
                box.Items.RemoveAt(0);
            }
        }
    }

    private Panel CreateAutonomousConfigPanel()
    {
        var defaults = GetAutonomousResourceDefaults();
        var cpuThreads = defaults.CpuThreads;
        var creatorCount = defaults.CreatorCount;

        var panel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = SystemColors.Control,
            Padding = new Padding(12)
        };

        var layout = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.TopDown,
            AutoScroll = true,
            WrapContents = false
        };

        layout.Controls.Add(new Label
        {
            Text = "Autonomous Training",
            Font = new Font("Segoe UI", 14, FontStyle.Bold),
            Height = 30
        });
        layout.Controls.Add(new Label
        {
            Text = "Mixed-dataset rounds run continuously. Start tiny, reuse local cache, and widen data horizon only as loss improves.",
            Width = 300,
            Height = 52
        });
        layout.Controls.Add(new Label
        {
            Text = $"Lean default profile: {creatorCount} datasets, round budget {defaults.RoundBudget}, generation workers {defaults.GenerationConcurrency}/{cpuThreads} threads.",
            Width = 300,
            Height = 34,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });
        layout.Controls.Add(new Label { Height = 6 });
        layout.Controls.Add(new Label { Text = "Datasets for this run", Height = 24, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        layout.Controls.Add(new Label
        {
            Text = "Toggle each dataset on/off. Only checked datasets are used in this run (and live updates apply next round).",
            Width = 300,
            Height = 34,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });
        var creatorSelection = new CheckedListBox
        {
            Width = 300,
            Height = 168,
            CheckOnClick = true,
            BorderStyle = BorderStyle.FixedSingle,
            Name = "AutoCreatorSelectionList"
        };
        var creatorItems = ExampleCreatorRegistry.All
            .OrderBy(c => c.EstimatedComplexity)
            .ThenBy(c => c.Name, StringComparer.OrdinalIgnoreCase)
            .Select(c => c.Name)
            .ToArray();
        creatorSelection.Items.AddRange(creatorItems);
        for (var i = 0; i < creatorSelection.Items.Count; i++)
            creatorSelection.SetItemChecked(i, true);
        layout.Controls.Add(creatorSelection);
        var creatorToggleButtons = new FlowLayoutPanel
        {
            Width = 300,
            Height = 34,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };
        var checkAllBtn = new Button
        {
            Text = "Check all",
            Width = 146,
            Height = 28
        };
        checkAllBtn.Click += (_, _) =>
        {
            for (var i = 0; i < creatorSelection.Items.Count; i++)
                creatorSelection.SetItemChecked(i, true);
        };
        var uncheckAllBtn = new Button
        {
            Text = "Uncheck all",
            Width = 146,
            Height = 28
        };
        uncheckAllBtn.Click += (_, _) =>
        {
            for (var i = 0; i < creatorSelection.Items.Count; i++)
                creatorSelection.SetItemChecked(i, false);
        };
        creatorToggleButtons.Controls.Add(checkAllBtn);
        creatorToggleButtons.Controls.Add(uncheckAllBtn);
        layout.Controls.Add(creatorToggleButtons);

        layout.Controls.Add(new Label { Text = "Seed values (mostly first-round only)", Height = 24, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        layout.Controls.Add(new Label { Text = "Initial Sample Count:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 10000,
            Value = defaults.InitialSampleCount,
            Width = 300,
            Height = 32,
            Name = "AutoSampleCountInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Initial horizon width per dataset source. Keep this tiny for organic growth.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Initial Difficulty:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 100,
            Value = defaults.InitialDifficulty,
            Width = 300,
            Height = 32,
            Name = "AutoDifficultyInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Base context depth for next-token windows before adaptive feedback takes over.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Initial Epochs:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 64,
            Value = defaults.InitialEpochs,
            Width = 300,
            Height = 32,
            Name = "AutoEpochsInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "How many training passes each round uses; kept stable unless you restart.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Initial Train Count:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 128,
            Value = defaults.InitialTrainCount,
            Width = 300,
            Height = 32,
            Name = "AutoTrainCountInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Initial trained slice per dataset. Usually match sample count for deterministic coverage.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Height = 8 });
        layout.Controls.Add(new Label { Text = "Live controls (applied next round)", Height = 24, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        layout.Controls.Add(new Label { Text = "Max Rounds:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = int.MaxValue,
            Value = defaults.MaxRounds,
            Width = 300,
            Height = 32,
            Name = "AutoMaxRoundsInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "0 means unlimited rounds. Raise or lower this while running to extend or end sooner.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Loss Threshold:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 10,
            DecimalPlaces = 3,
            Increment = 0.005M,
            Value = 0.100M,
            Width = 300,
            Height = 32,
            Name = "AutoLossThresholdInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Target token loss used to decide when to push harder or back off.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Sample Bounds:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new FlowLayoutPanel
        {
            Width = 300,
            Height = 40,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            Controls =
            {
                new Label { Text = "Min", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 1,
                    Maximum = 10000,
                    Value = defaults.MinSampleCount,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMinSampleInput"
                },
                new Label { Text = "Max", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 1,
                    Maximum = 10000,
                    Value = defaults.MaxSampleCount,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMaxSampleInput"
                }
            }
        });
        layout.Controls.Add(new Label
        {
            Text = "Hard floor/ceiling for horizon growth. Low minimum keeps API demand small.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Train Bounds:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new FlowLayoutPanel
        {
            Width = 300,
            Height = 40,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            Controls =
            {
                new Label { Text = "Min", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 1,
                    Maximum = 256,
                    Value = defaults.MinTrainCount,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMinTrainInput"
                },
                new Label { Text = "Max", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 1,
                    Maximum = 256,
                    Value = defaults.MaxTrainCount,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMaxTrainInput"
                }
            }
        });
        layout.Controls.Add(new Label
        {
            Text = "Per-dataset trained slice bounds; grows as mastery improves.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Max Difficulty:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 100,
            Value = defaults.MaxDifficulty,
            Width = 300,
            Height = 32,
            Name = "AutoMaxDifficultyInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Upper cap for planner-selected difficulty, even when loss is low.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Round Train Budget:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 4096,
            Value = defaults.RoundBudget,
            Width = 300,
            Height = 32,
            Name = "AutoRoundTrainBudgetInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Total train slots split across datasets each round (higher = more throughput).",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Generation Concurrency:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = Math.Max(32, defaults.CpuThreads * 2),
            Value = defaults.GenerationConcurrency,
            Width = 300,
            Height = 32,
            Name = "AutoGenerationConcurrencyInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Controls parallel dataset generation workers per round.",
            Width = 300,
            Height = 28,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Height = 10 });

        var startBtn = new Button
        {
            Text = "▶ Start Gym",
            Width = 300,
            Height = 40,
            BackColor = Color.FromArgb(51, 153, 102),
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 11, FontStyle.Bold),
            Name = "AutoTrainBtn"
        };
        startBtn.Click += (s, e) => _ = StartGymAsync();
        layout.Controls.Add(startBtn);

        var stopBtn = new Button
        {
            Text = "⏸ Pause Gym",
            Width = 300,
            Height = 40,
            BackColor = Color.FromArgb(204, 51, 51),
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 11, FontStyle.Bold),
            Name = "AutoStopBtn",
            Enabled = false
        };
        stopBtn.Click += (s, e) => _ = StopGymAsync();
        layout.Controls.Add(stopBtn);

        panel.Controls.Add(layout);
        return panel;
    }

    private AutonomousResourceDefaults GetAutonomousResourceDefaults()
    {
        var cpuThreads = Math.Max(1, Environment.ProcessorCount);
        var creatorCount = Math.Max(1, ExampleCreatorRegistry.All.Count);
        var generationConcurrency = Math.Clamp(Math.Min(cpuThreads, creatorCount), 1, 16);
        var maxRounds = Math.Clamp(Math.Max(12, cpuThreads * 2), 12, 64);
        var initialEpochs = 2;
        var initialSampleCount = Math.Clamp(Math.Max(4, cpuThreads / 2), 4, 32);
        var initialTrainCount = Math.Clamp(Math.Max(4, cpuThreads / 2), 4, 32);
        var initialDifficulty = 1;
        var maxSampleCount = 128;
        var maxTrainCount = Math.Clamp(Math.Max(16, cpuThreads * 2), 16, 256);
        var maxDifficulty = Math.Clamp(Math.Max(12, cpuThreads / 2), 12, 64);
        var roundBudget = Math.Clamp(cpuThreads * 6, creatorCount, 256);

        if (GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalMb, out var freeMb))
        {
            var usableMb = freeMb > 0 ? freeMb : totalMb;
            if (usableMb > 0)
            {
                var gpuScale = Math.Clamp(usableMb / 1024, 1, 12);
                initialSampleCount = Math.Clamp(Math.Max(initialSampleCount, gpuScale * 2), 4, 48);
                initialTrainCount = Math.Clamp(Math.Max(initialTrainCount, gpuScale * 2), 4, 48);
                initialDifficulty = Math.Clamp(Math.Max(initialDifficulty, gpuScale / 2), 1, 8);
                maxSampleCount = Math.Clamp(usableMb / 64, 32, 512);
                maxTrainCount = Math.Clamp(Math.Max(maxTrainCount, usableMb / 192), 16, 256);
                maxDifficulty = Math.Clamp(Math.Max(maxDifficulty, usableMb / 768), 12, 100);
                roundBudget = Math.Clamp(Math.Max(roundBudget, cpuThreads * gpuScale), creatorCount, 512);
                initialEpochs = Math.Clamp(Math.Max(2, gpuScale / 2), 2, 8);
            }
        }

        return new AutonomousResourceDefaults(
            CpuThreads: cpuThreads,
            CreatorCount: creatorCount,
            MaxRounds: maxRounds,
            InitialSampleCount: initialSampleCount,
            InitialDifficulty: initialDifficulty,
            InitialTrainCount: initialTrainCount,
            InitialEpochs: initialEpochs,
            MinSampleCount: 1,
            MaxSampleCount: maxSampleCount,
            MinTrainCount: 1,
            MaxTrainCount: maxTrainCount,
            MaxDifficulty: maxDifficulty,
            RoundBudget: roundBudget,
            GenerationConcurrency: generationConcurrency);
    }

    private sealed record AutonomousResourceDefaults(
        int CpuThreads,
        int CreatorCount,
        int MaxRounds,
        int InitialSampleCount,
        int InitialDifficulty,
        int InitialTrainCount,
        int InitialEpochs,
        int MinSampleCount,
        int MaxSampleCount,
        int MinTrainCount,
        int MaxTrainCount,
        int MaxDifficulty,
        int RoundBudget,
        int GenerationConcurrency);

    private Panel CreateAutonomousResultsPanel()
    {
        var panel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.White,
            Padding = new Padding(12)
        };

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(0)
        };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 72));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 28));

        layout.Controls.Add(new Label
        {
            Text = "Autonomous Training Results",
            Font = new Font("Segoe UI", 14, FontStyle.Bold),
            Dock = DockStyle.Fill
        }, 0, 0);

        layout.Controls.Add(new ListBox
        {
            Dock = DockStyle.Fill,
            Font = new Font("Consolas", 9),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.FromArgb(200, 220, 200),
            Name = "AutoOutputBox",
            SelectionMode = SelectionMode.MultiExtended
        }, 0, 1);

        layout.Controls.Add(new TextBox
        {
            Text = "Rounds: 0 | Last loss: n/a | Success: n/a | Data mix: n/a | Skipped: 0",
            Font = new Font("Consolas", 10),
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            WordWrap = false,
            Name = "AutoStatsText"
        }, 0, 2);

        panel.Controls.Add(layout);
        return panel;
    }

    private GenesisAutonomousTrainingRequest? TryBuildAutonomousTrainingRequest()
    {
        var defaults = GetAutonomousResourceDefaults();
        var maxRounds = GetControl<NumericUpDown>("AutoMaxRoundsInput");
        var sampleCount = GetControl<NumericUpDown>("AutoSampleCountInput");
        var difficulty = GetControl<NumericUpDown>("AutoDifficultyInput");
        var epochs = GetControl<NumericUpDown>("AutoEpochsInput");
        var trainCount = GetControl<NumericUpDown>("AutoTrainCountInput");
        var lossThreshold = GetControl<NumericUpDown>("AutoLossThresholdInput");
        var minSample = GetControl<NumericUpDown>("AutoMinSampleInput");
        var maxSample = GetControl<NumericUpDown>("AutoMaxSampleInput");
        var minTrain = GetControl<NumericUpDown>("AutoMinTrainInput");
        var maxTrain = GetControl<NumericUpDown>("AutoMaxTrainInput");
        var maxDifficulty = GetControl<NumericUpDown>("AutoMaxDifficultyInput");
        var roundBudget = GetControl<NumericUpDown>("AutoRoundTrainBudgetInput");
        var generationConcurrency = GetControl<NumericUpDown>("AutoGenerationConcurrencyInput");
        var creatorSelection = GetControl<CheckedListBox>("AutoCreatorSelectionList");
        var enabledCreators = creatorSelection is null
            ? ExampleCreatorRegistry.All.Select(c => c.Name).ToArray()
            : creatorSelection.CheckedItems
                .Cast<object>()
                .Select(item => item?.ToString())
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .Select(name => name!)
                .ToArray();
        if (enabledCreators.Length == 0)
        {
            MessageBox.Show(
                "Select at least one dataset in the autonomous dataset checklist before starting.",
                "Autonomous Training",
                MessageBoxButtons.OK,
                MessageBoxIcon.Warning);
            return null;
        }

        return new GenesisAutonomousTrainingRequest(
            MaxRounds: (int)(maxRounds?.Value ?? 0),
            InitialSampleCount: (int)(sampleCount?.Value ?? defaults.InitialSampleCount),
            InitialDifficulty: (int)(difficulty?.Value ?? defaults.InitialDifficulty),
            InitialEpochs: (int)(epochs?.Value ?? defaults.InitialEpochs),
            InitialTrainCount: (int)(trainCount?.Value ?? defaults.InitialTrainCount),
            LossThreshold: (double)(lossThreshold?.Value ?? 0.100M),
            MinSampleCount: (int)(minSample?.Value ?? defaults.MinSampleCount),
            MaxSampleCount: (int)(maxSample?.Value ?? defaults.MaxSampleCount),
            MinTrainCount: (int)(minTrain?.Value ?? defaults.MinTrainCount),
            MaxTrainCount: (int)(maxTrain?.Value ?? defaults.MaxTrainCount),
            MaxDifficulty: (int)(maxDifficulty?.Value ?? defaults.MaxDifficulty),
            RoundTrainBudget: (int)(roundBudget?.Value ?? defaults.RoundBudget),
            MaxGenerationConcurrency: (int)(generationConcurrency?.Value ?? defaults.GenerationConcurrency),
            EnabledCreators: enabledCreators);
    }

    private async Task StartAutonomousTraining()
    {
        var request = TryBuildAutonomousTrainingRequest();
        if (request is null)
        {
            MessageBox.Show("Please configure the autonomous training settings.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var startBtn = GetControl<Button>("AutoTrainBtn");
        var stopBtn = GetControl<Button>("AutoStopBtn");
        if (startBtn != null) startBtn.Enabled = false;
        if (stopBtn != null) stopBtn.Enabled = true;

        _autonomousTrainingCts = new CancellationTokenSource();

        try
        {
            var roundsLabel = request.MaxRounds <= 0 ? "∞" : request.MaxRounds.ToString();
            AppendOutput($"[auto] starting: rounds={roundsLabel} sample={request.InitialSampleCount} train={request.InitialTrainCount} difficulty={request.InitialDifficulty}");
            AppendOutput($"[auto] datasets enabled: {string.Join(", ", request.EnabledCreators ?? [])}");
            AppendOutput("[auto] device policy: CUDA training when available with CPU fallback, GPU inference (device 0, A3000 preferred)");
            
            // Estimate and display GPU sizing for this training session using live VRAM.
            var debugLines = new List<string>();
            var hasCuda = GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalVramMb, out var freeVramMb);
            var sizingVramMb = hasCuda ? (freeVramMb > 0 ? freeVramMb : totalVramMb) : 4096;
            var sizingTargetUtil = 0.72;
            var sizingReserveMb = 1536;
            var sizingHiddenCap = GpuCapacityPlanner.ResolveTrainingHiddenCap(sizingVramMb);
            AppendOutput($"[auto] training headroom cap: {sizingHiddenCap}");
            var estimatedHidden = GpuCapacityPlanner.EstimateHiddenSizeForInferenceOnly(
                new[] { new GenesisExample("sample", "output") },
                routeCount: 8,
                vramMb: sizingVramMb,
                targetUtilization: sizingTargetUtil,
                reserveVramMb: sizingReserveMb,
                debugOutput: line => debugLines.Add(line),
                maxHiddenSize: sizingHiddenCap);
            foreach (var line in debugLines)
                AppendOutput(line);

            // Extra guard for autonomous rounds on constrained VRAM.
            if (hasCuda)
            {
                if (estimatedHidden > sizingHiddenCap)
                {
                    AppendOutput(
                        $"[auto] hidden cap applied for training headroom: {estimatedHidden} → {sizingHiddenCap} (usable_vram_mb={sizingVramMb})");
                    estimatedHidden = sizingHiddenCap;
                }
            }
            AppendOutput(
                $"[auto] sizing profile: target_util={sizingTargetUtil:F2} reserve_mb={sizingReserveMb} hidden_cap={sizingHiddenCap} usable_vram_mb={sizingVramMb}");
             
            // CRITICAL: Apply the estimated hidden size BEFORE starting training
            AppendOutput($"[auto] applying hidden size: {_runtime.HiddenSize} → {estimatedHidden}");
            _runtime.EnsureHiddenSize(estimatedHidden);
            AppendOutput($"[auto] model ready: hidden={_runtime.HiddenSize}");
            
            AppendOutput("[auto] live controls update next round: max rounds, loss threshold, bounds, max difficulty, round budget, and generation concurrency.");
            // Run entire training pipeline on ThreadPool with reduced priority to prevent UI thread starvation
            var run = await RunLowPriorityTrainingAsync(
                async () => await _runtime.TrainAutonomousAsync(
                    request,
                    _autonomousTrainingCts.Token,
                    AppendOutput,
                    baseRequest => CaptureLiveAutonomousRequest(baseRequest),
                    onRoundProgress: payload => HandleAutonomousTrainingProgress((GenesisAutonomousTrainingEventPayload)payload)),
                _autonomousTrainingCts.Token);
            var final = run.FinalReport;
            if (final is not null)
            {
                AppendOutput($"[auto] complete: rounds={run.Rounds.Count} final_loss={final.AverageLoss.TokenLoss:F4}");
                HandleAutonomousTrainingProgress(new GenesisAutonomousTrainingEventPayload(
                    Round: run.Rounds.Count,
                    StepName: "Complete",
                    Dataset: run.Rounds.LastOrDefault()?.CreatorName ?? "mixed",
                    Loss: final.AverageLoss.TokenLoss,
                    ExampleSuccessRate: final.ExampleSuccessRate,
                    SamplesTrained: final.ExampleCount,
                    ElapsedMs: 0,
                    SkippedCorrectExampleCount: final.SkippedCorrectExampleCount,
                    PromptAnswerExampleCount: final.PromptAnswerExampleCount,
                    WindowedTextExampleCount: final.WindowedTextExampleCount,
                    CreatorSummary: BuildAutonomousCreatorSummary(final)));
            }
            else
            {
                AppendOutput("[auto] complete: no report returned");
            }
        }
        catch (OperationCanceledException)
        {
            AppendOutput("[auto] stopped by user.");
        }
        catch (Exception ex)
        {
            AppendExceptionReport("Autonomous training failed", ex, boxName: "AutoOutputBox");
        }
        finally
        {
            if (startBtn != null) startBtn.Enabled = true;
            if (stopBtn != null) stopBtn.Enabled = false;
            _autonomousTrainingCts?.Dispose();
            _autonomousTrainingCts = null;
        }
    }

    /// <summary>
    /// Run async operation on a low-priority thread to prevent UI thread starvation during long training runs.
    /// Uses BelowNormal priority to allow UI messages to be processed even during CPU-intensive work.
    /// </summary>
    private async Task<T> RunLowPriorityTrainingAsync<T>(Func<Task<T>> workAsync, CancellationToken ct)
    {
        return await Task.Run(async () =>
        {
            // Lower this thread's priority so UI thread gets CPU time
            var originalPriority = Thread.CurrentThread.Priority;
            try
            {
                Thread.CurrentThread.Priority = ThreadPriority.BelowNormal;
                return await workAsync();
            }
            finally
            {
                Thread.CurrentThread.Priority = originalPriority;
            }
        }, ct);
    }

    private void StopAutonomousTraining()
    {
        if (_autonomousTrainingCts is null || _autonomousTrainingCts.IsCancellationRequested)
            return;

        AppendOutput("[auto] stop requested: finishing current round...");
        _autonomousTrainingCts.Cancel();
    }

    // Friendly checkbox label for a gym muscle.
    private static string GymSkillLabel(GenesisNova.Train.GymSkill s) => s switch
    {
        GenesisNova.Train.GymSkill.Synonym => "synonym (a synonym for X)",
        GenesisNova.Train.GymSkill.Category => "category (what kind of thing is X)",
        GenesisNova.Train.GymSkill.NumberWord => "number ↔ word",
        GenesisNova.Train.GymSkill.Add => "add (a + b)",
        GenesisNova.Train.GymSkill.Subtract => "subtract (a - b)",
        GenesisNova.Train.GymSkill.Multiply => "multiply (a x b)",
        GenesisNova.Train.GymSkill.FoldAdd => "fold-add (a + b + c)",
        GenesisNova.Train.GymSkill.FoldMultiply => "fold-mul (a x b x c)",
        GenesisNova.Train.GymSkill.Expression => "expression (a x b + c)",
        GenesisNova.Train.GymSkill.Predicate => "predicate (compare)",
        GenesisNova.Train.GymSkill.WordedAdd => "worded add (what is a plus b)",
        GenesisNova.Train.GymSkill.ArithToWord => "arith → word (a plus b in words)",
        GenesisNova.Train.GymSkill.FunctionInduction => "function induction (few-shot)",
        _ => s.ToString(),
    };

    // ── GYM (in-app skill trainer; replaces the standalone ClaudeMemory daemon) ─────────────────────────────────
    private void StartGym()
    {
        if (_gymCts is not null && !_gymCts.IsCancellationRequested) return; // already running
        _gymCts?.Dispose();
        _gymCts = new CancellationTokenSource();
        var barNum = GetControl<NumericUpDown>("GymBar"); if (barNum != null) _gymBar = (double)barNum.Value;
        var tpcNum = GetControl<NumericUpDown>("GymTrainPerCycle"); if (tpcNum != null) _gymTrainPerCycle = (int)tpcNum.Value;
        var thrNum = GetControl<NumericUpDown>("GymThrottle"); if (thrNum != null) _gymThrottlePct = (int)thrNum.Value;

        // Build the ENABLED curricula from the checkboxes; the orchestrator runs them as one composite.
        Directory.CreateDirectory(_gymStateDir);
        var children = new List<ITrainingCurriculum>();
        var gymChildren = new List<GymTrainer>(); // ONE per enabled muscle — each its OWN mastery gate + level
        // PREBAKE — the run-first LANGUAGE curriculum: warms the substrate with the SCHEMAS of English (function words →
        // SVO → questions → nesting → multi-sentence), the prerequisite that gates the Merge parse / fact memory /
        // retrieval. Training DATA (real words + nonce salt), not a dispatch list — the structure is learned distributionally.
        if (GetControl<CheckBox>("CurPrebakeLanguage")?.Checked ?? false)
        {
            // WARM the function-word signal from BOTH a REAL TEXT CORPUS (Wikipedia — natural context breadth, the signal
            // warms and HOLDS without per-token tuning) AND the synthetic L1 prebake (clean, dense function-word frames with
            // nonce salt). The two are complementary foundations and both grade by the SAME space property (function-word
            // separation), so they reinforce the prerequisite the Merge parse / fact memory / learned cues (∘qst …) rest on.
            var synthFoundation = new PrebakeLanguageCurriculum(trainPerCycle: _gymTrainPerCycle, seed: 7);
            var corpusFoundation = new CorpusWarmCurriculum(trainPerCycle: _gymTrainPerCycle);
            try { _inspectGlue = synthFoundation.Glue.Take(14).ToArray(); _inspectContent = synthFoundation.SampleContent(12).ToArray(); } catch { } // Inspect-tab probe words = actual trained vocab
            children.Add(new FoundationBlendCurriculum(synthFoundation, corpusFoundation, trainPerCycle: _gymTrainPerCycle, syntheticFraction: 0.0));
            AppendOutput("[train] PREBAKE: warming function-word recognition from 100% REAL CORPUS (Wikipedia) — natural text gives function-word breadth for free, no per-word synthetic hand-tuning (synthetic kept only for the inspect-probe vocab)");
        }
        // CORPUS PREDICTION (masked cloze) — the genesis-native "base-model" objective (PLATONIC_MIND: reasoning = the
        // field relaxing its own surprise): predict a held-out CONTENT word from its context, drive the error back via the
        // substrate's own FineEditFromExample coupling, so corpus text actively BUILDS knowledge (NextTrainBatch emits the
        // masked skeleton→target pairs; SelfAssess reports the HELD-OUT cloze curve). Foundation-tier (warms before the
        // skills, and the retained function-word skeleton also warms recognition — could subsume the passive prebake).
        // EXPERIMENTAL, default-OFF (opt-in) — same shared Wikipedia source CorpusWarm uses. See [[nova-corpus-masked-cloze]].
        if (GetControl<CheckBox>("CurCorpusPrediction")?.Checked ?? false)
        {
            children.Add(new CorpusPredictionCurriculum(CorpusWarmCurriculum.SharedCorpus, trainPerCycle: _gymTrainPerCycle));
            AppendOutput("[train] corpus-prediction (masked cloze): predict a held-out content word from context; held-out cloze accuracy is the climb signal (self-supervised base-model objective on the substrate)");
        }
        if (GetControl<CheckBox>("CurCreators")?.Checked ?? false)
        {
            children.AddRange(CreatorUnit.SkillLadder(trainCount: _gymTrainPerCycle));   // each creator = one focusable trainer
            AppendOutput($"[train] creators skill ladder: {string.Join(", ", ExampleCreatorRegistry.All.Select(c => c.Name))}");
        }
        // GYM MUSCLES — each TICKED procedural skill (flat peers; no master toggle). EACH becomes its OWN curriculum
        // unit → its OWN mastery gate + independent level (the orchestrator grades + gates each unit separately;
        // Focused drives one to mastery then the next, Composite trains all but still levels each). Own level file each.
        var enabledSkills = Enum.GetValues<GenesisNova.Train.GymSkill>()
            .Where(s => GetControl<CheckBox>("GymSkill_" + s)?.Checked ?? false).ToList();
        foreach (var skill in enabledSkills)
        {
            var lvlPath = Path.Combine(_gymStateDir, "gym-" + skill.ToString().ToLowerInvariant() + "-level.txt");
            var startLevel = 1;
            try { if (File.Exists(lvlPath) && int.TryParse(File.ReadAllText(lvlPath).Trim(), out var lv) && lv >= 1) startLevel = lv; } catch { }
            var child = new GymTrainer(startLevel, skills: new[] { skill }) { MasteryBar = _gymBar, TrainPerCycle = _gymTrainPerCycle };
            children.Add(child);
            gymChildren.Add(child);
        }
        if (enabledSkills.Count > 0)
            AppendOutput($"[train] gym muscles ({enabledSkills.Count}, each its own gate): {string.Join(", ", enabledSkills)}");
        if (GetControl<CheckBox>("CurMemCode")?.Checked ?? false)
        {
            var mem = GetControl<TextBox>("MemPath")?.Text?.Trim() ?? string.Empty;
            var code = GetControl<TextBox>("CodePath")?.Text?.Trim() ?? string.Empty;
            try
            {
                var mc = new MemoryCodeCurriculum(mem, code, trainPerCycle: _gymTrainPerCycle);
                children.Add(mc);
                AppendOutput($"[train] memory+code: {mc.Stats.Cues} cues, {mc.Stats.Edges} edges, {mc.Stats.Probes} probes");
            }
            catch (Exception ex) { AppendOutput($"[train] memory+code load failed: {ex.Message}"); }
        }
        // FUNDAMENTALS — learned dispatch signals the gym muscles rely on, each warmed by its OWN curriculum (no
        // hardcoded list). OP-CUES: "the product of x and y" resolves × by a LEARNED cue→op relation (the engine has no
        // TryOpCue); clean worded frames feed LearnArithmeticCue → sum/difference/product/quotient. See OpCueCurriculum.
        if (GetControl<CheckBox>("CurOpCues")?.Checked ?? false)
        {
            children.Add(new OpCueCurriculum(trainPerCycle: _gymTrainPerCycle));
            AppendOutput("[train] op-cues: worded arithmetic synonyms (learns sum/difference/product/quotient → op word, no hardcoded list)");
        }
        // NUMBER-WORDS: clean digit↔word pairs plant the learned number-word LEXICON atoms (de-hardcoding #5), the rest
        // composing by universal place value (replaces the hardcoded NumberWordVocabulary codec).
        if (GetControl<CheckBox>("CurNumberWords")?.Checked ?? false)
        {
            children.Add(new NumberWordCurriculum(trainPerCycle: _gymTrainPerCycle));
            AppendOutput("[train] number-words: clean digit↔word (learns the lexicon atoms → no hardcoded codec)");
        }
        // NAV-REASONING (M4): grows a multi-hop is-a taxonomy + emits level-cue text frames as DATA (the cue stays
        // LEARNED from answer graph-depth) so the gym self-teaches the navigator's reasoning, and plants a HELD-OUT
        // member set the sampler never trains on. The held-out resolve/accuracy curve (logged as [nav-heldout]) is the
        // M4 acceptance signal — it must CLIMB overnight. Needs the navigator toggle ON to move (it trains the walker).
        if (GetControl<CheckBox>("CurNavReasoning")?.Checked ?? false)
        {
            children.Add(new NavReasoningCurriculum(_runtime, trainPerCycle: _gymTrainPerCycle));
            AppendOutput("[train] nav-reasoning: is-a taxonomy + learned level cues; navigator HELD-OUT curve in [nav-heldout] (ensure 'Train Navigator' is ticked)");
        }
        var personalityOn = GetControl<CheckBox>("CurPersonality")?.Checked ?? false;
        PersonalityCurriculum? persona = null;
        if (personalityOn)
        {
            // The persona is SEEDED as reply CHUNKS, not decode-trained: in the conscious field the GRU decoder is
            // bypassed (TryFieldRespond retrieves a reply as a whole chunk), and decoding a reply token-by-token only
            // builds stray cue→WORD edges that crowd the chunk out of retrieval. Seed once + turn the talk route on, and
            // add it as a PROBE-ONLY unit below so it shows in the train list and gets graded each cycle (its correct
            // probes reinforce the chunks). See [[nova-talk-by-chunk]].
            try
            {
                // Restore the persona's level across restarts (its chunks persist in the space checkpoint, so its
                // level should too — otherwise it shows L1 against a space that already knows the full repertoire).
                var personaStart = 1;
                var personaLvlPath = Path.Combine(_gymStateDir, "personality-level.txt");
                try { if (File.Exists(personaLvlPath) && int.TryParse(File.ReadAllText(personaLvlPath).Trim(), out var pl) && pl >= 1) personaStart = pl; } catch { }
                persona = new PersonalityCurriculum(trainPerCycle: _gymTrainPerCycle, startLevel: personaStart);
                _runtime.SeedConversationalChunks(persona.Repertoire);
                _runtime.SetConversationalMode(true);
                AppendOutput("[train] personality (rude chatbot): reply chunks SEEDED + talk route ON (graded as a probe-only unit, not decode-trained)");
            }
            catch (Exception ex) { AppendOutput($"[train] personality seed failed: {ex.Message}"); persona = null; }
        }
        else
        {
            try { _runtime.SetConversationalMode(false); } catch { } // non-chat training stays byte-identical
        }
        // PREREQUISITE-FIRST ORDER: the prebake + fundamentals warm the LEARNED mechanisms the gym muscles then rely on
        // — the function-word signal (prebake), the op-cue→op relations, the number-word lexicon. Float them to the
        // FRONT so dependent muscles never train against an un-warmed signal/cue/lexicon. FocusedCurriculum introduces
        // never-seen units in LIST order (see NextWeakest), so this prefix IS that order.
        static int BootstrapRank(ITrainingCurriculum c) => c switch
        {
            FoundationBlendCurriculum => 0, // prerequisite — warms function-word recognition (100% real corpus; synthetic kept only for inspect-probe vocab)
            CorpusWarmCurriculum => 0,      // (if used standalone) same prerequisite role
            PrebakeLanguageCurriculum => 0, // (if used standalone) same prerequisite role
            CorpusPredictionCurriculum => 0, // masked-cloze base-model objective — foundation-tier (warms before the skills, after the prebake by list order)
            OpCueCurriculum => 1,                // worded arithmetic synonyms → op
            NumberWordCurriculum => 2,           // digit↔word lexicon atoms
            NavReasoningCurriculum => 2,         // taxonomy + level cues (a fundamentals-tier foundation)
            _ => 3,                              // the gym muscles + anything else, in their existing order
        };
        children = children.OrderBy(BootstrapRank).ToList(); // OrderBy is STABLE: the prerequisite/fundamentals float to front in rank order; the rest keep their relative order

        // FOUNDATION ORDER. Personality is SEEDED above at startup — established before cycle 1, the earliest possible
        // (retrieval chunks, NOT gradient-trained: decode-training breaks chunk retrieval, see [[nova-talk-by-chunk]]).
        // Then the trained foundation leads with the prebake (the function-word signal everything parses through), then
        // the op-cue and number-word bootstraps, then the gym skills that depend on all of the above.
        AppendOutput("[train] order: personality SEEDED first (identity) → prebake function-words → op-cues → number-words → gym skills");

        if (children.Count == 0)
        {
            // Personality alone has nothing to gradient-TRAIN (it's retrieval) — it's seeded + chat-ready, but a probe
            // loop with no training is idle, so don't start one. Tick a gym skill to also train alongside it.
            AppendOutput(persona is not null
                ? "[train] personality is SEEDED + chat-ready. Tick a Gym skill to run the training loop alongside it."
                : "[train] no curriculum enabled — tick Gym and/or Memory+Code.");
            _gymCts.Cancel();
            _gymLoopTask = null; // nothing launched — clear any stale tracked loop so a later start isn't blocked
            return;
        }
        _gym = gymChildren.FirstOrDefault(); // for /status (first muscle; each muscle persists its own level below)

        SetGymButtons(running: true);
        // Multiple trainers: FOCUSED with REHEARSAL (FocusedCurriculum) by default — it focuses one muscle, hands
        // off via EXHAUSTION (gym muscles are UNBOUNDED so they never "master"; a low focusBudget makes the handoff
        // prompt instead of trapping on the first), and the already-touched muscles RIDE ALONG as light replay,
        // converging to full interleave once every muscle has had its turn. That built-in rehearsal is what keeps
        // the thin SHARED heads (op-classifier, decode) from mode-collapsing toward the current skill — the
        // catastrophic forgetting a plain rotate-one-at-a-time scheduler caused. Untick → CompositeCurriculum
        // (every muscle, full batch, every cycle). One trainer → just run it.
        // RESUMING = any muscle restored above level 1 (it was trained before). Then mark all units introduced so the
        // FULL trained mix is probed from cycle 1 — otherwise the reported accuracy starts at just the first focus
        // muscle and only climbs back as the rotation re-introduces them (reads as "starts lower than it ended").
        // RESUMING = the model carries PRIOR TRAINING — either a gym muscle restored above level 1, OR a resumed
        // NON-EMPTY platonic space (AutoResume loaded a checkpoint with substantial nodes). The latter matters during a
        // FOUNDATION-ONLY stage (no gym skills yet, all muscles at level 1) where the muscle check alone reads false.
        var spaceNodes = (_runtime?.State?.Memory as DialecticalSpace)?.NodeCount ?? 0;
        var resuming = gymChildren.Any(g => g.Level > 1) || spaceNodes >= 256;
        ITrainingCurriculum curriculum;
        if (children.Count == 1)
            curriculum = children[0];
        else if (GetControl<CheckBox>("CurFocused")?.Checked ?? true)
        {
            var focused = new FocusedCurriculum(children, focusBudget: 8, resuming: resuming); // focus + rider-replay; prompt handoff for unbounded muscles
            // THE FIX: assess each unit's CURRENT mastery against the just-resumed model BEFORE the first cycle, so an
            // already-warm foundation (a mastered prebake) does NOT re-claim focus ahead of a newly-added weak skill
            // (e.g. op-cues just enabled). The mastered foundation rides as bounded rehearsal; the weakest UNMASTERED
            // unit leads. A genuinely-COLD model (nothing assessable scores) still trains the foundation first.
            if (_runtime is not null) focused.SeedFromAssessment(_runtime);
            curriculum = focused;
        }
        else
            curriculum = new CompositeCurriculum(children);
        if (persona is not null)
            curriculum = new ProbeAlongsideCurriculum(curriculum, persona); // graded each cycle → shows in the list, kept alive
        var ct = _gymCts.Token;
        // Read the navigator toggle on the UI thread NOW (captured into the loop so the background thread never touches
        // a WinForms control). Default ON so an overnight run trains the platonic walker.
        var trainNavigator = GetControl<CheckBox>("CurTrainNavigator")?.Checked ?? true;
        // TRACK the loop Task so a stop/restart can AWAIT it actually drain before the final save (no fire-and-forget).
        _gymLoopTask = RunLowPriorityTrainingAsync(async () => { await GymLoopAsync(curriculum, gymChildren, persona, trainNavigator, ct); return 0; }, ct);
    }

    // ── GYM LIFECYCLE (gate-serialized so start/stop/restart never overlap and a start can never race ahead of a
    // pending stop's final save). All three acquire _gymLifecycleGate; the HTTP control endpoints AWAIT these so the
    // response only returns once the requested state (and, for stop, the on-disk save) is reached. ──────────────────

    // CLEAN STOP: cancel the loop, AWAIT it to fully finish, then AWAIT one final SaveAsync. When this returns the
    // model IS on disk. Caller must hold _gymLifecycleGate.
    private async Task StopGymCoreAsync()
    {
        var cts = _gymCts;
        var loop = _gymLoopTask;
        if (cts is not null && !cts.IsCancellationRequested)
        {
            AppendOutput("[gym] stop requested — draining loop, then saving...");
            try { cts.Cancel(); } catch { }
        }
        await InvokeOnUiAsync(() => SetGymButtons(running: false));
        // (a) AWAIT the loop Task to actually finish shutting down (it observes the cancellation and exits RunAsync).
        if (loop is not null)
        {
            try { await loop; }
            catch (OperationCanceledException) { }
            catch (Exception ex) { AppendOutput($"[gym] loop ended with error: {ex.Message}"); }
        }
        _gymLoopTask = null;
        // (b) AWAIT one final save through the model gate — guaranteed on disk before we return.
        try { await _runtime.SaveAsync(_runtime.AutoCheckpointPath); AppendOutput("[gym] model saved to disk."); }
        catch (Exception ex) { AppendOutput($"[gym] final save failed: {ex.Message}"); }
    }

    // Public stop entry (HTTP /gym/stop|/gym/pause + the Pause button). Returns only after drain + save complete.
    private async Task StopGymAsync()
    {
        await _gymLifecycleGate.WaitAsync();
        try { await StopGymCoreAsync(); }
        finally { _gymLifecycleGate.Release(); }
    }

    // Public start entry (HTTP /gym/start|/gym/resume + the Start button). Waits for any in-flight stop+save to fully
    // release the gate FIRST (so we never start a new loop before the old one is saved/drained), then builds the
    // curricula + launches the loop on the UI thread. StartGym's own guard makes a start on an already-running gym a no-op.
    private async Task StartGymAsync()
    {
        await _gymLifecycleGate.WaitAsync();
        try { await InvokeOnUiAsync(StartGym); }
        finally { _gymLifecycleGate.Release(); }
    }

    // RESTART: clean stop (awaits the final save) → start fresh with the CURRENT checkbox config (for safe reconfiguration).
    private async Task RestartGymAsync()
    {
        await _gymLifecycleGate.WaitAsync();
        try
        {
            await StopGymCoreAsync();
            await InvokeOnUiAsync(StartGym);
        }
        finally { _gymLifecycleGate.Release(); }
    }

    private void SetGymButtons(bool running)
    {
        var startBtn = GetControl<Button>("AutoTrainBtn"); if (startBtn != null) startBtn.Enabled = !running;
        var stopBtn = GetControl<Button>("AutoStopBtn"); if (stopBtn != null) stopBtn.Enabled = running;
    }

    // Marshal a UI-thread action/func and AWAIT it (WinForms has no awaitable Invoke). Runs inline if already on the UI
    // thread; otherwise BeginInvoke + a TaskCompletionSource so the caller's await completes when the UI work is done.
    private Task InvokeOnUiAsync(Action action) => InvokeOnUiAsync(() => { action(); return true; });

    private Task<T> InvokeOnUiAsync<T>(Func<T> func)
    {
        if (!InvokeRequired) return Task.FromResult(func());
        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        BeginInvoke(new Action(() => { try { tcs.SetResult(func()); } catch (Exception ex) { tcs.SetException(ex); } }));
        return tcs.Task;
    }

    private bool GymRunning => _gymCts is not null && !_gymCts.IsCancellationRequested;

    /// <summary>
    /// The gym training loop — hosted IN the app (this is what the retired ClaudeMemory daemon did). Trains the
    /// shared <see cref="GymTrainer"/> procedural curriculum against the in-app runtime, mastery-gating an
    /// UNBOUNDED difficulty level. Runs on a low-priority background thread; renders to the training panels.
    /// REPL queries share the runtime safely via its internal model-ops gate.
    /// </summary>
    private async Task GymLoopAsync(ITrainingCurriculum curriculum, IReadOnlyList<GymTrainer> gymChildren, PersonalityCurriculum? persona, bool trainNavigator, CancellationToken ct)
    {
        AppendOutput($"[train] started: {curriculum.Name} (model {_gymStateDir}){(trainNavigator ? " | navigator: training each cycle" : "")}");

        // The unified MODULAR orchestrator drives the chosen curriculum (gym, memory+code, or a composite).
        var orchestrator = new GenesisModularTrainingOrchestrator();
        var options = new GenesisModularTrainingOrchestrator.Options
        {
            MasteryBar = _gymBar,
            RequirePlatonic = true,
            ThrottlePercent = () => _gymThrottlePct,   // live 0..500% backoff knob
            WorkDir = _gymStateDir,
            TrainOnFailureOnly = true,                 // don't reinforce already-correct answers; train only failures
        };
        var lastLevels = gymChildren.ToDictionary(g => g, g => g.Level);
        var lastPersonaLevel = persona?.Level ?? 0;
        await orchestrator.RunAsync(_runtime, curriculum, options, m =>
        {
            _gymCycle = m.Cycle; _lastDifficulty = m.Difficulty; _lastLoss = m.Loss; _lastAcc = m.Accuracy; _lastRoute = m.RoutePurity; _lastConf = m.Confidence;
            lock (_warmHistory) { _warmHistory.Add((m.Cycle, m.Accuracy, m.Loss)); if (_warmHistory.Count > 400) _warmHistory.RemoveAt(0); } // Inspect-tab warming curve
            // Each muscle advances + persists its OWN level independently (Name = "gym-<skill>").
            foreach (var g in gymChildren)
            {
                if (g.Level == lastLevels[g]) continue;
                lastLevels[g] = g.Level;
                try { File.WriteAllText(Path.Combine(_gymStateDir, g.Name + "-level.txt"), g.Level.ToString()); } catch { }
                AppendOutput($"[train] {g.Name} LEVEL → {g.Level}");
            }
            // The persona's level grows its pool — when it unlocks a new situation, SEED the now-larger repertoire so
            // the freshly-active intents are retrievable before they're probed next cycle.
            if (persona is not null && persona.Level != lastPersonaLevel)
            {
                lastPersonaLevel = persona.Level;
                try { _runtime.SeedConversationalChunks(persona.Repertoire); } catch { }
                try { File.WriteAllText(Path.Combine(_gymStateDir, "personality-level.txt"), persona.Level.ToString()); } catch { }
                AppendOutput($"[train] personality LEVEL → {persona.Level} (new situation unlocked + seeded)");
            }
            UpdateGymStats(m);
            // THE NAVIGATOR LEARNS THE LIVE SPACE — a LIGHT training step per gym cycle (serialized with model-train /
            // REPL / save via the runtime's model gate; bounded; CUDA-OOM-safe). Accumulates overnight so the platonic
            // walker keeps pace with the growing space. Logged alongside the gym warm-history.
            if (trainNavigator)
            {
                try
                {
                    var nav = _runtime.TrainNavigatorCycle();
                    _lastNavTrain = nav;
                    if (nav.Queries > 0)
                        AppendOutput($"[nav] cyc {m.Cycle} | loss {nav.Loss:F3} | queries {nav.Queries} | resolve {nav.ResolvePct:P0} abstain {nav.AbstainPct:P0} | running resolve {_runtime.NavResolveRunningPct:P0}");
                    // HELD-OUT generalization curve (M4) — the honest "is reasoning improving" signal. NavReasoningCurriculum
                    // registers the held-out set + already evaluates it in SelfAssess each cycle; surface the newest point so
                    // an overnight run shows it CLIMBING. Skipped (count 0) when no held-out set is registered.
                    if (_runtime.NavHeldOutCount > 0)
                    {
                        var ho = _runtime.NavHeldOutHistory;
                        if (ho.Count > 0)
                        {
                            var h = ho[^1];
                            AppendOutput($"[nav-heldout] eval {h.Cycle} | accuracy {h.AccuracyPct:P0} resolve {h.ResolvePct:P0} | over {h.Count} HELD-OUT queries (never trained) ← M4 curve");
                        }
                    }
                }
                catch (Exception ex) { AppendOutput($"[nav] skipped cyc {m.Cycle}: {ex.Message}"); }
            }
            // Op-head class-balance window [abstain,add,sub,mul,div] — one dominant entry = the head COLLAPSING (the
            // erosion failure mode, now visible live). Shown as the four operator shares.
            var opStr = m.OpClassBalance is { Count: 5 } op ? $" | op +{op[1]} -{op[2]} x{op[3]} /{op[4]}" : "";
            AppendOutput($"[train] cyc {m.Cycle} diff {m.Difficulty} | loss {m.Loss:F3} acc {m.Accuracy:P0} route {m.RoutePurity:P0} conf {m.Confidence:F2} | trained {m.TrainedCount}/{m.GeneratedCount}{opStr} | {m.CycleSeconds:F0}s");
            foreach (var s in m.Samples)   // ✓ correct+platonic / ~ right value but NEURAL (not platonic) / ✗ wrong
            {
                var mark = s.Correct ? "✓" : s.ValueCorrect ? "~" : "✗";
                var note = s.Correct ? ""
                         : s.ValueCorrect ? "  (right value, but NEURAL route — gym credits only the platonic path)"
                         : string.IsNullOrEmpty(s.Expected) ? "" : $"  (want \"{s.Expected}\")";
                AppendOutput($"      {mark} {(s.Platonic ? "P" : "n")}  \"{s.Query}\"  →  \"{s.Output}\"{note}");
            }
        }, ct);
        AppendOutput("[train] paused — model persists (AutoPersist).");
    }

    // UNIFIED progress view: the whole picture at once — an overall summary line plus EVERY lesson (each gym muscle +
    // personality) with its own level, this-cycle accuracy, and mastered state. Replaces the single conflated "Level"
    // that made the per-muscle sub-lessons feel disjoint.
    private void UpdateGymStats(CycleMetrics m)
    {
        if (InvokeRequired) { Invoke(() => UpdateGymStats(m)); return; }
        var t = GetControl<TextBox>("AutoStatsText");
        if (t == null) return;

        var units = m.Units ?? Array.Empty<UnitProgress>();
        var mastered = units.Count(u => u.Mastered);
        var lines = new List<string>
        {
            "GYM — unified progress (every lesson + personality)",
            $"Cycle {m.Cycle}   overall acc {m.Accuracy:P0}   route {m.RoutePurity:P0}   conf {m.Confidence:F2}",
            $"mastered {mastered}/{units.Count} lessons   |   loss {m.Loss:F3}   |   {m.CycleSeconds:F0}s/cycle",
            new string('─', 44),
        };
        // Sort so the eye reads progress: still-learning (lowest accuracy) first, mastered last.
        foreach (var u in units.OrderBy(u => u.Mastered).ThenBy(u => u.Accuracy).ThenBy(u => u.Name))
            lines.Add($" {(u.Mastered ? "✓" : "·")} {Shorten(u.Name),-24} L{u.Level,-3} {u.Accuracy,5:P0}");
        t.Text = string.Join(Environment.NewLine, lines);
    }

    // Trim the unit key to a readable lesson label ("gym-multiply" → "multiply", "focused(...)" stays, "personality").
    private static string Shorten(string name) =>
        name.StartsWith("gym-", StringComparison.Ordinal) ? name.Substring(4) : name;

    // ── INSPECT TAB ─────────────────────────────────────────────────────────────────────────────────────────────────
    // A live window into the platonic space — the foundation-warming debugger. Reads the live DialecticalSpace BEST-EFFORT
    // off the UI's timer (heavy reads on a background task, all wrapped in try/catch, tolerant of mid-mutation), so it
    // never blocks the UI or fights the throttle-0 /status starvation. Four panes: SPACE vitals + train split, the
    // FUNCTION-WORD SEPARATION table (the thing we keep chasing — see which words bridge vs cluster, live), the WARMING
    // CURVE (acc over cycles, with the 0.85 mastery line — watch it hold vs collapse), and a CONCEPT PROBE (type a word →
    // fn-like? / coherence / degree / nearest neighbours). Tip: it's cleanest to read with /gym/pause, but live works too.
    private static readonly Font InspMono = new("Consolas", 9f);
    private TabPage CreateInspectTab()
    {
        var tab = new TabPage("Inspect");
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2, Padding = new Padding(6), BackColor = Color.FromArgb(18, 18, 22) };
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 48));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 52));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 56));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var vitals = new Label { Name = "InspVitals", Dock = DockStyle.Fill, Font = InspMono, ForeColor = Color.Gainsboro,
            Text = "(start the gym to populate the space)" };
        root.Controls.Add(vitals, 0, 0); root.SetColumnSpan(vitals, 2);

        var funcBox = new TextBox { Name = "InspFunc", Dock = DockStyle.Fill, Multiline = true, ReadOnly = true,
            ScrollBars = ScrollBars.Vertical, Font = InspMono, BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro };
        root.Controls.Add(funcBox, 0, 1);

        var right = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3 };
        right.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        right.RowStyles.Add(new RowStyle(SizeType.Absolute, 26));
        right.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        var curve = new Panel { Name = "InspCurve", Dock = DockStyle.Fill, BackColor = Color.FromArgb(14, 14, 18) };
        curve.Paint += DrawWarmingCurve;
        right.Controls.Add(curve, 0, 0);
        var probeIn = new TextBox { Name = "InspProbeIn", Dock = DockStyle.Fill, Font = InspMono,
            BackColor = Color.FromArgb(30, 30, 36), ForeColor = Color.White, Text = "type a word + Enter to probe…" };
        probeIn.GotFocus += (s, e) => { if (probeIn.Text.StartsWith("type a word")) probeIn.Text = ""; };
        probeIn.KeyDown += (s, e) => { if (e.KeyCode == Keys.Enter) { e.SuppressKeyPress = true; ProbeConcept(probeIn.Text); } };
        right.Controls.Add(probeIn, 0, 1);
        var probeOut = new TextBox { Name = "InspProbe", Dock = DockStyle.Fill, Multiline = true, ReadOnly = true,
            ScrollBars = ScrollBars.Vertical, Font = InspMono, BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro,
            Text = "Probe a concept: is it function-like? its coherence, degree, and nearest neighbours in meaning space." };
        right.Controls.Add(probeOut, 0, 2);
        root.Controls.Add(right, 1, 1);

        // Host the existing SPACE view + a NAVIGATOR view as two sub-tabs inside Inspect (read-only observation surfaces).
        var inner = new TabControl { Dock = DockStyle.Fill };
        var spaceTab = new TabPage("Space") { BackColor = Color.FromArgb(18, 18, 22) };
        spaceTab.Controls.Add(root);
        inner.TabPages.Add(spaceTab);

        var navTab = new TabPage("Navigator") { BackColor = Color.FromArgb(18, 18, 22) };
        var navBox = new TextBox { Name = "InspNav", Dock = DockStyle.Fill, Multiline = true, ReadOnly = true,
            ScrollBars = ScrollBars.Vertical, Font = InspMono, BackColor = Color.FromArgb(24, 24, 28), ForeColor = Color.Gainsboro,
            Text = "The platonic navigator (gym-trained, read live). Run /nav <concept> [genus|domain|root] in the REPL to walk a query." };
        navTab.Controls.Add(navBox);
        inner.TabPages.Add(navTab);

        tab.Controls.Add(inner);
        _inspectTimer = new System.Windows.Forms.Timer { Interval = 1500 };
        _inspectTimer.Tick += (s, e) => RefreshInspect();
        _inspectTimer.Start();
        return tab;
    }

    private void RefreshInspect()
    {
        GetControl<Panel>("InspCurve")?.Invalidate();                 // redraw the curve from captured history (cheap, UI thread)
        if (_tabControl.SelectedTab?.Text != "Inspect" || _inspectReading) return;
        if (_runtime?.State?.Memory is not DialecticalSpace space) return;
        _inspectReading = true;
        Task.Run(() =>
        {
            try
            {
                var (vitals, func) = BuildInspectStrings(space);
                string nav;
                try { nav = BuildNavString(_runtime.GetNavigatorDiagnostics()); } catch { nav = string.Empty; }
                BeginInvoke(new Action(() =>
                {
                    var v = GetControl<Label>("InspVitals"); if (v != null && vitals.Length > 0) v.Text = vitals;
                    var f = GetControl<TextBox>("InspFunc"); if (f != null && func.Length > 0) f.Text = func;
                    var n = GetControl<TextBox>("InspNav"); if (n != null && nav.Length > 0) n.Text = nav;
                }));
            }
            catch { /* best-effort live read — skip this tick */ }
            finally { _inspectReading = false; }
        });
    }

    private (string vitals, string func) BuildInspectStrings(DialecticalSpace ds)
    {
        var ex = GenesisNova.Train.GenesisTrainer.ProfExamples;
        var split = ex > 0
            ? $"   |   per-ex: NN {GenesisNova.Train.GenesisTrainer.ProfNnMs / ex:F0}ms  Substrate {GenesisNova.Train.GenesisTrainer.ProfSubstrateMs / ex:F0}ms  Other {GenesisNova.Train.GenesisTrainer.ProfOtherMs / ex:F0}ms"
            : "";
        var vitals =
            $"SPACE   nodes {ds.NodeCount:N0}    relations {ds.RelationCount:N0}    archived {ds.ArchivedNodeCount:N0}    atoms {ds.AtomCount}{split}\n" +
            $"GYM     cycle {_gymCycle}   acc {_lastAcc:P0}   loss {_lastLoss:F2}   level {_lastDifficulty}   throttle {_gymThrottlePct}%";

        // Probe the ACTUAL trained vocabulary (pulled from the live foundation curriculum) so content words are really in
        // the space — a fixed guessed list reads all-zero because those words were never trained / got discharged as noise.
        var glue = _inspectGlue is { Length: > 0 } ? _inspectGlue : new[] { "the", "of", "is", "a", "to", "in", "and", "with", "my", "your", "for", "on" };
        var content = _inspectContent is { Length: > 0 } ? _inspectContent : new[] { "cat", "dog", "red", "blue", "green", "pink", "cow", "pig", "hen", "fox" };
        double mean = 0, thr = 0; var active = 0; var gotStats = false;
        try { var s = ds.FunctionStats("the"); mean = s.Mean; thr = s.Threshold; active = s.Active; gotStats = true; } catch { }

        var sb = new System.Text.StringBuilder();
        int gCoh = 0, gPmi = 0, cCoh = 0, cPmi = 0;
        var rows = new System.Text.StringBuilder();
        rows.AppendLine("  GLUE (want ✓)       coh graph   assoc PMI");
        foreach (var w in glue) { var (cf, coh, deg, pf, asc) = ProbeWord(ds, w); if (cf) gCoh++; if (pf) gPmi++;
            rows.AppendLine($"    {w,-13}{coh,5:F2} {(cf ? "✓" : "·"),-5}  {asc,6:F2} {(pf ? "✓" : "·")}"); }
        rows.AppendLine();
        rows.AppendLine("  CONTENT (want ·)    coh graph   assoc PMI");
        foreach (var w in content) { var (cf, coh, deg, pf, asc) = ProbeWord(ds, w); if (cf) cCoh++; if (pf) cPmi++;
            rows.AppendLine($"    {w,-13}{coh,5:F2} {(cf ? "✓" : "·"),-5}  {asc,6:F2} {(pf ? "✓" : "·")}"); }

        var graphSep = (double)gCoh / glue.Length - (double)cCoh / content.Length;
        var pmiSep = (double)gPmi / glue.Length - (double)cPmi / content.Length;
        sb.AppendLine($"FUNCTION-WORD SEPARATION   graph {graphSep:P0}   |   PMI {pmiSep:P0}");
        sb.AppendLine($"  graph: glue {gCoh}/{glue.Length} content {cCoh}/{content.Length}   PMI: glue {gPmi}/{glue.Length} content {cPmi}/{content.Length}");
        if (gotStats) sb.AppendLine($"  coherence mean {mean:F2} ≤{thr:F2}   active {active:N0}");
        sb.AppendLine();
        sb.Append(rows);
        return (vitals, sb.ToString());
    }

    // The NAVIGATOR sub-panel: vitals (last nav-train loss / running resolve% / last-cycle queries+abstain% / ‖SelfField‖),
    // the self FOCUS (live concepts nearest the self — what the mind is dwelling on), and the LAST /nav walk. All from the
    // gated GetNavigatorDiagnostics read, so this never touches torch on the UI/timer thread.
    private static string BuildNavString(GenesisNova.Runtime.NavigatorDiagnostics d)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("NAVIGATOR   the platonic query-walker (gym-trained, read live)");
        sb.AppendLine(new string('─', 60));
        sb.AppendLine("VITALS");
        sb.AppendLine($"  last cycle   loss {d.LastLoss:F3}   queries {d.LastQueries}   resolve {d.LastResolvePct:P0}   abstain {d.LastAbstainPct:P0}");
        sb.AppendLine($"  running      resolve {d.RunningResolvePct:P0}");
        if (d.LastHeldOut.Count > 0)
            sb.AppendLine($"  HELD-OUT     accuracy {d.LastHeldOut.AccuracyPct:P0}   resolve {d.LastHeldOut.ResolvePct:P0}   over {d.LastHeldOut.Count} never-trained queries   (M4 curve)");
        sb.AppendLine($"  self         ‖SelfField‖ {d.SelfMagnitude:F3}   conditioning {(d.SelfConditions ? "ON" : "OFF")}");
        sb.AppendLine();
        sb.AppendLine("SELF FOCUS   live concepts nearest the self (what the mind is dwelling on)");
        if (d.SelfFocus.Count == 0)
            sb.AppendLine("  (the self is empty — the mind has not dwelt on anything yet)");
        else
            foreach (var (c, sim) in d.SelfFocus)
                sb.AppendLine($"    {c,-22}{sim,7:F3}");
        sb.AppendLine();
        sb.AppendLine("LAST WALK   most recent /nav");
        if (d.LastWalk is { } w)
        {
            var traj = w.Trajectory is { Count: > 0 } ? string.Join(" → ", w.Trajectory) : "(none)";
            sb.AppendLine($"  '{w.Anchor}'  cue={w.Cue}  →  {w.Answer}  ({(w.Reached ? "reached" : "abstained")})");
            sb.AppendLine($"  trajectory: {traj}");
            sb.AppendLine($"  self {(w.SelfConditioned ? "conditioned the walk" : "was empty")}   ·   {w.Message}");
        }
        else
            sb.AppendLine("  (run /nav <concept> [genus|domain|root] in the REPL)");
        return sb.ToString();
    }

    // Both verdicts computed DIRECTLY from the stats (not IsFunctionLike) so the panel shows graph-clustering vs the PMI
    // distributional metric SIDE-BY-SIDE on the same warmed space — the live A/B for "does PMI separate what the graph misses".
    private static (bool cohFn, double coh, int deg, bool pmiFn, double assoc) ProbeWord(DialecticalSpace ds, string w)
    {
        try
        {
            var s = ds.FunctionStats(w);
            var a = ds.AssociationStats(w);
            var cohFn = s.MinWarm >= 6 && s.Std > 1e-6 && s.Centrality <= s.Threshold && s.Centrality <= s.Floor; // graph verdict
            var pmiFn = s.MinWarm >= 6 && a.Std > 1e-6 && a.Assoc <= a.Threshold;                                 // PMI verdict
            return (cohFn, s.Centrality, s.MinWarm, pmiFn, a.Assoc);
        }
        catch { return (false, 0, 0, false, 0); }
    }

    private void ProbeConcept(string word)
    {
        word = (word ?? "").Trim().ToLowerInvariant();
        if (word.Length == 0) return;
        if (_runtime?.State?.Memory is not DialecticalSpace space) { var o0 = GetControl<TextBox>("InspProbe"); if (o0 != null) o0.Text = "(no space — start the gym)"; return; }
        Task.Run(() =>
        {
            string text;
            try
            {
                var sb = new System.Text.StringBuilder();
                var exists = space.ContainsConcept(word);
                sb.AppendLine($"'{word}'  {(exists ? "" : "— not in the active space (discharged as noise, or never seen)")}");
                if (exists)
                {
                    var s = space.FunctionStats(word);
                    sb.AppendLine($"  function-like : {(space.IsFunctionLike(word) ? "YES — bridges unrelated words (glue)" : "no — clusters with kin (content)")}");
                    sb.AppendLine($"  coherence     : {s.Centrality:F3}     (function if ≤ {s.Threshold:F2})");
                    sb.AppendLine($"  degree        : {s.MinWarm}     (relation neighbours)");
                    sb.AppendLine();
                    sb.AppendLine("  nearest in meaning space:");
                    foreach (var (sym, dist) in space.GetNearestConcepts(word, maxNeighbors: 12))
                        sb.AppendLine($"    {dist,5:F2}   {sym}");
                }
                text = sb.ToString();
            }
            catch (Exception ex) { text = $"(read failed — space busy: {ex.GetType().Name})"; }
            try { BeginInvoke(new Action(() => { var o = GetControl<TextBox>("InspProbe"); if (o != null) o.Text = text; })); } catch { }
        });
    }

    private void DrawWarmingCurve(object? sender, PaintEventArgs e)
    {
        var p = (Panel)sender!; var g = e.Graphics; g.Clear(p.BackColor);
        (int Cycle, double Acc, double Loss)[] hist; lock (_warmHistory) hist = _warmHistory.ToArray();
        int w = p.Width, h = p.Height, pad = 6, top = 18, bot = h - 6;
        float Y(double a) => (float)(bot - Math.Clamp(a, 0, 1) * (bot - top));
        g.DrawString("acc warming curve (0–1)", InspMono, Brushes.Gray, pad, 2);
        using (var grid = new Pen(Color.FromArgb(48, 48, 56)))
            foreach (var lvl in new[] { 0.0, 0.5, 1.0 }) g.DrawLine(grid, pad, Y(lvl), w - pad, Y(lvl));
        using (var master = new Pen(Color.FromArgb(120, 90, 200, 90))) { g.DrawLine(master, pad, Y(0.85), w - pad, Y(0.85)); g.DrawString("0.85 mastery", InspMono, Brushes.DarkSeaGreen, w - 96, Y(0.85) - 14); }
        if (hist.Length >= 2)
        {
            float X(int i) => pad + (w - 2 * pad) * (i / (float)(hist.Length - 1));
            using var accPen = new Pen(Color.FromArgb(80, 200, 255), 2f);
            for (var i = 1; i < hist.Length; i++) g.DrawLine(accPen, X(i - 1), Y(hist[i - 1].Acc), X(i), Y(hist[i].Acc));
            g.DrawString($"{hist[^1].Acc:P0} @cyc {hist[^1].Cycle}", new Font("Consolas", 9f, FontStyle.Bold), Brushes.Aqua, pad + 2, top);
        }
        else g.DrawString("(no cycles yet — start the gym)", InspMono, Brushes.Gray, pad + 2, h / 2);
    }

    // ── Control endpoint: a tiny localhost HTTP server so headless tools (and Claude) can drive the LIVE in-app
    // model — recall against it, read gym status, pause/resume — without a second process fighting for the GPU.
    private void StartControlServer(int port = 8787)
    {
        try
        {
            _control = new HttpListener();
            _control.Prefixes.Add($"http://127.0.0.1:{port}/");
            _control.Start();
            AppendOutput($"[ctrl] http://127.0.0.1:{port}/   endpoints: /status  /recall?q=  /probe?q=  /log?tail=N  /nav  /throttle?pct=  /gym/start  /gym/stop  /gym/restart  /gym/trainers[?set=|enable=|disable=]");
            _ = Task.Run(async () =>
            {
                while (_control.IsListening)
                {
                    HttpListenerContext ctx;
                    try { ctx = await _control.GetContextAsync(); }
                    catch { break; }
                    _ = Task.Run(() => HandleControlRequest(ctx));
                }
            });
        }
        catch (Exception ex) { AppendOutput($"[ctrl] control server failed: {ex.Message}"); }
    }

    private async Task HandleControlRequest(HttpListenerContext ctx)
    {
        string body; var status = 200;
        try
        {
            var path = (ctx.Request.Url?.AbsolutePath ?? "").TrimEnd('/').ToLowerInvariant();
            var q = ctx.Request.QueryString["q"];
            switch (path)
            {
                case "":
                case "/status":
                    var mem = _runtime.State?.Memory;
                    body = JsonSerializer.Serialize(new
                    {
                        running = GymRunning,
                        cycle = _gymCycle,
                        level = _lastDifficulty,
                        loss = _lastLoss,
                        acc = _lastAcc,
                        route = _lastRoute,
                        conf = _lastConf,
                        throttle = _gymThrottlePct,
                        hidden = _runtime.HiddenSize,
                        nodes = mem?.NodeCount ?? 0,
                        relations = mem?.RelationCount ?? 0,
                        archived = mem?.ArchivedNodeCount ?? 0,
                        profEx = GenesisNova.Train.GenesisTrainer.ProfExamples,
                        profNnMs = GenesisNova.Train.GenesisTrainer.ProfNnMs,
                        profSubMs = GenesisNova.Train.GenesisTrainer.ProfSubstrateMs,
                        profCloneMs = GenesisNova.Train.GenesisTrainer.ProfCloneMs,
                        profOtherMs = GenesisNova.Train.GenesisTrainer.ProfOtherMs,
                        model = _gymStateDir
                    });
                    break;
                case "/recall":
                case "/query":
                    if (string.IsNullOrWhiteSpace(q)) { status = 400; body = "{\"error\":\"missing ?q=\"}"; break; }
                    // The gym holds the model gate near-continuously (train batch + its own probes), so be
                    // patient: a long-ish gate wait lets the recall acquire a slot at an inter-cycle release.
                    GenesisPredictTaskData? pr = null;
                    for (var i = 0; i < 24 && pr is null; i++) pr = await _runtime.TryPredictAsync(q, gateWaitMilliseconds: 1500);
                    if (pr?.Result is null) { status = 503; body = "{\"error\":\"no model slot (busy training) — retry\"}"; break; }
                    body = JsonSerializer.Serialize(new
                    {
                        query = q,
                        output = pr.Result.Output,
                        platonic = !pr.Result.UsedNeuralFallback,
                        neuralFallback = pr.Result.UsedNeuralFallback,
                        confidence = pr.Result.PlatonicConfidence,
                        decisionPath = pr.Result.DecisionPath  // navigator-walk / field-relax / platonic-abstain / neural-token ...
                    });
                    break;
                case "/log":
                    {
                        var tail = int.TryParse(ctx.Request.QueryString["tail"], out var t) ? Math.Clamp(t, 1, 400) : 40;
                        var lines = LogTail(tail);
                        body = JsonSerializer.Serialize(new { count = lines.Length, lines });
                    }
                    break;
                case "/nav":
                    {
                        var nav = _runtime.LastNavTrain;
                        var curve = _runtime.NavHeldOutHistory
                            .Select(h => new { cycle = h.Cycle, accuracyPct = h.AccuracyPct, resolvePct = h.ResolvePct, count = h.Count })
                            .ToArray();
                        body = JsonSerializer.Serialize(new
                        {
                            lastTrain = new { loss = nav.Loss, queries = nav.Queries, resolvePct = nav.ResolvePct, abstainPct = nav.AbstainPct },
                            resolveRunningPct = _runtime.NavResolveRunningPct,
                            heldOutCount = _runtime.NavHeldOutCount,
                            heldOut = curve
                        });
                    }
                    break;
                case "/probe":
                    if (string.IsNullOrWhiteSpace(q)) { status = 400; body = "{\"error\":\"missing ?q=\"}"; break; }
                    ConceptProbe? probe = null;
                    try { probe = _runtime.ProbeConcept(q); } catch (Exception ex) { status = 500; body = JsonSerializer.Serialize(new { error = ex.Message }); break; }
                    if (probe is null) { status = 503; body = "{\"error\":\"no diagnostics (non-dialectical space or no model)\"}"; break; }
                    body = JsonSerializer.Serialize(new
                    {
                        concept = probe.Concept,
                        exists = probe.Exists,
                        isFunctionLike = probe.IsFunctionLike,
                        neighbourCoherence = probe.NeighbourCoherence,
                        coherenceThreshold = probe.CoherenceThreshold,
                        coherenceFloor = probe.CoherenceFloor,
                        functionEvidence = probe.FunctionEvidence,
                        strongRelationDegree = probe.StrongRelationDegree,
                        relationDegree = probe.RelationDegree,
                        activeConcepts = probe.ActiveConcepts,
                        nearest = probe.Nearest.Select(n => new { symbol = n.Symbol, distance = n.Distance }).ToArray()
                    });
                    break;
                case "/gym/pause":
                case "/gym/stop":
                    // CLEAN async shutdown: cancel + AWAIT the loop drain + AWAIT one final save. The response below is
                    // only written AFTER this returns, so when the client sees 200 the model IS on disk.
                    await StopGymAsync();
                    body = "{\"ok\":\"stopped\",\"saved\":true}";
                    break;
                case "/gym/resume":
                case "/gym/start":
                    // Waits for any in-flight stop+save to fully complete (gate) before launching — no start-before-save,
                    // no overlapping loops. No-op if the gym is already running.
                    await StartGymAsync();
                    body = $"{{\"ok\":\"started\",\"running\":{GymRunning.ToString().ToLowerInvariant()}}}";
                    break;
                case "/gym/restart":
                    // Clean stop (awaits the final save) → start fresh with the current trainer config.
                    await RestartGymAsync();
                    body = $"{{\"ok\":\"restarted\",\"running\":{GymRunning.ToString().ToLowerInvariant()}}}";
                    break;
                case "/gym/trainers":
                    body = await HandleGymTrainersAsync(ctx);
                    break;
                case "/throttle":
                    if (!int.TryParse(ctx.Request.QueryString["pct"], out var pct) || pct < 0 || pct > 500) { status = 400; body = "{\"error\":\"?pct=0..500\"}"; break; }
                    _gymThrottlePct = pct;
                    BeginInvoke(new Action(() => { var n = GetControl<NumericUpDown>("GymThrottle"); if (n != null) n.Value = pct; }));
                    body = $"{{\"ok\":\"throttle {pct}%\"}}";
                    break;
                default:
                    status = 404;
                    body = "{\"error\":\"unknown path\",\"endpoints\":[\"/status\",\"/recall?q=\",\"/query?q=\",\"/probe?q=\",\"/log?tail=N\",\"/nav\",\"/throttle?pct=\",\"/gym/pause\",\"/gym/resume\",\"/gym/start\",\"/gym/stop\",\"/gym/restart\",\"/gym/trainers\",\"/gym/trainers?set=A,B\",\"/gym/trainers?enable=A\",\"/gym/trainers?disable=A\"]}";
                    break;
            }
        }
        catch (Exception ex) { status = 500; body = JsonSerializer.Serialize(new { error = ex.Message }); }
        try
        {
            var buf = System.Text.Encoding.UTF8.GetBytes(body);
            ctx.Response.StatusCode = status;
            ctx.Response.ContentType = "application/json";
            ctx.Response.OutputStream.Write(buf, 0, buf.Length);
            ctx.Response.OutputStream.Close();
        }
        catch { /* client gone */ }
    }

    // The progressive-trainer registry — every curriculum checkbox an external orchestrator can stage, mapped to its
    // friendly Name (used in ?set=/?enable=/?disable=), its UI control, and a human label. Order = staging order
    // (prebake → op-cues → number-words → nav-reasoning → gym skills → the rest). Generated each call so the GymSkill
    // enum is the single source of truth for the gym muscles.
    private static IReadOnlyList<(string Name, string Control, string Label)> GymTrainerRegistry()
    {
        var list = new List<(string, string, string)>
        {
            ("PrebakeLanguage", "CurPrebakeLanguage", "Prebake — warm function-word recognition (real corpus + synthetic L1)"),
            ("CorpusPrediction","CurCorpusPrediction","Corpus prediction (masked cloze) — self-supervised base-model objective (default on)"),
            ("OpCues",          "CurOpCues",          "Op-cue words — sum / difference / product / quotient → operator"),
            ("NumberWords",     "CurNumberWords",     "Number words — digit ↔ word lexicon"),
            ("NavReasoning",    "CurNavReasoning",    "Nav-reasoning — is-a taxonomy + level cues (HELD-OUT curve)"),
            ("TrainNavigator",  "CurTrainNavigator",  "Train Navigator — the platonic walker learns the live space each cycle"),
            ("Creators",        "CurCreators",        "Creators — skill ladder"),
            ("MemCode",         "CurMemCode",         "Memory + Code index"),
            ("Personality",     "CurPersonality",     "Personality — rude chatbot (seeded reply chunks)"),
        };
        foreach (var skill in Enum.GetValues<GenesisNova.Train.GymSkill>())
            list.Add(("Gym" + skill, "GymSkill_" + skill, "Gym — " + GymSkillLabel(skill)));
        return list;
    }

    // GET /gym/trainers                      → JSON list of every trainer + enabled(checked) state + running + a note.
    // /gym/trainers?set=Name1,Name2,...      → enable EXACTLY those, disable the rest (thread-safe via UI Invoke).
    // /gym/trainers?enable=Name[,Name] / ?disable=Name[,Name] → incremental toggles.
    // A token matches a trainer by its friendly Name OR its control name, case-insensitively. Changes take effect on
    // the NEXT /gym/start or /gym/restart (StartGym reads the checkboxes); we say so in the response.
    private async Task<string> HandleGymTrainersAsync(HttpListenerContext ctx)
    {
        var reg = GymTrainerRegistry();
        var setParam = ctx.Request.QueryString["set"];
        var enableParam = ctx.Request.QueryString["enable"];
        var disableParam = ctx.Request.QueryString["disable"];

        string? Resolve(string tok)
        {
            tok = tok.Trim();
            foreach (var r in reg)
                if (string.Equals(r.Name, tok, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(r.Control, tok, StringComparison.OrdinalIgnoreCase))
                    return r.Control;
            return null;
        }
        static IEnumerable<string> Split(string s) =>
            s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var unknown = new List<string>();
        var changed = new List<string>();
        if (setParam != null || enableParam != null || disableParam != null)
        {
            // Apply ALL checkbox writes on the UI thread in one marshalled+awaited unit (checkboxes are UI-thread state).
            await InvokeOnUiAsync(() =>
            {
                if (setParam != null)
                {
                    var wanted = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    foreach (var tok in Split(setParam)) { var c = Resolve(tok); if (c != null) wanted.Add(c); else unknown.Add(tok); }
                    foreach (var r in reg)
                    {
                        var cb = GetControl<CheckBox>(r.Control); if (cb == null) continue;
                        var on = wanted.Contains(r.Control);
                        if (cb.Checked != on) { cb.Checked = on; changed.Add((on ? "+" : "-") + r.Name); }
                    }
                }
                foreach (var (param, on) in new[] { (enableParam, true), (disableParam, false) })
                {
                    if (param == null) continue;
                    foreach (var tok in Split(param))
                    {
                        var c = Resolve(tok);
                        if (c == null) { unknown.Add(tok); continue; }
                        var cb = GetControl<CheckBox>(c); if (cb == null) continue;
                        if (cb.Checked != on) { cb.Checked = on; changed.Add((on ? "+" : "-") + reg.First(r => r.Control == c).Name); }
                    }
                }
                return true;
            });
            if (changed.Count > 0) AppendOutput($"[ctrl] trainers reconfigured: {string.Join(" ", changed)} (applies on next /gym/start or /gym/restart)");
        }

        // Read the (possibly just-updated) state on the UI thread.
        var trainers = await InvokeOnUiAsync(() => reg.Select(r => new
        {
            name = r.Name,
            control = r.Control,
            label = r.Label,
            enabled = GetControl<CheckBox>(r.Control)?.Checked ?? false
        }).ToArray());

        return JsonSerializer.Serialize(new
        {
            running = GymRunning,
            note = "enabled = checked. Changes take effect on the next /gym/start or /gym/restart.",
            changed = changed.ToArray(),
            unknown = unknown.ToArray(),
            trainers
        });
    }

    private void HandleAutonomousTrainingProgress(GenesisAutonomousTrainingEventPayload payload)
    {
        if (InvokeRequired)
        {
            Invoke(() => HandleAutonomousTrainingProgress(payload));
            return;
        }

        try
        {
            // Update stats label with round progress
            var statsLabel = GetControl<TextBox>("AutoStatsText");
            if (statsLabel != null)
            {
                var summary = string.IsNullOrWhiteSpace(payload.CreatorSummary)
                    ? "Creator stats: n/a"
                    : $"Creator stats:{Environment.NewLine}{payload.CreatorSummary}";
                var success = double.IsNaN(payload.ExampleSuccessRate) ? "n/a" : payload.ExampleSuccessRate.ToString("P1");
                statsLabel.Text =
                    $"Phase: {payload.StepName} | Round: {payload.Round} | Loss: {payload.Loss:F4} | Success: {success} | Samples: {payload.SamplesTrained} | Elapsed: {payload.ElapsedMs}ms | Dataset: {payload.Dataset}" +
                    Environment.NewLine +
                    $"Mix: prompt-answer {payload.PromptAnswerExampleCount} | windowed text {payload.WindowedTextExampleCount} | skipped correct {payload.SkippedCorrectExampleCount}" +
                    Environment.NewLine +
                    summary;
            }
        }
        catch (Exception ex)
        {
            AppendOutput($"[ui] Error updating progress: {ex.Message}");
        }
    }

    private GenesisAutonomousTrainingRequest CaptureLiveAutonomousRequest(GenesisAutonomousTrainingRequest baseline)
    {
        if (InvokeRequired)
        {
            var captured = Invoke(new Func<GenesisAutonomousTrainingRequest>(() => CaptureLiveAutonomousRequest(baseline)));
            return captured is GenesisAutonomousTrainingRequest request ? request : baseline;
        }

        var live = TryBuildAutonomousTrainingRequest();
        if (live is null)
            return baseline;

        return baseline with
        {
            MaxRounds = live.MaxRounds,
            LossThreshold = live.LossThreshold,
            MinSampleCount = live.MinSampleCount,
            MaxSampleCount = live.MaxSampleCount,
            MinTrainCount = live.MinTrainCount,
            MaxTrainCount = live.MaxTrainCount,
            MaxDifficulty = live.MaxDifficulty,
            RoundTrainBudget = live.RoundTrainBudget,
            MaxGenerationConcurrency = live.MaxGenerationConcurrency,
            EnabledCreators = live.EnabledCreators
        };
    }

    private async Task<GenesisTrainingReport> ExecuteTraining(string filePath, int epochs)
    {
        var sw = Stopwatch.StartNew();
        Interlocked.Increment(ref _activeTrainingOperations);
        AppendOutput($"Loading training data from: {Path.GetFileName(filePath)}");
        
        var examples = await GenesisTrainingDataLoader.LoadFromFileAsync(filePath);
        AppendOutput($"Loaded {examples.Count} examples");
        
        // Calculate optimal hidden size based on GPU VRAM for inference
        var targetHidden = EstimateHiddenSizeFromDataset(examples);
        if (targetHidden > _runtime.HiddenSize)
        {
            AppendOutput($"Expanding model: hidden {_runtime.HiddenSize} → {targetHidden} for {examples.Count} examples on GPU");
            _runtime.EnsureHiddenSize(targetHidden);
        }
        
        AppendOutput($"Training: {examples.Count} examples, {epochs} epochs, policy=CUDA-train/GPU-infer with CPU fallback");
        AppendOutput($"GPU inference device: 0 (A3000 preferred), CUDA training when available");
        AggressiveMemoryCleanup("pre-training");

        try
        {
            var logPath = Path.Combine(AppContext.BaseDirectory, "train.log");
            GenesisTrainingReport? report = null;
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                report = await _runtime.TrainAsync(
                    filePath,
                    1,
                    logPath: logPath);
            }
            if (report is null)
                throw new InvalidOperationException("Training report missing.");
            
            sw.Stop();
            AppendOutput($"Training complete in {sw.Elapsed.TotalSeconds:F1}s");
            AppendOutput($"Final loss: {report.AverageLoss.TokenLoss:F4}");
            AppendOutput(
                $"Platonic space: cycles={report.SpaceManagementCycles} nodes={report.FinalNodeCount}n/{report.FinalRelationCount}r " +
                $"noise={report.SpaceNoiseRatio:F3}");
            AppendOutput("(Model trained, ready for prediction)");
            return report;

        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            if (IsCudaOutOfMemory(ex))
            {
                AppendOutput("CUDA OOM detected - running aggressive cleanup so you can continue without restart.");
                AggressiveMemoryCleanup("post-oom");
            }
            throw;
        }
        finally
        {
            AggressiveMemoryCleanup("post-training");
            Interlocked.Decrement(ref _activeTrainingOperations);
        }
    }

    private int EstimateHiddenSizeFromDataset(IReadOnlyList<GenesisExample> examples)
    {
        // Query GPU VRAM and size model to fit exactly in dedicated VRAM (fast memory)
        // Overflow will use shared memory as needed
        if (!GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalMb, out var freeMb))
        {
            AppendOutput($"[GPU] Could not query NVIDIA GPU; using current hidden size {_runtime.HiddenSize}");
            return _runtime.HiddenSize;
        }

        // Target exactly 6GB dedicated VRAM (fast), allow shared memory overspill for edge cases
        const int targetFastVramMb = 6144;  // 6GB
        var estimatedHidden = GpuCapacityPlanner.EstimateHiddenSizeForInferenceOnly(
            examples,
            routeCount: 2,  // NumRoutes in GenesisNeuralModel
            vramMb: targetFastVramMb,
            targetUtilization: 0.98,  // Pack tight into dedicated VRAM
            reserveVramMb: 50,        // Minimal reserve for kernel overhead
            debugOutput: msg => AppendOutput(msg));

        AppendOutput($"[GPU] Target: 6GB fast VRAM (dedicated) | Estimated hidden: {estimatedHidden} | Overspill to shared: allowed");
        return estimatedHidden;
    }

    private string ResolveTrainingPath(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
            throw new FormatException("trainfile requires a path.");

        if (Path.IsPathRooted(rawPath))
            return rawPath;

        var combined = Path.Combine(_exampleFolder, rawPath);
        return File.Exists(combined) ? combined : rawPath;
    }

    private static (string Path, int Epochs) ParsePathAndEpochs(string payload)
    {
        var parts = payload.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            throw new FormatException("trainfile requires a path.");

        var epochs = 1;
        if (parts.Length > 1 && int.TryParse(parts[1], out var parsed))
            epochs = Math.Max(1, parsed);
        return (parts[0], epochs);
    }

    private static string ResolveDefaultExampleFolder()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 10 && current != null; i++)
        {
            var dataPath = Path.Combine(current.FullName, "data");
            if (Directory.Exists(dataPath))
                return dataPath;
            current = current.Parent;
        }
        return AppContext.BaseDirectory;
    }

    private static string BuildAutonomousCreatorSummary(GenesisTrainingReport report)
    {
        if (report.CreatorProgress is not { Count: > 0 })
            return string.Empty;

        var header = $"mix: prompt-answer {report.PromptAnswerExampleCount}, windowed text {report.WindowedTextExampleCount}, skipped correct {report.SkippedCorrectExampleCount}";
        var body = string.Join(
            Environment.NewLine,
            report.CreatorProgress
                .OrderBy(x => x.CreatorName, StringComparer.OrdinalIgnoreCase)
                .Select(x => $"{x.CreatorName} [{x.TrainingKind}]: loss {x.AverageTokenLoss:F3}, succ {x.SuccessRate:P0}, seen {x.SeenCount}"));
        return string.Join(Environment.NewLine, header, body);
    }

    private void AppendOutput(string message)
    {
        // Mirror into the bounded ring buffer FIRST (thread-safe, no UI marshalling) so the headless /log endpoint
        // sees every line even before/independent of the UI marshalling in AppendToLogBox.
        lock (_logRing)
        {
            _logRing.Enqueue($"[{DateTime.Now:HH:mm:ss}] {message}");
            while (_logRing.Count > LogRingCapacity) _logRing.Dequeue();
        }
        AppendToLogBox("AutoOutputBox", message);
    }

    /// <summary>The last <paramref name="tail"/> lines of the app output log from the in-memory ring buffer (newest
    /// last), for the headless /log endpoint. Capped at the ring capacity.</summary>
    private string[] LogTail(int tail)
    {
        lock (_logRing)
        {
            var all = _logRing.ToArray();
            var n = Math.Clamp(tail, 1, all.Length);
            return all.Length <= n ? all : all[^n..];
        }
    }

     private void AppendToLogBox(string boxName, string message)
     {
         if (InvokeRequired)
         {
             BeginInvoke(new Action(() => AppendToLogBox(boxName, message)));
             return;
         }

         var outputBox = GetControl<ListBox>(boxName);
         if (outputBox == null || outputBox.IsDisposed)
         {
             System.Diagnostics.Debug.WriteLine($"[log] ListBox '{boxName}' not found or disposed");
             return;
         }

         try
         {
             // ListBox: just add item (much safer than RichTextBox)
             var logEntry = $"[{DateTime.Now:HH:mm:ss}] {message}";
             outputBox.Items.Add(logEntry);
             System.Diagnostics.Debug.WriteLine($"[log] Added to {boxName}: {message}");

             // Trim if needed - ListBox trimming is safer
             if (outputBox.Items.Count > MaxTrainingLogChars / 50)  // Rough estimate: ~50 chars per line
             {
                 var now = DateTime.UtcNow;
                 if ((now - _lastTrimTime).TotalMilliseconds >= TrimDebounceMs)
                 {
                     _lastTrimTime = now;
                     // Remove oldest items
                     var itemsToRemove = Math.Max(1, outputBox.Items.Count - (MaxTrainingLogChars / 60));
                     for (var i = 0; i < itemsToRemove; i++)
                     {
                         outputBox.Items.RemoveAt(0);
                     }
                 }
             }

             // Auto-scroll to last item
             if (outputBox.Items.Count > 0)
             {
                 outputBox.TopIndex = outputBox.Items.Count - 1;
             }
         }
         catch (Exception ex)
         {
             System.Diagnostics.Debug.WriteLine($"[log] Error in AppendToLogBox ({boxName}): {ex.GetType().Name}: {ex.Message}");
         }
     }

    private void AppendExceptionReport(string context, Exception ex, string boxName = "AutoOutputBox")
    {
        AppendToLogBox(boxName, $"{context}: {ex.GetType().Name}: {ex.Message}");

        var depth = 1;
        var inner = ex.InnerException;
        while (inner != null && depth <= 8)
        {
            AppendToLogBox(boxName, $"  ↳ Inner {depth}: {inner.GetType().Name}: {inner.Message}");
            inner = inner.InnerException;
            depth++;
        }

        var flattened = FlattenExceptionMessages(ex).ToLowerInvariant();
        if (flattened.Contains("cuda out of memory", StringComparison.Ordinal) ||
            flattened.Contains("outofmemory", StringComparison.Ordinal))
        {
            AppendToLogBox(boxName, "  Hint: GPU memory exhausted. Reduce epochs/hidden size, switch to CPU, or restart app to release VRAM.");
        }
    }

    private static bool IsCudaOutOfMemory(Exception ex)
    {
        var flattened = FlattenExceptionMessages(ex).ToLowerInvariant();
        return flattened.Contains("cuda out of memory", StringComparison.Ordinal) ||
               flattened.Contains("outofmemory", StringComparison.Ordinal);
    }

    private void AggressiveMemoryCleanup(string phase)
    {
        try
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            AppendOutput($"[mem] cleanup completed ({phase})");
        }
        catch
        {
            // best effort only
        }
    }

    private static string FlattenExceptionMessages(Exception ex)
    {
        var parts = new List<string>();
        var current = ex;
        var depth = 0;
        while (current != null && depth < 12)
        {
            parts.Add(current.Message);
            current = current.InnerException;
            depth++;
        }
        return string.Join(" | ", parts);
    }

    private T? GetControl<T>(string name) where T : Control
    {
        foreach (var control in GetAllControls(this))
        {
            if (control.Name == name && control is T typed)
                return typed;
        }
        return null;
    }

    private IEnumerable<Control> GetAllControls(Control parent)
    {
        foreach (Control control in parent.Controls)
        {
            yield return control;
            foreach (var child in GetAllControls(control))
                yield return child;
        }
    }

}
