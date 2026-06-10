using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Runtime;
using GenesisNova.Train;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace GenesisNova.UI;

public class MainWindow : Form
{
    private const int MaxTrainingLogChars = 240_000;
    private const int TargetTrainingLogChars = 180_000;
    private const int MaxReplLogChars = 200_000;
    private const int TargetReplLogChars = 150_000;
    private const int AutoScrollSlackChars = 256;

    private readonly GenesisEvalAppRuntime _runtime;
    private CancellationTokenSource? _autonomousTrainingCts;
    private string _exampleFolder;
    private TabControl _tabControl = null!;
    private string? _lastReplUserInput;
    private string? _lastReplModelOutput;
    private bool _replTraceEnabled;
    private PlatonicActivationView? _latestActivation;
    private string _latestVisualizerRoute = "decision=n/a";
    private int _activeTrainingOperations;
    private SplitContainer? _replVisualizerSplit;
    private DateTime _lastTrimTime = DateTime.MinValue;
    private const int TrimDebounceMs = 500;  // Only trim if >500ms since last trim

    public MainWindow()
    {
        var reliableStateDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenesisNova",
            "models");
        _runtime = new GenesisEvalAppRuntime(new GenesisNovaConfig
        {
            Backend = ComputeBackend.Gpu,
            HiddenSize = 256,  // Conservative initial size; will be updated based on GPU VRAM after loading data
            AutoPersist = true,
            AutoResume = true,
            LocalStateDirectory = reliableStateDir,
            AutoScaleVram = true,
            TargetVramUtilization = 0.82,
            ReserveVramMb = 1536,
            TrainingTickMultiplier = 16
        });
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
        };
        
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _autonomousTrainingCts?.Cancel();
        _autonomousTrainingCts?.Dispose();
        _autonomousTrainingCts = null;
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

        Controls.Add(_tabControl);
    }

    private TabPage CreateAutonomousTrainingTab()
    {
        var tab = new TabPage("Autonomous");

        var mainLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            Padding = new Padding(12)
        };
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 34));
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 66));

        mainLayout.Controls.Add(CreateAutonomousConfigPanel(), 0, 0);
        mainLayout.Controls.Add(CreateAutonomousResultsPanel(), 1, 0);

        tab.Controls.Add(mainLayout);
        return tab;
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
        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(12)
        };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 110));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));

        var summary = new RichTextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            Font = new Font("Consolas", 9),
            BackColor = Color.FromArgb(20, 20, 20),
            ForeColor = Color.FromArgb(220, 220, 220),
            Name = "VizSummary",
            Text =
                "Reasoning diagnostics (how answer was reached):\n" +
                "- Route: whether output came from platonic query, neural fallback, or mixed path.\n" +
                "- Anchors: concepts found directly in your input.\n" +
                "- Nodes/edges: strongest concepts and links steering the output.\n" +
                "Run a REPL query to populate live evidence."
        };
        layout.Controls.Add(summary, 0, 0);

        var tables = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1
        };
        tables.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        tables.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

        var nodes = new ListView
        {
            Dock = DockStyle.Fill,
            View = View.Details,
            FullRowSelect = true,
            GridLines = true,
            Name = "VizNodes"
        };
        nodes.Columns.Add("Node", 150);
        nodes.Columns.Add("Influence", 75);
        nodes.Columns.Add("Seen", 65);
        nodes.Columns.Add("Anchor", 60);
        tables.Controls.Add(nodes, 0, 0);

        var edges = new ListView
        {
            Dock = DockStyle.Fill,
            View = View.Details,
            FullRowSelect = true,
            GridLines = true,
            Name = "VizEdges"
        };
        edges.Columns.Add("Left", 110);
        edges.Columns.Add("Right", 110);
        edges.Columns.Add("Trust", 60);
        edges.Columns.Add("Contrad", 65);
        edges.Columns.Add("Seen", 55);
        tables.Controls.Add(edges, 1, 0);
        layout.Controls.Add(tables, 0, 1);

        var graphPanel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(8, 10, 20),
            Name = "VizGraph"
        };
        graphPanel.Paint += (_, e) => DrawActivationGraph(e.Graphics, graphPanel.ClientRectangle);
        graphPanel.Resize += (_, _) => graphPanel.Invalidate();
        layout.Controls.Add(graphPanel, 0, 2);

        return layout;
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
            try
            {
                var result = await ExecuteReplPromptAsync(command);
                AppendToRepl($"{result}\n> ", Color.Cyan);
            }
            catch (Exception ex)
            {
                AppendToRepl($"Error: {ex.Message}\n> ", Color.Red);
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
                var report = await _runtime.TrainAsync(resolvedPath, epochs);
                return
                    $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4} " +
                    $"success={report.ExampleSuccessRate:P1}";
            }

            return $"Unknown command: {trimmed}";
        }

        var modelInput = trimmed;
        try
        {
            var predict = await _runtime.PredictAsync(modelInput);
            var output = predict.Result?.Output?.Trim();
            if (string.IsNullOrWhiteSpace(output))
                output = "(no response)";

            _lastReplUserInput = trimmed;
            _lastReplModelOutput = output;
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
  /trainfile <path> [epochs] - Train from file
  /help                      - Show this message";
    }

    private string BuildReplTrace(string input, GenesisNova.Infer.GenerationResult result, string output)
    {
        var inputTokenIds = _runtime.EncodeTokens(input);
        var generatedTokenIds = result.GeneratedTokens ?? [];
        var inputTokens = inputTokenIds.Select(id => $"{id}:{_runtime.TokenText(id)}");
        var outputTokens = generatedTokenIds.Select(id => $"{id}:{_runtime.TokenText(id)}");
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
        _latestVisualizerRoute =
            $"decision={result.DecisionPath}, platonic={result.UsedPlatonicQuery}, neuralFallback={result.UsedNeuralFallback}, confidence={result.PlatonicConfidence:F3}";

        var summary = GetControl<RichTextBox>("VizSummary");
        if (summary is not null)
        {
            var anchors = _latestActivation.Anchors.Length == 0
                ? "(none)"
                : string.Join(", ", _latestActivation.Anchors);
            var topNodes = _latestActivation.Nodes
                .Take(3)
                .Select(n => $"{n.Name}({n.Score:F2})")
                .DefaultIfEmpty("(none)")
                .ToArray();
            var topEdges = _latestActivation.Edges
                .Take(3)
                .Select(e => $"{e.Left}→{e.Right} trust={e.Score:F2} contrad={e.Contradiction:F2}")
                .DefaultIfEmpty("(none)")
                .ToArray();
            var routeFlavor = DescribeDecisionPath(result);
            var confidenceFlavor = DescribeConfidence(result.PlatonicConfidence);
            summary.Text =
                $"input: {input}\n" +
                $"answer route: {_latestVisualizerRoute}\n" +
                $"route flavor: {routeFlavor}\n" +
                $"confidence flavor: {confidenceFlavor}\n" +
                $"bias+hops: bias={result.AppliedBiasCount} avgBias={result.AverageBiasMagnitude:F4} hops={result.PlatonicHopCount} chunks={result.ChunksGenerated}\n" +
                $"anchors (direct input matches): {anchors}\n" +
                $"top node evidence: {string.Join(", ", topNodes)}\n" +
                $"top edge evidence: {string.Join(", ", topEdges)}\n" +
                $"tokens: {string.Join(", ", _latestActivation.InputTokens)}\n" +
                $"nodes shown: {_latestActivation.Nodes.Length}, edges shown: {_latestActivation.Edges.Length}";
        }

        var nodeList = GetControl<ListView>("VizNodes");
        if (nodeList is not null)
        {
            nodeList.BeginUpdate();
            nodeList.Items.Clear();
            foreach (var node in _latestActivation.Nodes)
            {
                var item = new ListViewItem(node.Name);
                item.SubItems.Add(node.Score.ToString("F3"));
                item.SubItems.Add(node.ObservationCount.ToString());
                item.SubItems.Add(node.IsAnchor ? "yes" : "");
                if (node.IsAnchor)
                    item.BackColor = Color.FromArgb(255, 246, 221);
                nodeList.Items.Add(item);
            }
            nodeList.EndUpdate();
        }

        var edgeList = GetControl<ListView>("VizEdges");
        if (edgeList is not null)
        {
            edgeList.BeginUpdate();
            edgeList.Items.Clear();
            foreach (var edge in _latestActivation.Edges)
            {
                var item = new ListViewItem(edge.Left);
                item.SubItems.Add(edge.Right);
                item.SubItems.Add(edge.Score.ToString("F3"));
                item.SubItems.Add(edge.Contradiction.ToString("F3"));
                item.SubItems.Add(edge.ObservationCount.ToString());
                edgeList.Items.Add(item);
            }
            edgeList.EndUpdate();
        }

        GetControl<Panel>("VizGraph")?.Invalidate();
    }

    private void DrawActivationGraph(Graphics g, Rectangle bounds)
    {
        g.Clear(Color.FromArgb(8, 10, 20));
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

        if (_latestActivation is null || _latestActivation.Nodes.Length == 0)
        {
            using var brush = new SolidBrush(Color.FromArgb(180, 180, 180));
            g.DrawString("No activation yet. Run a REPL query.", new Font("Segoe UI", 10), brush, new PointF(12, 12));
            return;
        }

        var nodes = _latestActivation.Nodes.Take(12).ToArray();
        var nodeSet = new HashSet<string>(nodes.Select(n => n.Name), StringComparer.OrdinalIgnoreCase);
        var edges = _latestActivation.Edges
            .Where(e => nodeSet.Contains(e.Left) && nodeSet.Contains(e.Right))
            .Take(24)
            .ToArray();

        var centerX = bounds.Width / 2f;
        var centerY = bounds.Height / 2f;
        var radius = Math.Max(40f, Math.Min(bounds.Width, bounds.Height) * 0.33f);
        var positions = new Dictionary<string, PointF>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < nodes.Length; i++)
        {
            var angle = (Math.PI * 2 * i) / Math.Max(1, nodes.Length);
            var x = centerX + (float)(Math.Cos(angle) * radius);
            var y = centerY + (float)(Math.Sin(angle) * radius);
            positions[nodes[i].Name] = new PointF(x, y);
        }

        foreach (var edge in edges)
        {
            if (!positions.TryGetValue(edge.Left, out var p1) || !positions.TryGetValue(edge.Right, out var p2))
                continue;

            var alpha = (int)Math.Clamp(50 + (edge.Score * 180), 50, 230);
            using var pen = new Pen(Color.FromArgb(alpha, 120, 180, 255), 1.5f);
            g.DrawLine(pen, p1, p2);
        }

        foreach (var node in nodes)
        {
            if (!positions.TryGetValue(node.Name, out var p))
                continue;

            var size = 16f + (float)(node.Score * 16f);
            var rect = new RectangleF(p.X - size / 2f, p.Y - size / 2f, size, size);
            var fillColor = node.IsAnchor
                ? Color.FromArgb(255, 190, 90)
                : Color.FromArgb(90, 160, 255);
            using var brush = new SolidBrush(fillColor);
            using var outline = new Pen(Color.FromArgb(20, 20, 20), 1.5f);
            g.FillEllipse(brush, rect);
            g.DrawEllipse(outline, rect);

            using var labelBrush = new SolidBrush(Color.FromArgb(230, 230, 230));
            g.DrawString(node.Name, new Font("Segoe UI", 8), labelBrush, p.X + size / 2f + 2f, p.Y - 8f);
        }
    }

    private static string DescribeDecisionPath(GenesisNova.Infer.GenerationResult result)
    {
        if (result.UsedPlatonicQuery && !result.UsedNeuralFallback)
            return "Direct platonic retrieval dominated this answer.";
        if (result.UsedPlatonicQuery && result.UsedNeuralFallback)
            return "Hybrid path: platonic retrieval plus neural fallback.";
        if (result.AppliedBiasCount > 0)
            return "Neural decode with platonic bias shaping token choices.";
        return "Pure neural decode path.";
    }

    private static string DescribeConfidence(double confidence)
    {
        if (confidence >= 0.85)
            return "High confidence: strong structural match.";
        if (confidence >= 0.60)
            return "Moderate confidence: useful structure, some ambiguity.";
        if (confidence > 0.0)
            return "Low confidence: weak structural support.";
        return "No platonic confidence signal (neural-only path).";
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
            Text = "▶ Start Autonomous",
            Width = 300,
            Height = 40,
            BackColor = Color.FromArgb(51, 153, 102),
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 11, FontStyle.Bold),
            Name = "AutoTrainBtn"
        };
        startBtn.Click += async (s, e) => await StartAutonomousTraining();
        layout.Controls.Add(startBtn);

        var stopBtn = new Button
        {
            Text = "⏹ Stop Autonomous",
            Width = 300,
            Height = 40,
            BackColor = Color.FromArgb(204, 51, 51),
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 11, FontStyle.Bold),
            Name = "AutoStopBtn",
            Enabled = false
        };
        stopBtn.Click += (s, e) => StopAutonomousTraining();
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
            MaxGenerationConcurrency: (int)(generationConcurrency?.Value ?? defaults.GenerationConcurrency));
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
            AppendAutonomousOutput($"[auto] starting: rounds={roundsLabel} sample={request.InitialSampleCount} train={request.InitialTrainCount} difficulty={request.InitialDifficulty}");
            AppendAutonomousOutput("[auto] device policy: CUDA training when available with CPU fallback, GPU inference (device 0, A3000 preferred)");
            
            // Estimate and display GPU sizing for this training session using live VRAM.
            var debugLines = new List<string>();
            var hasCuda = GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalVramMb, out var freeVramMb);
            var sizingVramMb = hasCuda ? (freeVramMb > 0 ? freeVramMb : totalVramMb) : 4096;
            var sizingTargetUtil = 0.72;
            var sizingReserveMb = 1536;
            var sizingHiddenCap = GpuCapacityPlanner.ResolveTrainingHiddenCap(sizingVramMb);
            AppendAutonomousOutput($"[auto] training headroom cap: {sizingHiddenCap}");
            var estimatedHidden = GpuCapacityPlanner.EstimateHiddenSizeForInferenceOnly(
                new[] { new GenesisExample("sample", "output") },
                routeCount: 8,
                vramMb: sizingVramMb,
                targetUtilization: sizingTargetUtil,
                reserveVramMb: sizingReserveMb,
                debugOutput: line => debugLines.Add(line),
                maxHiddenSize: sizingHiddenCap);
            foreach (var line in debugLines)
                AppendAutonomousOutput(line);

            // Extra guard for autonomous rounds on constrained VRAM.
            if (hasCuda)
            {
                if (estimatedHidden > sizingHiddenCap)
                {
                    AppendAutonomousOutput(
                        $"[auto] hidden cap applied for training headroom: {estimatedHidden} → {sizingHiddenCap} (usable_vram_mb={sizingVramMb})");
                    estimatedHidden = sizingHiddenCap;
                }
            }
            AppendAutonomousOutput(
                $"[auto] sizing profile: target_util={sizingTargetUtil:F2} reserve_mb={sizingReserveMb} hidden_cap={sizingHiddenCap} usable_vram_mb={sizingVramMb}");
             
            // CRITICAL: Apply the estimated hidden size BEFORE starting training
            AppendAutonomousOutput($"[auto] applying hidden size: {_runtime.HiddenSize} → {estimatedHidden}");
            _runtime.EnsureHiddenSize(estimatedHidden);
            AppendAutonomousOutput($"[auto] model ready: hidden={_runtime.HiddenSize}");
            
            AppendAutonomousOutput("[auto] live controls update next round: max rounds, loss threshold, bounds, max difficulty, round budget, and generation concurrency.");
            // Run entire training pipeline on ThreadPool with reduced priority to prevent UI thread starvation
            var run = await RunLowPriorityTrainingAsync(
                async () => await _runtime.TrainAutonomousAsync(
                    request,
                    _autonomousTrainingCts.Token,
                    AppendAutonomousOutput,
                    baseRequest => CaptureLiveAutonomousRequest(baseRequest),
                    onRoundProgress: payload => HandleAutonomousTrainingProgress((GenesisAutonomousTrainingEventPayload)payload)),
                _autonomousTrainingCts.Token);
            var final = run.FinalReport;
            if (final is not null)
            {
                AppendAutonomousOutput($"[auto] complete: rounds={run.Rounds.Count} final_loss={final.AverageLoss.TokenLoss:F4}");
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
                AppendAutonomousOutput("[auto] complete: no report returned");
            }
        }
        catch (OperationCanceledException)
        {
            AppendAutonomousOutput("[auto] stopped by user.");
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

        AppendAutonomousOutput("[auto] stop requested: finishing current round...");
        _autonomousTrainingCts.Cancel();
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
            AppendAutonomousOutput($"[ui] Error updating progress: {ex.Message}");
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
            MaxGenerationConcurrency = live.MaxGenerationConcurrency
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
            var report = await _runtime.TrainAsync(
                filePath, 
                epochs, 
                logPath: logPath);
            
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
        AppendToLogBox("AutoOutputBox", message);
    }

    private void AppendAutonomousOutput(string message)
    {
        AppendToLogBox("AutoOutputBox", message);
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

    private static void TrimRichTextHistory(RichTextBox box, int maxChars, int targetChars)
    {
        // Deprecated: ListBox handles trimming internally in AppendToLogBox
    }

    private static bool IsNearBottom(RichTextBox box)
    {
        // Deprecated: ListBox auto-scrolls in AppendToLogBox
        return true;
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
