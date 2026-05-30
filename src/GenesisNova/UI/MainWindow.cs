using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Runtime;
using GenesisNova.Train;
using System.Diagnostics;

namespace GenesisNova.UI;

public class MainWindow : Form
{
    private const int MaxTrainingLogChars = 240_000;
    private const int TargetTrainingLogChars = 180_000;
    private const int MaxReplLogChars = 200_000;
    private const int TargetReplLogChars = 150_000;
    private const int AutoScrollSlackChars = 256;

    private sealed record GeneratorTrainingRequest(
        string CreatorName,
        int SampleCount,
        int Difficulty,
        int Epochs,
        bool ProgressiveCycle,
        double LossThreshold,
        int MaxCurriculumRounds);

    private readonly GenesisEvalAppRuntime _runtime;
    private CancellationTokenSource? _trainingCts;
    private CancellationTokenSource? _autonomousTrainingCts;
    private string _exampleFolder;
    private readonly Dictionary<string, string> _trainingFilePathByLabel = new(StringComparer.OrdinalIgnoreCase);
    private TabControl _tabControl = null!;
    private string? _lastReplUserInput;
    private string? _lastReplModelOutput;
    private bool _replTraceEnabled;
    private PlatonicActivationView? _latestActivation;
    private string _latestVisualizerRoute = "decision=n/a";
    private int _activeTrainingOperations;
    private SplitContainer? _replVisualizerSplit;

    public MainWindow()
    {
        _runtime = new GenesisEvalAppRuntime(new GenesisNovaConfig
        {
            Backend = ComputeBackend.Gpu,
            HiddenSize = 8192,
            AutoPersist = true,
            AutoResume = true,
            AutoScaleVram = true,
            TargetVramUtilization = 0.9,
            ReserveVramMb = 512
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
        _trainingCts?.Cancel();
        _trainingCts?.Dispose();
        _trainingCts = null;
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

        // Tab 1: Training
        _tabControl.TabPages.Add(CreateTrainingTab());
        
        // Tab 2: Autonomous Training
        _tabControl.TabPages.Add(CreateAutonomousTrainingTab());

        // Tab 3: REPL
        _tabControl.TabPages.Add(CreateReplTab());

        Controls.Add(_tabControl);
    }

    private TabPage CreateTrainingTab()
    {
        var tab = new TabPage("Training");
        
        var mainLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            Padding = new Padding(12)
        };
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
        mainLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 70));

        var leftPanel = CreateConfigPanel();
        mainLayout.Controls.Add(leftPanel, 0, 0);

        var rightPanel = CreateResultsPanel();
        mainLayout.Controls.Add(rightPanel, 1, 0);

        tab.Controls.Add(mainLayout);
        RefreshExampleFiles();
        return tab;
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
        var outputBox = new RichTextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            Font = new Font("Consolas", 10),
            BackColor = Color.Black,
            ForeColor = Color.Lime,
            Name = "ReplOutput"
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
            Text = "Run a REPL query to see activated nodes and routes."
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
        nodes.Columns.Add("Node", 160);
        nodes.Columns.Add("Score", 70);
        nodes.Columns.Add("Obs", 70);
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
        edges.Columns.Add("Left", 120);
        edges.Columns.Add("Right", 120);
        edges.Columns.Add("Score", 70);
        edges.Columns.Add("Obs", 70);
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
                return $"trained examples={report.ExampleCount} epochs={report.Epochs} loss={report.AverageLoss.TotalLoss:F4}";
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
            summary.Text =
                $"input: {input}\n" +
                $"tokens: {string.Join(", ", _latestActivation.InputTokens)}\n" +
                $"anchors: {anchors}\n" +
                $"{_latestVisualizerRoute}\n" +
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

    private void AppendToRepl(string text, Color color)
    {
        var box = GetControl<RichTextBox>("ReplOutput");
        if (box == null || box.IsDisposed)
            return;

        if (box.InvokeRequired)
        {
            box.Invoke(() => AppendToRepl(text, color));
            return;
        }

        var shouldAutoScroll = IsNearBottom(box);
        box.SelectionColor = color;
        box.AppendText(text);
        TrimRichTextHistory(box, MaxReplLogChars, TargetReplLogChars);
        if (shouldAutoScroll)
        {
            box.SelectionStart = box.TextLength;
            box.SelectionLength = 0;
            box.ScrollToCaret();
        }
    }

    private Panel CreateConfigPanel()
    {
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

        var titleLabel = new Label { Text = "Configuration", Font = new Font("Segoe UI", 14, FontStyle.Bold), Height = 30 };
        layout.Controls.Add(titleLabel);

        layout.Controls.Add(new Label { Text = "Data Generator:", Height = 20, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        
        layout.Controls.Add(new Label { Text = "Creator:", Height = 20, Font = new Font("Segoe UI", 9) });
        var creatorCombo = new ComboBox
        {
            Width = 280,
            Height = 32,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Name = "CreatorCombo"
        };
        
        // Populate with all available creators
        foreach (var creator in ExampleCreatorRegistry.All)
        {
            creatorCombo.Items.Add(creator.Name);
        }
        if (creatorCombo.Items.Count > 0) creatorCombo.SelectedIndex = 0;
        layout.Controls.Add(creatorCombo);

        layout.Controls.Add(new Label { Text = "Sample Count:", Height = 20, Font = new Font("Segoe UI", 9) });
        var countInput = new NumericUpDown { Minimum = 10, Maximum = 10000, Value = 100, Width = 280, Height = 32, Name = "SampleCountInput" };
        layout.Controls.Add(countInput);

        layout.Controls.Add(new Label { Text = "Difficulty:", Height = 20, Font = new Font("Segoe UI", 9) });
        var difficultyInput = new NumericUpDown { Minimum = 0, Maximum = 100, Value = 1, Width = 280, Height = 32, Name = "DifficultyInput" };
        layout.Controls.Add(difficultyInput);
        var progressiveCheck = new CheckBox { Text = "Progressive Difficulty Cycle", Width = 280, Height = 24, Checked = true, Name = "ProgressiveCycleCheck" };
        layout.Controls.Add(progressiveCheck);
        layout.Controls.Add(new Label { Text = "Difficulty Threshold (loss):", Height = 20, Font = new Font("Segoe UI", 9) });
        var thresholdInput = new NumericUpDown { Minimum = 0, Maximum = 10, DecimalPlaces = 3, Increment = 0.005M, Value = 0.100M, Width = 280, Height = 32, Name = "DifficultyThresholdInput" };
        layout.Controls.Add(thresholdInput);
        layout.Controls.Add(new Label { Text = "Max Curriculum Rounds:", Height = 20, Font = new Font("Segoe UI", 9) });
        var maxRoundsInput = new NumericUpDown { Minimum = 1, Maximum = int.MaxValue, Value = 30, Width = 280, Height = 32, Name = "MaxCurriculumRoundsInput" };
        layout.Controls.Add(maxRoundsInput);

        layout.Controls.Add(new Label { Height = 10 });
        layout.Controls.Add(new Label { Text = "Training Parameters:", Height = 20, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        layout.Controls.Add(new Label
        {
            Text = "Policy: training stays on CPU; inference uses the GPU-backed model when available.",
            Width = 280,
            Height = 40
        });
        var epochsInput = new NumericUpDown { Value = 3, Minimum = 1, Maximum = 100, Width = 280, Height = 32, Name = "EpochsInput" };
        layout.Controls.Add(epochsInput);

        layout.Controls.Add(new Label { Height = 10 });
        layout.Controls.Add(new Label { Text = "Model Compression:", Height = 20, Font = new Font("Segoe UI", 10, FontStyle.Bold) });
        layout.Controls.Add(new Label
        {
            Text = "L2 Regularization: Penalize large weights to force learning into platonic space (symbolic layer). Under pressure, the model learns to compress knowledge symbolically.",
            Width = 280,
            Height = 50,
            AutoSize = false
        });
        
        var l2ComboLabel = new Label { Text = "L2 Weight Decay Preset:", Height = 20, Font = new Font("Segoe UI", 9) };
        layout.Controls.Add(l2ComboLabel);
        var l2Combo = new ComboBox
        {
            Width = 280,
            Height = 32,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Name = "L2PresetCombo"
        };
        l2Combo.Items.Add("Off (0.0) - No compression");
        l2Combo.Items.Add("Mild (1e-5) - Neural layer comfortable");
        l2Combo.Items.Add("Balanced (1e-4)");
        l2Combo.Items.Add("Aggressive (1e-3) - Force symbolic learning ⚡");
        l2Combo.Items.Add("Extreme (1e-2) - Nearly all symbolic");
        l2Combo.SelectedIndex = 0;  // Default to off
        l2Combo.SelectedIndexChanged += (s, e) => ApplyL2Preset(l2Combo.SelectedIndex);
        layout.Controls.Add(l2Combo);

        layout.Controls.Add(new Label { Height = 10 });

        var trainBtn = new Button { Text = "▶ Start Training", Width = 280, Height = 40, BackColor = Color.FromArgb(51, 153, 102), ForeColor = Color.White, Font = new Font("Segoe UI", 11, FontStyle.Bold), Name = "TrainBtn" };
        trainBtn.Click += (s, e) =>
        {
            trainBtn.Enabled = false;
            var stopBtn = GetControl<Button>("StopBtn");
            if (stopBtn != null) stopBtn.Enabled = true;

            var request = TryBuildGeneratorTrainingRequest();
            if (request is null)
            {
                trainBtn.Enabled = true;
                if (stopBtn != null) stopBtn.Enabled = false;
                return;
            }

            _ = Task.Run(async () =>
            {
                try
                {
                    await StartTrainingWithGenerator(request);
                }
                catch (OperationCanceledException)
                {
                    AppendOutput("Training stopped by user.");
                }
                catch (Exception ex)
                {
                    AppendExceptionReport("Training failed", ex);
                }
                finally
                {
                    if (IsHandleCreated)
                    {
                        BeginInvoke(new Action(() =>
                        {
                            trainBtn.Enabled = true;
                            if (stopBtn != null) stopBtn.Enabled = false;
                        }));
                    }
                }
            });
        };
        layout.Controls.Add(trainBtn);

        var stopBtn = new Button { Text = "⏹ Stop Training", Width = 280, Height = 40, BackColor = Color.FromArgb(204, 51, 51), ForeColor = Color.White, Font = new Font("Segoe UI", 11, FontStyle.Bold), Name = "StopBtn", Enabled = false };
        stopBtn.Click += (s, e) => StopTraining();
        layout.Controls.Add(stopBtn);

        panel.Controls.Add(layout);
        return panel;
    }

    private Panel CreateAutonomousConfigPanel()
    {
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
            Text = "The model will pick the next creator, difficulty, sample count, and train micro-batch from feedback.",
            Width = 300,
            Height = 40
        });

        layout.Controls.Add(new Label { Text = "Starting Creator:", Height = 20, Font = new Font("Segoe UI", 9) });
        var creatorCombo = new ComboBox
        {
            Width = 300,
            Height = 32,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Name = "AutoCreatorCombo"
        };
        creatorCombo.Items.Add("Auto (cycle creators)");
        foreach (var creator in ExampleCreatorRegistry.All)
            creatorCombo.Items.Add(creator.Name);
        creatorCombo.SelectedIndex = 0;
        layout.Controls.Add(creatorCombo);

        layout.Controls.Add(new Label { Text = "Max Rounds:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = int.MaxValue,
            Value = 12,
            Width = 300,
            Height = 32,
            Name = "AutoMaxRoundsInput"
        });
        layout.Controls.Add(new Label
        {
            Text = "Set to 0 for no limit.",
            Width = 300,
            Height = 20,
            Font = new Font("Segoe UI", 8),
            ForeColor = Color.DimGray
        });

        layout.Controls.Add(new Label { Text = "Initial Sample Count:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 4,
            Maximum = 10000,
            Value = 24,
            Width = 300,
            Height = 32,
            Name = "AutoSampleCountInput"
        });

        layout.Controls.Add(new Label { Text = "Initial Difficulty:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 100,
            Value = 0,
            Width = 300,
            Height = 32,
            Name = "AutoDifficultyInput"
        });

        layout.Controls.Add(new Label { Text = "Initial Epochs:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 64,
            Value = 1,
            Width = 300,
            Height = 32,
            Name = "AutoEpochsInput"
        });

        layout.Controls.Add(new Label { Text = "Initial Train Count:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 1,
            Maximum = 128,
            Value = 4,
            Width = 300,
            Height = 32,
            Name = "AutoTrainCountInput"
        });

        layout.Controls.Add(new Label { Text = "Loss Threshold:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 10,
            DecimalPlaces = 3,
            Increment = 0.005M,
            Value = 1.200M,
            Width = 300,
            Height = 32,
            Name = "AutoLossThresholdInput"
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
                    Minimum = 4,
                    Maximum = 10000,
                    Value = 12,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMinSampleInput"
                },
                new Label { Text = "Max", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 8,
                    Maximum = 10000,
                    Value = 128,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMaxSampleInput"
                }
            }
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
                    Value = 2,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMinTrainInput"
                },
                new Label { Text = "Max", Width = 32, TextAlign = ContentAlignment.MiddleLeft },
                new NumericUpDown
                {
                    Minimum = 1,
                    Maximum = 256,
                    Value = 8,
                    Width = 80,
                    Height = 28,
                    Name = "AutoMaxTrainInput"
                }
            }
        });

        layout.Controls.Add(new Label { Text = "Max Difficulty:", Height = 20, Font = new Font("Segoe UI", 9) });
        layout.Controls.Add(new NumericUpDown
        {
            Minimum = 0,
            Maximum = 100,
            Value = 8,
            Width = 300,
            Height = 32,
            Name = "AutoMaxDifficultyInput"
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

        layout.Controls.Add(new RichTextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            Font = new Font("Consolas", 9),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.FromArgb(200, 220, 200),
            Name = "AutoOutputBox"
        }, 0, 1);

        layout.Controls.Add(new Label
        {
            Text = "Rounds: 0 | Last loss: n/a | Creator: n/a",
            Font = new Font("Consolas", 10),
            Dock = DockStyle.Fill,
            AutoSize = false,
            Name = "AutoStatsText"
        }, 0, 2);

        panel.Controls.Add(layout);
        return panel;
    }

    private Panel CreateResultsPanel()
    {
        var panel = new Panel { Dock = DockStyle.Fill, BackColor = Color.White, Padding = new Padding(12) };
        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(0)
        };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 70));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 30));

        var titleLabel = new Label { Text = "Training Results", Font = new Font("Segoe UI", 14, FontStyle.Bold), Dock = DockStyle.Fill };
        layout.Controls.Add(titleLabel, 0, 0);

        var outputBox = new RichTextBox
        {
            Dock = DockStyle.Fill,
            ReadOnly = true,
            Font = new Font("Consolas", 9),
            BackColor = Color.FromArgb(30, 30, 30),
            ForeColor = Color.FromArgb(200, 200, 200),
            Name = "OutputBox"
        };
        layout.Controls.Add(outputBox, 0, 1);

        var statsLayout = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, BackColor = Color.FromArgb(245, 245, 245), Padding = new Padding(8) };
        
        var statsLabel = new Label { Text = "Model Statistics", Font = new Font("Segoe UI", 11, FontStyle.Bold), Height = 25, AutoSize = false };
        statsLayout.Controls.Add(statsLabel);

        var statsText = new Label
        {
            Text = BuildStatsText(exampleCount: null, loss: null),
            Font = new Font("Consolas", 10),
            Height = 72,
            AutoSize = false,
            Name = "StatsText"
        };
        statsLayout.Controls.Add(statsText);

        var predictLayout = new FlowLayoutPanel { FlowDirection = FlowDirection.LeftToRight, Height = 40, AutoSize = false };
        var predictLabel = new Label { Text = "Predict:", Width = 70, TextAlign = ContentAlignment.MiddleLeft };
        var predictInput = new TextBox { Width = 150, Height = 25, Name = "PredictInput" };
        var predictBtn = new Button { Text = "Run", Width = 60, Height = 25, Name = "PredictBtn" };
        predictBtn.Click += async (s, e) => await RunPrediction(predictInput);
        var predictOutput = new Label { Text = "", Width = 150, AutoSize = false, Name = "PredictOutput" };
        
        predictLayout.Controls.AddRange(new Control[] { predictLabel, predictInput, predictBtn, predictOutput });
        statsLayout.Controls.Add(predictLayout);

        layout.Controls.Add(statsLayout, 0, 2);

        panel.Controls.Add(layout);
        return panel;
    }

    private void RefreshExampleFiles()
    {
        var filesListBox = GetControl<ListBox>("FilesList");
        if (filesListBox == null) return;
        var folderBox = GetControl<TextBox>("TrainingFolderBox");
        if (folderBox != null)
            _exampleFolder = folderBox.Text;

        filesListBox.Items.Clear();
        _trainingFilePathByLabel.Clear();
        try
        {
            if (!Directory.Exists(_exampleFolder))
                Directory.CreateDirectory(_exampleFolder);

            var files = Directory
                .EnumerateFiles(_exampleFolder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(path =>
                {
                    var ext = Path.GetExtension(path);
                    return ext.Equals(".txt", StringComparison.OrdinalIgnoreCase) ||
                           ext.Equals(".jsonl", StringComparison.OrdinalIgnoreCase) ||
                           ext.Equals(".data", StringComparison.OrdinalIgnoreCase);
                })
                .OrderByDescending(path => new FileInfo(path).Length)
                .ThenBy(path => path, StringComparer.OrdinalIgnoreCase);

            foreach (var file in files)
            {
                var info = new FileInfo(file);
                var label = $"{info.Name} ({Math.Max(1, info.Length / 1024)}KB)";
                _trainingFilePathByLabel[label] = file;
                filesListBox.Items.Add(label);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error loading files: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private GeneratorTrainingRequest? TryBuildGeneratorTrainingRequest()
    {
        var creatorCombo = GetControl<ComboBox>("CreatorCombo");
        var countInput = GetControl<NumericUpDown>("SampleCountInput");
        var difficultyInput = GetControl<NumericUpDown>("DifficultyInput");

        if (creatorCombo == null || creatorCombo.SelectedIndex < 0)
        {
            MessageBox.Show("Please select a creator", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return null;
        }

        var creatorName = creatorCombo.SelectedItem?.ToString();
        if (string.IsNullOrWhiteSpace(creatorName))
        {
            MessageBox.Show("Please select a creator", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return null;
        }

        var sampleCount = (int)(countInput?.Value ?? 100);
        var difficulty = (int)(difficultyInput?.Value ?? 1);
        var epochs = (int)(GetControl<NumericUpDown>("EpochsInput")?.Value ?? 3);
        var progressiveCycle = GetControl<CheckBox>("ProgressiveCycleCheck")?.Checked ?? true;
        var lossThreshold = (double)(GetControl<NumericUpDown>("DifficultyThresholdInput")?.Value ?? 0.100M);
        var maxCurriculumRounds = (int)(GetControl<NumericUpDown>("MaxCurriculumRoundsInput")?.Value ?? 30);

        return new GeneratorTrainingRequest(
            CreatorName: creatorName,
            SampleCount: sampleCount,
            Difficulty: difficulty,
            Epochs: epochs,
            ProgressiveCycle: progressiveCycle,
            LossThreshold: lossThreshold,
            MaxCurriculumRounds: maxCurriculumRounds);
    }

    private async Task StartTrainingWithGenerator(GeneratorTrainingRequest request)
    {
        var creatorName = request.CreatorName;
        var sampleCount = request.SampleCount;
        var difficulty = request.Difficulty;

        // Find the creator by name
        var creator = ExampleCreatorRegistry.All.FirstOrDefault(c => c.Name == creatorName);
        if (creator == null)
        {
            AppendOutput($"Creator not found: {creatorName}");
            return;
        }

        _trainingCts = new CancellationTokenSource();

        try
        {
            if (!request.ProgressiveCycle)
            {
                await RunSingleGeneratorRound(
                    creator,
                    creatorName,
                    sampleCount,
                    difficulty,
                    request.Epochs,
                    suffix: "single");
                return;
            }

            AppendOutput($"[curriculum] Starting progressive cycle 0→{difficulty}, threshold={request.LossThreshold:F4}, maxRounds={request.MaxCurriculumRounds}");
            var currentDifficulty = 0;
            var round = 0;

            while (currentDifficulty <= difficulty && round < request.MaxCurriculumRounds)
            {
                round++;
                if (_trainingCts?.IsCancellationRequested == true)
                    throw new OperationCanceledException();

                AppendOutput($"[curriculum] Round {round}: training difficulty={currentDifficulty}");
                var report = await RunSingleGeneratorRound(
                    creator,
                    creatorName,
                    sampleCount,
                    currentDifficulty,
                    request.Epochs,
                    suffix: $"d{currentDifficulty}_r{round}");

                var loss = report.AverageLoss.TokenLoss;
                if (loss <= request.LossThreshold)
                {
                    AppendOutput($"[curriculum] Threshold met at difficulty={currentDifficulty} (loss={loss:F4}) → advancing");
                    currentDifficulty++;
                }
                else
                {
                    AppendOutput($"[curriculum] Threshold not met at difficulty={currentDifficulty} (loss={loss:F4}) → repeating");
                }
            }

            if (currentDifficulty > difficulty)
                AppendOutput($"[curriculum] Completed through difficulty {difficulty}.");
            else
                AppendOutput($"[curriculum] Stopped at difficulty {currentDifficulty} after {round} rounds (max rounds reached).");
        }
        finally
        {
            _trainingCts?.Dispose();
            _trainingCts = null;
        }
    }

    private GenesisAutonomousTrainingRequest? TryBuildAutonomousTrainingRequest()
    {
        var creatorCombo = GetControl<ComboBox>("AutoCreatorCombo");
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

        if (creatorCombo == null || creatorCombo.SelectedIndex < 0)
            return null;

        var preferredCreator = creatorCombo.SelectedIndex == 0
            ? null
            : creatorCombo.SelectedItem?.ToString();

        return new GenesisAutonomousTrainingRequest(
            MaxRounds: (int)(maxRounds?.Value ?? 12),
            InitialSampleCount: (int)(sampleCount?.Value ?? 24),
            InitialDifficulty: (int)(difficulty?.Value ?? 0),
            InitialEpochs: (int)(epochs?.Value ?? 1),
            InitialTrainCount: (int)(trainCount?.Value ?? 4),
            LossThreshold: (double)(lossThreshold?.Value ?? 1.200M),
            MinSampleCount: (int)(minSample?.Value ?? 12),
            MaxSampleCount: (int)(maxSample?.Value ?? 128),
            MinTrainCount: (int)(minTrain?.Value ?? 2),
            MaxTrainCount: (int)(maxTrain?.Value ?? 8),
            MaxDifficulty: (int)(maxDifficulty?.Value ?? 8),
            PreferredCreator: preferredCreator);
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
            var run = await _runtime.TrainAutonomousAsync(request, _autonomousTrainingCts.Token, AppendAutonomousOutput);
            var final = run.FinalReport;
            if (final is not null)
            {
                AppendAutonomousOutput($"[auto] complete: rounds={run.Rounds.Count} final_loss={final.AverageLoss.TokenLoss:F4}");
                UpdateAutonomousStats(run.Rounds.Count, final.AverageLoss.TokenLoss, run.Rounds.LastOrDefault()?.CreatorName ?? "n/a");
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

    private void StopAutonomousTraining()
    {
        if (_autonomousTrainingCts is null || _autonomousTrainingCts.IsCancellationRequested)
            return;

        AppendAutonomousOutput("[auto] stop requested: finishing current round...");
        _autonomousTrainingCts.Cancel();
    }

    private async Task<GenesisTrainingReport> RunSingleGeneratorRound(
        IExampleCreator creator,
        string creatorName,
        int sampleCount,
        int difficulty,
        int epochs,
        string suffix)
    {
        AppendOutput($"Generating {sampleCount} examples from '{creatorName}' at difficulty {difficulty}...");
        var examples = creator.Generate(sampleCount, difficulty, forTraining: true);
        AppendOutput($"Generated {examples.Length} examples");

        var safeCreator = creatorName.Replace(':', '_');
        var tmpPath = Path.Combine(
            AppContext.BaseDirectory,
            $"generated_{safeCreator}_{difficulty}_{suffix}_{Guid.NewGuid():N}.txt");
        try
        {
            var lines = examples.Select(static ex => $"{ex.Input} => {ex.Output}");
            await File.WriteAllLinesAsync(tmpPath, lines);

            return await ExecuteTraining(tmpPath, epochs);
        }
        catch (Exception ex)
        {
            AppendOutput($"Error saving generated data: {ex.Message}");
            throw;
        }
        finally
        {
            try { File.Delete(tmpPath); } catch { }
        }
    }

    private async Task StartTraining(ListBox filesListBox)
    {
        if (filesListBox.SelectedIndex < 0)
        {
            MessageBox.Show("Please select a training file", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var selectedFile = filesListBox.SelectedItem?.ToString();
        if (selectedFile == null) return;

        var filePath = _trainingFilePathByLabel.TryGetValue(selectedFile, out var mappedPath)
            ? mappedPath
            : Path.Combine(_exampleFolder, selectedFile.Split(' ')[0]);
        
        if (!File.Exists(filePath))
        {
            MessageBox.Show($"File not found: {filePath}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        var epochs = (int)(GetControl<NumericUpDown>("EpochsInput")?.Value ?? 3);

        var trainBtn = GetControl<Button>("TrainBtn");
        var stopBtn = GetControl<Button>("StopBtn");
        if (trainBtn != null) trainBtn.Enabled = false;
        if (stopBtn != null) stopBtn.Enabled = true;

        _trainingCts = new CancellationTokenSource();
        
        try
        {
            _ = await ExecuteTraining(filePath, epochs);
        }
        catch (OperationCanceledException)
        {
            AppendOutput("Training stopped by user.");
        }
        catch (Exception ex)
        {
            AppendExceptionReport("Training failed", ex);
        }
        finally
        {
            if (trainBtn != null) trainBtn.Enabled = true;
            if (stopBtn != null) stopBtn.Enabled = false;
            _trainingCts?.Dispose();
            _trainingCts = null;
        }
    }

    private async Task<GenesisTrainingReport> ExecuteTraining(string filePath, int epochs)
    {
        var sw = Stopwatch.StartNew();
        Interlocked.Increment(ref _activeTrainingOperations);
        AppendOutput($"Loading training data from: {Path.GetFileName(filePath)}");
        
        var examples = GenesisTrainingDataLoader.LoadFromFile(filePath);
        AppendOutput($"Loaded {examples.Count} examples");
        AppendOutput($"Training: {examples.Count} examples, {epochs} epochs, policy=CPU-train/VRAM-infer");
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
                $"Space manager: cycles={report.SpaceManagementCycles} pruned={report.NodesPruned}n/{report.RelationsPruned}r " +
                $"final={report.FinalNodeCount}n/{report.FinalRelationCount}r noise={report.SpaceNoiseRatio:F3}");
            UpdateStats(examples.Count, report.AverageLoss.TokenLoss);
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

    private void StopTraining()
    {
        _trainingCts?.Cancel();
    }

    private void ChooseTrainingFolder()
    {
        using var dialog = new FolderBrowserDialog
        {
            Description = "Select folder containing training files",
            InitialDirectory = Directory.Exists(_exampleFolder) ? _exampleFolder : AppContext.BaseDirectory,
            ShowNewFolderButton = false
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
            return;

        _exampleFolder = dialog.SelectedPath;
        var folderBox = GetControl<TextBox>("TrainingFolderBox");
        if (folderBox != null)
            folderBox.Text = _exampleFolder;
        RefreshExampleFiles();
    }

    private void ChooseTrainingFile()
    {
        using var dialog = new OpenFileDialog
        {
            Title = "Choose training file",
            InitialDirectory = Directory.Exists(_exampleFolder) ? _exampleFolder : AppContext.BaseDirectory,
            Filter = "Training files (*.txt;*.jsonl;*.data)|*.txt;*.jsonl;*.data|All files (*.*)|*.*",
            Multiselect = false
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
            return;

        var path = dialog.FileName;
        _exampleFolder = Path.GetDirectoryName(path) ?? _exampleFolder;

        var folderBox = GetControl<TextBox>("TrainingFolderBox");
        if (folderBox != null)
            folderBox.Text = _exampleFolder;

        RefreshExampleFiles();
        var filesListBox = GetControl<ListBox>("FilesList");
        if (filesListBox == null)
            return;

        var selectedLabel = _trainingFilePathByLabel
            .FirstOrDefault(kv => kv.Value.Equals(path, StringComparison.OrdinalIgnoreCase))
            .Key;
        if (!string.IsNullOrWhiteSpace(selectedLabel))
            filesListBox.SelectedItem = selectedLabel;
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

    private async Task RunPrediction(TextBox input)
    {
        if (string.IsNullOrWhiteSpace(input.Text))
            return;

        var predictBtn = GetControl<Button>("PredictBtn");
        if (predictBtn != null) predictBtn.Enabled = false;
        
        try
        {
            var result = await _runtime.PredictAsync(input.Text);
            var output = result.Result?.Output ?? "No output";
            var route = result.Result?.DecisionPath ?? "unknown";
            var confidence = result.Result?.PlatonicConfidence ?? 0.0;
            var hops = result.Result?.PlatonicHopCount ?? 0;
            var fallback = result.Result?.UsedNeuralFallback ?? false;
            var chunks = result.Result?.ChunksGenerated ?? 0;
            var predictOutput = GetControl<Label>("PredictOutput");
            if (predictOutput != null) predictOutput.Text = $"→ {output} [{route} c={confidence:F2} hops={hops} fallback={fallback} chunks={chunks}]";
        }
        catch (Exception ex)
        {
            var predictOutput = GetControl<Label>("PredictOutput");
            if (predictOutput != null) predictOutput.Text = $"Error: {ex.Message}";
        }
        finally
        {
            if (predictBtn != null) predictBtn.Enabled = true;
        }
    }

    private void UpdateStats(int exampleCount, double loss)
    {
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => UpdateStats(exampleCount, loss)));
            return;
        }

        var statsText = GetControl<Label>("StatsText");
        if (statsText != null)
        {
            statsText.Text = BuildStatsText(exampleCount, loss);
        }
    }

    private void UpdateAutonomousStats(int roundCount, double loss, string creator)
    {
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => UpdateAutonomousStats(roundCount, loss, creator)));
            return;
        }

        var statsText = GetControl<Label>("AutoStatsText");
        if (statsText != null)
        {
            statsText.Text = $"Rounds: {roundCount} | Last loss: {loss:F4} | Creator: {creator}";
        }
    }

    private string BuildStatsText(int? exampleCount, double? loss)
    {
        var vocab = Math.Max(1, _runtime.VocabularySize);
        var hidden = Math.Max(1, _runtime.HiddenSize);
        var parameterCount = (2L * vocab * hidden) + vocab; // embeddings + output + bias
        var modelSizeMiB = (parameterCount * sizeof(float)) / (1024.0 * 1024.0);

        var examplesPart = exampleCount.HasValue ? exampleCount.Value.ToString() : "n/a";
        var lossPart = loss.HasValue ? loss.Value.ToString("F4") : "n/a";

        return
            $"Examples: {examplesPart} | Loss: {lossPart}\n" +
            $"Vocab: {vocab:N0} | Hidden: {hidden:N0} | Params≈{parameterCount:N0} | FP32≈{modelSizeMiB:F2} MiB\n" +
            $"Policy: CPU train | GPU/VRAM infer";
    }

    private void AppendOutput(string message)
    {
        AppendToLogBox("OutputBox", message);
    }

    private void AppendAutonomousOutput(string message)
    {
        AppendToLogBox("AutoOutputBox", message);
    }

    private void AppendToLogBox(string boxName, string message)
    {
        if (InvokeRequired)
        {
            Invoke(() => AppendToLogBox(boxName, message));
            return;
        }

        var outputBox = GetControl<RichTextBox>(boxName);
        if (outputBox != null && !outputBox.IsDisposed)
        {
            var shouldAutoScroll = IsNearBottom(outputBox);
            outputBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}\n");
            TrimRichTextHistory(outputBox, MaxTrainingLogChars, TargetTrainingLogChars);
            if (shouldAutoScroll)
            {
                outputBox.SelectionStart = outputBox.TextLength;
                outputBox.SelectionLength = 0;
                outputBox.ScrollToCaret();
            }
        }
    }

    private static void TrimRichTextHistory(RichTextBox box, int maxChars, int targetChars)
    {
        if (box.IsDisposed || !box.IsHandleCreated)
            return;

        if (box.TextLength <= maxChars)
            return;

        var removeChars = box.TextLength - targetChars;
        var text = box.Text;
        var newline = text.IndexOf('\n', Math.Max(0, removeChars));
        var trimAt = newline >= 0 ? newline + 1 : removeChars;
        if (trimAt <= 0)
            return;

        trimAt = Math.Min(trimAt, box.TextLength);
        var previousSelectionStart = box.SelectionStart;
        var previousSelectionLength = box.SelectionLength;

        box.Select(0, trimAt);
        box.SelectedText = string.Empty;

        var restoredStart = Math.Max(0, previousSelectionStart - trimAt);
        var maxLength = Math.Max(0, box.TextLength - restoredStart);
        var restoredLength = Math.Min(previousSelectionLength, maxLength);
        box.Select(restoredStart, restoredLength);
    }

    private static bool IsNearBottom(RichTextBox box)
    {
        if (box.TextLength == 0)
            return true;

        var visibleBottomIndex = box.GetCharIndexFromPosition(new Point(4, Math.Max(4, box.ClientSize.Height - 4)));
        return box.TextLength - visibleBottomIndex <= AutoScrollSlackChars;
    }

    private void AppendExceptionReport(string context, Exception ex, string boxName = "OutputBox")
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

    private void ApplyL2Preset(int presetIndex)
    {
        double coefficient = presetIndex switch
        {
            0 => 0.0,    // Off
            1 => 1e-5,   // Mild
            2 => 1e-4,   // Balanced
            3 => 1e-3,   // Aggressive
            4 => 1e-2,   // Extreme
            _ => 0.0
        };
        
        _runtime.UpdateConfig(c => c with { L2RegularizationCoefficient = coefficient });
        
        var description = presetIndex switch
        {
            0 => "Off: No L2 compression. Best for: maximum learning speed/capacity.",
            1 => "Mild: Neural layer comfortable, can grow if needed. Best for: early training, testing model capacity.",
            2 => "Balanced: Medium weight penalty.",
            3 => "Aggressive ⚡: Heavy penalty forces symbolic learning. Best for: maximizing platonic space, lean models.",
            4 => "Extreme: Nearly all learning symbolic. Warning: may hurt convergence; best for: debugging, understanding discovery.",
            _ => "Unknown preset"
        };
        
        AppendOutput($"✓ L2 Regularization: {description}");
    }
}
