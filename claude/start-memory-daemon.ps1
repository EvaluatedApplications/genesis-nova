# Auto-start the ClaudeMemory background daemon (idempotent) + emit a SessionStart bootstrap.
# Wired to the Claude Code SessionStart hook (.claude/settings.json). Safe to run by hand.
#
# Contract with Claude Code SessionStart:
#   - stdout MUST be a single JSON object. `systemMessage` is shown to the USER; `hookSpecificOutput.
#     additionalContext` is injected into the MODEL's context (the actual bootstrap). Diagnostics go to stderr.
#   - Idempotent: if a serve daemon already runs, it is not duplicated.
#   - Never fails the session: errors degrade to a JSON note and exit 0.

$ErrorActionPreference = 'SilentlyContinue'
$repo = Split-Path -Parent $PSScriptRoot
$exe  = Join-Path $repo 'claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe'

$boot = 'Genesis-Nova memory is available: a continuous-mastery Genesis-Nova INDEX over MEMORY.md (the file ' +
        'memory is the source of truth). CLAUDE.md has the substrate overview + golden training paths + memory rules. ' +
        'Recall relevant memory keys (GRU-routed) with: ' +
        '.\claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe recall "<topic>" 2>$null  ' +
        '(-> open memory/<key>.md). Observe the daemon with `ClaudeMemory.exe watch` (or `metrics`). ' +
        'Save durable facts/gotchas proactively.'

function Emit([string]$systemMessage, [string]$context) {
    $payload = [ordered]@{
        systemMessage      = $systemMessage
        hookSpecificOutput = [ordered]@{ hookEventName = 'SessionStart'; additionalContext = $context }
    }
    $payload | ConvertTo-Json -Compress -Depth 6
    exit 0
}

if (-not (Test-Path $exe)) {
    Emit '[claude-memory] memory tool not built — run: dotnet build claude/ClaudeMemory -c Release' `
         ("ClaudeMemory is not built yet (no daemon running). " + $boot)
}

$running = Get-CimInstance Win32_Process -Filter "Name='ClaudeMemory.exe'" |
    Where-Object { $_.CommandLine -match 'serve' }

if ($running) {
    [Console]::Error.WriteLine("[claude-memory] daemon already running (PID $([string]::Join(',', $running.ProcessId)))")
    Emit '[claude-memory] memory daemon already running' ('Genesis-Nova memory daemon is already running. ' + $boot)
}

$index = $env:CLAUDE_MEMORY_FILE
if (-not $index -or -not (Test-Path $index)) {
    $known = 'C:\Users\dongy\.claude\projects\C--Users-dongy\memory\MEMORY.md'
    if (Test-Path $known) { $index = $known } else { $index = Join-Path $repo 'claude\truth\memory.truth' }
}

# Launch the daemon in a VISIBLE window (so it can be observed for the whole session), with native libtorch
# stderr (benign non-leaf .grad warnings) redirected to a log — so the window shows ONLY the daemon's own
# stdout: banner, license/GPU lines, and the continuous-mastery learning curve. A generated .cmd does the
# OS-level `2>` redirect that Start-Process alone can't do while still showing a window.
$errLog  = Join-Path $repo '.claude-nova\daemon.stderr.log'
$null    = New-Item -ItemType Directory -Force (Split-Path $errLog) -ErrorAction SilentlyContinue
$cmdFile = Join-Path $env:TEMP 'claude-memory-daemon.cmd'
Set-Content -Path $cmdFile -Encoding ASCII -Value "@echo off`r`ntitle ClaudeMemory daemon`r`n`"$exe`" serve --gpu --index `"$index`" 2>`"$errLog`""
Start-Process -FilePath $cmdFile -WindowStyle Normal
[Console]::Error.WriteLine("[claude-memory] daemon started (index: $index)")
Emit '[claude-memory] memory daemon started' ("Genesis-Nova memory daemon started (index $index). " + $boot)
