# Demo: Genesis Nova REPL with Interactive Training

$exe = "C:\Users\cex\repos-working\genesis-nova\src\bin\Release\net8.0\GenesisNova.exe"
$examplesFile = "$($exe | Split-Path)\examples-50.jsonl"

# Generate examples if they don't exist
if (-not (Test-Path $examplesFile)) {
    Write-Host "Generating training examples..." -ForegroundColor Cyan
    & $exe --genesis-gen-examples --count 50 --difficulty 0 --output $examplesFile
}

Write-Host "`n=== GENESIS NOVA INTERACTIVE REPL ===" -ForegroundColor Green
Write-Host "This demo shows:" -ForegroundColor Cyan
Write-Host "1. Starting idle introspection in background" 
Write-Host "2. Kicking off training with live feedback"
Write-Host "3. Running queries while training happens"
Write-Host ""

# Simulate interactive commands
$commands = @(
    "stats",
    "introspect-idle",
    "verbose"
    "trainfile $examplesFile 1",
    "stats",
    "predict hello",
    "introspect-stop",
    "exit"
)

Write-Host "Ready to run REPL. To use interactively:" -ForegroundColor Yellow
Write-Host "  cd $($exe | Split-Path)"
Write-Host "  .\GenesisNova.exe --genesis-repl"
Write-Host ""
Write-Host "Key commands:" -ForegroundColor Yellow
Write-Host "  introspect-idle      - Start background introspection cycles" -ForegroundColor Cyan
Write-Host "  trainfile <path> N   - Train for N epochs with progress feedback" -ForegroundColor Cyan
Write-Host "  predict <text>       - Get model output (works during training!)" -ForegroundColor Cyan
Write-Host "  verbose              - Toggle verbose idle introspection logging" -ForegroundColor Cyan
Write-Host "  help                 - Show all commands" -ForegroundColor Cyan
