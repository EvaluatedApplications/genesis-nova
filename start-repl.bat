@echo off
REM Genesis Nova - Interactive Training UI Quick Start (Windows)

echo.
echo === Genesis Nova Interactive REPL ===
echo.
echo Starting REPL with idle introspection support...
echo.

cd /d "%~dp0src\bin\Release\net8.0"

REM Generate training examples if not present
if not exist "examples-50.jsonl" (
    echo Generating training examples...
    GenesisNova.exe --genesis-gen-examples --count 50 --difficulty 0 --output examples-50.jsonl
    echo.
)

echo REPL Commands:
echo   introspect-idle      - Start background introspection (non-blocking)
echo   trainfile FILE N     - Train for N epochs with live feedback
echo   predict TEXT         - Generate output
echo   stats                - Show model stats
echo   verbose              - Toggle verbose logging
echo   help                 - Show all commands
echo.
echo Example workflow:
echo   1. introspect-idle   (start background processing)
echo   2. trainfile examples-50.jsonl 3  (train with feedback)
echo   3. predict hello     (query during training!)
echo   4. stats             (check queue depth)
echo   5. introspect-stop   (stop background)
echo.

GenesisNova.exe --genesis-repl
