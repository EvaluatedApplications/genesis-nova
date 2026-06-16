using System;
using Xunit;

namespace GenesisNova.Tests;

/// <summary>
/// Marks a test that trains the neural model (slow, GPU-bound). These are SKIPPED by default so the default
/// <c>dotnet test</c> run stays fast on the behaviour suite. They run only when the environment variable
/// <c>RUN_SLOW</c> is set.
///
/// Run the heavy suite:
///   PowerShell:  <c>$env:RUN_SLOW=1; dotnet test</c>
///   bash:        <c>RUN_SLOW=1 dotnet test</c>
/// Run only one heavy test:  <c>$env:RUN_SLOW=1; dotnet test --filter FullyQualifiedName~SeqComposerTests</c>
/// </summary>
public sealed class SlowFactAttribute : FactAttribute
{
    public SlowFactAttribute()
    {
        if (!SlowTests.Enabled)
            Skip = SlowTests.SkipReason;
    }
}

/// <summary>Theory variant of <see cref="SlowFactAttribute"/> — skipped unless <c>RUN_SLOW</c> is set.</summary>
public sealed class SlowTheoryAttribute : TheoryAttribute
{
    public SlowTheoryAttribute()
    {
        if (!SlowTests.Enabled)
            Skip = SlowTests.SkipReason;
    }
}

internal static class SlowTests
{
    public const string SkipReason = "Long-running (trains the neural model). Set RUN_SLOW=1 to run.";

    public static bool Enabled
    {
        get
        {
            var v = Environment.GetEnvironmentVariable("RUN_SLOW");
            return v is "1" or "true" or "TRUE" or "yes";
        }
    }
}
