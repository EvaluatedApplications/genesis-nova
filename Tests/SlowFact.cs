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

    /// <summary>Skip reason for the bare-subject fact-memory experiments. After de-hardcoding the grammar parser (the
    /// engine has no copula/possessive word-list — roles are LEARNED), tagging a BARE or "the"-determiner subject
    /// ("alice is doctor", "the password is plum") as SUBJECT needs a GRU that has been TRAINED — the role head reads
    /// the GRU's features, and on a near-random GRU it only generalises to POSSESSIVE subjects ("my name is X", which
    /// still passes — FactRecallExperiment). The capability is real WITH training: DurableMechanismTests asserts bare
    /// "qzx is red" and passes because it trains the full gym first. These lightweight probes do no GRU training, so
    /// they need a full-gym warm-up to be re-enabled (the head-only GrammarWarmup is not enough).</summary>
    public const string BareSubjectWarmup =
        "Pending full-gym warm-up: the de-hardcoded NN role parser tags bare/'the' subjects only with a TRAINED GRU "
        + "(head-only GrammarWarmup covers possessive subjects — see FactRecallExperiment; DurableMechanismTests passes "
        + "bare subjects via full gym training).";

    public static bool Enabled
    {
        get
        {
            var v = Environment.GetEnvironmentVariable("RUN_SLOW");
            return v is "1" or "true" or "TRUE" or "yes";
        }
    }
}
