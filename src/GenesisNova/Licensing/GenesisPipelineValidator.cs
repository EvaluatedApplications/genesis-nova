using EvalApp.Consumer;
using System.Security.Cryptography;
using System.Text;

namespace GenesisNova.Licensing;

/// <summary>
/// Genesis Nova pipeline license validator. EvalApp's adaptive CPU/GPU tuning (the non-"sequential" mode) is
/// licensed by passing <see cref="ActiveKey"/> into the pipeline's <c>.Build(licenseKey)</c>. This validator
/// MIRRORS EvalApp's real gate (product "EVALAPP" — see EvalApp.Licensing.LicenseGateFactory) so the mode it
/// reports matches exactly what <c>.Build</c> will accept: a valid key → Licensed → adaptive tuning; an
/// invalid/expired key → Unlicensed → free sequential mode (and ActiveKey is null, so .Build never throws).
/// </summary>
public static class GenesisPipelineValidator
{
    /// <summary>EvalApp license key (product EVALAPP, expires 2027-03-12). Shared with genesis-engine.</summary>
    private const string LicenseKey = "20270312-gwZ8hyAovecW9DmRm_OQ13xOG7oCZWvyBkYHzy_ZS8k";

    // EVALAPP product parameters: a key "<yyyyMMdd>-<sig>" is valid when <sig> equals the base64url HMAC-SHA256
    // of "EVALAPP|<yyyyMMdd>" under this seed and the date has not passed. Kept in sync with EvalApp's gate.
    private const string ProductId = "EVALAPP";
    private static readonly byte[] Seed = Convert.FromBase64String("RKn4nR+D+B6w0jtlIVwOtCEQByUr+Pg85/fwElg45oA=");

    /// <summary>
    /// The key to hand to the pipeline's <c>.Build(licenseKey)</c>, or null when invalid/expired so the
    /// pipeline degrades to free sequential mode instead of throwing <c>InvalidLicenseException</c>.
    /// </summary>
    public static string? ActiveKey => ValidateLicense() == LicenseMode.Licensed ? LicenseKey : null;

    /// <summary>Validates the Genesis Nova (EvalApp) license and returns the mode.</summary>
    public static LicenseMode ValidateLicense()
    {
        try
        {
            return ValidateLicenseKey(LicenseKey) ? LicenseMode.Licensed : LicenseMode.Unlicensed;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"License validation failed: {ex.Message}");
            return LicenseMode.Unlicensed;
        }
    }

    /// <summary>Validates periodically (default 60s between checks) for hot paths.</summary>
    public static LicenseMode ValidateLicensePeriodic(TimeSpan? interval = null)
    {
        if (_cachedMode != null && DateTime.UtcNow - _lastCheck < (interval ?? TimeSpan.FromSeconds(60)))
            return _cachedMode.Value;

        _cachedMode = ValidateLicense();
        _lastCheck = DateTime.UtcNow;
        return _cachedMode.Value;
    }

    private static LicenseMode? _cachedMode;
    private static DateTime _lastCheck = DateTime.MinValue;

    /// <summary>
    /// Validates an EVALAPP license key. Format: <c>YYYYMMDD-base64urlSignature</c>. Split on the FIRST '-'
    /// (the base64url signature may itself contain '-'). Verifies expiry then the HMAC signature.
    /// </summary>
    private static bool ValidateLicenseKey(string licenseKey)
    {
        if (string.IsNullOrWhiteSpace(licenseKey))
            return false;

        var dash = licenseKey.IndexOf('-');
        if (dash <= 0 || dash >= licenseKey.Length - 1)
            return false;

        var datePart = licenseKey[..dash];
        var sigPart = licenseKey[(dash + 1)..];
        if (datePart.Length != 8 || !int.TryParse(datePart, out var dateValue))
            return false;

        try
        {
            var expiration = new DateTime(dateValue / 10000, (dateValue / 100) % 100, dateValue % 100, 0, 0, 0, DateTimeKind.Utc);
            if (DateTime.UtcNow > expiration)
                return false; // expired

            using var hmac = new HMACSHA256(Seed);
            var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes($"{ProductId}|{datePart}"));
            var expected = Convert.ToBase64String(hash).TrimEnd('=').Replace('+', '-').Replace('/', '_');
            return sigPart == expected;
        }
        catch
        {
            return false;
        }
    }
}
