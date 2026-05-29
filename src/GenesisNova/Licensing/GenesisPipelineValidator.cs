using EvalApp.Consumer;
using System.Security.Cryptography;
using System.Text;

namespace GenesisNova.Licensing;

/// <summary>
/// Genesis Nova pipeline license validator. Enables adaptive tuning for smart CPU/GPU gating.
/// </summary>
public static class GenesisPipelineValidator
{
    /// <summary>Genesis Nova license key (expires 2027-05-29)</summary>
    private const string LicenseKey = "20270529-rwNNXcgcp1XSdkhdzGUKmEm5BGCxKt1ocE8BNsSPfOI";

    /// <summary>
    /// Validates the Genesis Nova license and returns the mode.
    /// Licensed mode enables adaptive tuning for smart CPU/GPU parallelism.
    /// </summary>
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

    /// <summary>
    /// Validates license periodically (default 60s between checks) for hot paths.
    /// </summary>
    public static LicenseMode ValidateLicensePeriodic(TimeSpan? interval = null)
    {
        // Cache validation result for hot paths
        if (_cachedMode != null && DateTime.UtcNow - _lastCheck < (interval ?? TimeSpan.FromSeconds(60)))
        {
            return _cachedMode.Value;
        }

        _cachedMode = ValidateLicense();
        _lastCheck = DateTime.UtcNow;
        return _cachedMode.Value;
    }

    private static LicenseMode? _cachedMode;
    private static DateTime _lastCheck = DateTime.MinValue;

    /// <summary>
    /// Validates license key format and expiration.
    /// Format: YYYYMMDD-signature
    /// </summary>
    private static bool ValidateLicenseKey(string licenseKey)
    {
        if (string.IsNullOrEmpty(licenseKey))
            return false;

        var parts = licenseKey.Split('-');
        if (parts.Length != 2)
            return false;

        if (!int.TryParse(parts[0], out var dateValue))
            return false;

        // Parse YYYYMMDD format
        var year = dateValue / 10000;
        var month = (dateValue / 100) % 100;
        var day = dateValue % 100;

        try
        {
            var expirationDate = new DateTime(year, month, day);
            if (DateTime.UtcNow > expirationDate)
                return false; // License expired

            // Verify signature using HMAC-SHA256
            const string productId = "GENESIS";
            var seed = Convert.FromBase64String("h8Jk2LpQ7vN+3xR9mKoD4wT6yZaB1cF5gH0nJsL8vP2=");
            
            using (var hmac = new HMACSHA256(seed))
            {
                var dataToSign = $"{productId}:{parts[0]}";
                var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(dataToSign));
                var expectedSignature = Convert.ToBase64String(hash);
                
                return parts[1] == expectedSignature.TrimEnd('=');
            }
        }
        catch
        {
            return false;
        }
    }
}
