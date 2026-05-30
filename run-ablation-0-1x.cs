#!/usr/bin/env dotnet script
// Simple script to run ablation and write results

using System;
using System.Diagnostics;
using System.IO;

var testProj = @"C:\Users\cex\repos-working\genesis-nova\Tests\GenesisNova.Tests.csproj";
var testDll = @"C:\Users\cex\repos-working\genesis-nova\Tests\bin\Debug\net8.0-windows\GenesisNova.Tests.dll";

var psi = new ProcessStartInfo("dotnet", "test --no-build --filter \"WhenTrainingWithVsWithoutPlatonicPhases\"")
{
    WorkingDirectory = @"C:\Users\cex\repos-working\genesis-nova",
    UseShellExecute = false,
    RedirectStandardOutput = true,
    RedirectStandardError = true,
};

using (var proc = Process.Start(psi))
{
    var output = proc.StandardOutput.ReadToEnd();
    var error = proc.StandardError.ReadToEnd();
    proc.WaitForExit();
    
    var resultsFile = Path.Combine(Path.GetTempPath(), "ablation-0-1x-results.txt");
    File.WriteAllText(resultsFile, $"=== ABLATION TEST (0.1x coefficients) ===\n{output}\n\n=== ERRORS ===\n{error}");
    Console.WriteLine($"Results written to: {resultsFile}");
    Console.WriteLine($"Exit code: {proc.ExitCode}");
}
