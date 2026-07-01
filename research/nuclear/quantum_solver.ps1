<#
  quantum_solver.ps1  -  radial Schrodinger eigensolver for the nuclear mean field.

  Goes below the N^p toy of nuclear_predictor.ps1 to the actual quantum level: solve
     -hbar^2/2m u'' + [ V(r) + hbar^2 l(l+1)/2m r^2 ] u = E u
  by finite differences, with eigenvalues found via the Sturm sequence of the resulting
  symmetric tridiagonal matrix (counts eigenvalues below lambda; bisection isolates each).

  This file (STAGE 1) validates the solver against the EXACT 3-D harmonic oscillator,
  where E_{n_r,l} = hbar*omega (2 n_r + l + 3/2).  Woods-Saxon + spin-orbit + Coulomb are
  added only once this passes.  See README.md for status.
#>
param(
  [double]$H = 0.05,     # radial grid step (fm)
  [int]$M = 600,         # grid points  -> r_max = M*H fm
  [double]$C = 0.5       # HO potential V(r)=C*r^2 for the validation
)

$hbar2_2m = 20.7355      # MeV*fm^2  (hbar^2 / 2 m_nucleon)
$rmax = $M * $H

# effective potential array for a given l (validation potential = C r^2 + centrifugal)
function VeffHO([int]$l) {
  $v = New-Object 'double[]' ($M+1)
  for ($i=1; $i -le $M; $i++) {
    $r = $i*$H
    $v[$i] = $C*$r*$r + $hbar2_2m*$l*($l+1)/($r*$r)
  }
  return ,$v
}

# tridiagonal: diag d[i] = 2*hbar2_2m/h^2 + Veff[i] ; off-diagonal e = -hbar2_2m/h^2 (constant)
$offd = -$hbar2_2m/($H*$H)
$offd2 = $offd*$offd

# Sturm sequence: number of eigenvalues strictly below lambda
function CountBelow($diag, [double]$lambda) {
  $q = $diag[1] - $lambda
  $count = 0
  if ($q -lt 0) { $count++ }
  for ($i=2; $i -le $M; $i++) {
    if ([math]::Abs($q) -lt 1e-300) { $q = 1e-300 }
    $q = ($diag[$i] - $lambda) - $offd2/$q
    if ($q -lt 0) { $count++ }
  }
  return $count
}

# n-th eigenvalue (n=0 lowest) by bisection on the Sturm count
function NthEigen($diag, [int]$n, [double]$lo, [double]$hi) {
  for ($it=0; $it -lt 200; $it++) {
    $mid = 0.5*($lo+$hi)
    if ((CountBelow $diag $mid) -ge ($n+1)) { $hi = $mid } else { $lo = $mid }
    if (($hi-$lo) -lt 1e-6) { break }
  }
  return 0.5*($lo+$hi)
}

function DiagFor([int]$l) {
  $veff = VeffHO $l
  $d = New-Object 'double[]' ($M+1)
  $base = 2*$hbar2_2m/($H*$H)
  for ($i=1; $i -le $M; $i++) { $d[$i] = $base + $veff[$i] }
  return ,$d
}

# hbar*omega for V=C r^2 = 1/2 k r^2 -> k=2C ; hbar*omega = sqrt(4*C*hbar2_2m)
$hw = [math]::Sqrt(4*$C*$hbar2_2m)
Write-Host "=================================================================="
Write-Host " RADIAL SCHRODINGER SOLVER  -  validation vs 3-D harmonic oscillator"
Write-Host " grid: h=$H fm, M=$M (r_max=$rmax fm);  hbar*omega(exact)=$([math]::Round($hw,4)) MeV"
Write-Host "=================================================================="
Write-Host ""
Write-Host ("{0,-4} {1,-4} {2,-12} {3,-12} {4}" -f "l","n_r","E_numeric","E_exact","err%")
$maxerr = 0.0
foreach ($l in 0,1,2,3,4) {
  $d = DiagFor $l
  for ($nr=0; $nr -le 2; $nr++) {
    $E = NthEigen $d $nr 0.0 400.0
    $Eexact = $hw*(2*$nr + $l + 1.5)
    $err = [math]::Abs($E-$Eexact)/$Eexact*100
    if ($err -gt $maxerr) { $maxerr = $err }
    Write-Host ("{0,-4} {1,-4} {2,-12} {3,-12} {4}" -f $l,$nr,[math]::Round($E,4),[math]::Round($Eexact,4),[math]::Round($err,3))
  }
}
Write-Host ""
Write-Host ("Max error vs exact HO: {0}%   {1}" -f [math]::Round($maxerr,3), $(if($maxerr -lt 1.0){"PASS (solver validated)"}else{"FAIL - fix grid/solver before trusting Woods-Saxon"}))
