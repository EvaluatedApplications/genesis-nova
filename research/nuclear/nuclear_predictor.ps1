<#
  nuclear_predictor.ps1  -  Genesis folding model of the nucleus, as a predictor.

  WHAT THIS IS
    A first-principles STRUCTURAL predictor of nuclear shell closures (magic numbers)
    and the valley of stability, derived from one rule -- "fold a genesis triangle to
    resolve a contradiction" -- plus the geometry of a sphere growing from a kernel.

  HONEST STATUS (read before trusting an output)
    DERIVED from geometry (not fitted, not hardcoded):
      - shell structure: shell N holds orbitals l = N, N-2, ... ; capacity 2j+1  (tetrahedral / 3-D)
      - the 2*T_N oscillator skeleton (tetrahedral packing = harmonic-oscillator degeneracy)
      - spin-orbit FORM and SIGN:   -(l.s), co-rotating folds bind tighter
      - spin-orbit SHAPE 1/sqrt(N): surface/volume = 1/r, harmonic radius r ~ sqrt(N)  (the sphere)
      - realistic-well DIRECTION:   E = N^p with p>1  (sphere's expanding shells -> Woods-Saxon regime)
      - curl-sharpness FORM:        -(l^2 - <l^2>_N), sharper folds bind more
      - Coulomb / symmetry FORMS:   Z^2 / A^(1/3)  and  (N-Z)^2 / A   (charges on the sphere; balanced folds)

    IMPORTED (empirical force constants -- the ONLY free inputs; each is a physical SCALE,
    never an answer. The magic numbers are OUTPUTS, never entered anywhere):
      - a  : spin-orbit strength          (strong-force spin dependence)
      - p  : exact potential exponent     (direction p>1 derived; value empirical)
      - b  : l^2 curl-sharpness strength   (LOAD-BEARING: b=0 collapses the model)
      - k  : Coulomb / symmetry ratio     (contains the fine-structure constant; never geometric)

    So: geometry fixes the ENTIRE PATTERN; nature's force constants set the scales. This is the
    same footing as the Mayer-Jensen shell model -- with the four terms being exactly the
    physically-necessary ones -- plus the folding model additionally GENERATES the 3-D arena
    the shell model assumes. It is NOT a zero-input oracle, and it does NOT hardcode the answers.

  USAGE
    .\nuclear_predictor.ps1                 # predict + self-verify at default (physical) scales
    .\nuclear_predictor.ps1 -Robustness     # also run the parameter-band robustness sweep
    .\nuclear_predictor.ps1 -A 0.4 -P 1.2 -B 0.03 -K 0.0154 -Nmax 8
#>
param(
  [double]$A = 0.40,      # spin-orbit strength
  [double]$P = 1.20,      # potential exponent (p>1 = Woods-Saxon regime)
  [double]$B = 0.03,      # l^2 curl-sharpness strength (load-bearing)
  [double]$K = 0.0154,    # Coulomb/symmetry ratio
  [int]$Nmax = 8,         # highest fold-shell to build (N=8 reaches the 184 island)
  [int]$TopDrops = 10,    # how many strongest S2n drops to report as magic candidates
  [switch]$Robustness     # run the robustness sweep over the fitted scales
)

# ---------------------------------------------------------------------------
# DERIVED: build the fold-shell orbitals from pure geometry. No magic numbers here.
#   shell N -> orbitals l = N, N-2, ... ; each l splits into j=l+1/2 (co-rotating, cap 2l+2)
#   and j=l-1/2 (opposed, cap 2l). ls = <l.s> = +l/2 (aligned) or -(l+1)/2 (opposed).
# ---------------------------------------------------------------------------
function Build-Orbitals([int]$nmax) {
  $orb = @()
  for ($N = 0; $N -le $nmax; $N++) {
    for ($l = $N; $l -ge 0; $l -= 2) {
      $orb += [pscustomobject]@{ N=$N; l=$l; cap=(2*$l+2); ls=($l/2.0) }      # j = l + 1/2
      if ($l -ge 1) {
        $orb += [pscustomobject]@{ N=$N; l=$l; cap=(2*$l);   ls=(-($l+1)/2.0) } # j = l - 1/2
      }
    }
  }
  return ,$orb
}

# shell-centred <l(l+1)>_N (Nilsson convention: the l^2 term only reorders WITHIN a shell)
function Build-ShellAvg($orb, [int]$nmax) {
  $avg = @{}
  for ($N = 0; $N -le $nmax; $N++) {
    $sub = $orb | Where-Object { $_.N -eq $N }
    $w = 0.0; $s = 0.0
    foreach ($o in $sub) { $w += $o.cap; $s += $o.cap * $o.l * ($o.l + 1) }
    $avg[$N] = $s / $w
  }
  return $avg
}

# ---------------------------------------------------------------------------
# The single-particle energy.  Every TERM is derived; only a, p, b are scales.
#   E(N,l,j) = N^p  -  (a/sqrt(N)) * (l.s)  -  b * ( l(l+1) - <l(l+1)>_N )
# ---------------------------------------------------------------------------
function Get-SortedLevels($orb, $avg, [double]$a, [double]$p, [double]$b) {
  $orb | ForEach-Object {
    $n = [math]::Max($_.N, 1)
    [pscustomobject]@{
      E   = ([math]::Pow($_.N, $p) - ($a/[math]::Sqrt($n))*$_.ls - $b*($_.l*($_.l+1) - $avg[$_.N]))
      cap = $_.cap
    }
  } | Sort-Object E
}

# ---------------------------------------------------------------------------
# Magic numbers as OUTPUTS: fill nucleons into the levels, and magic numbers are the
# biggest drops in the two-nucleon separation energy S2 (the physical DEFINITION).
# ---------------------------------------------------------------------------
function Get-MagicNumbers($orb, $avg, [double]$a, [double]$p, [double]$b, [int]$top) {
  $lv = Get-SortedLevels $orb $avg $a $p $b
  $eps = @(0.0)  # eps[n] = energy of the n-th nucleon (1-indexed)
  foreach ($o in $lv) { for ($c = 0; $c -lt $o.cap; $c++) { $eps += $o.E } }
  $drops = @()
  for ($N = 2; $N -le ($eps.Count - 3); $N += 2) {
    $s2_here = -1*($eps[$N]   + $eps[$N-1])
    $s2_next = -1*($eps[$N+2] + $eps[$N+1])
    $drops += [pscustomobject]@{ N=$N; drop=($s2_here - $s2_next) }
  }
  ($drops | Sort-Object drop -Descending | Select-Object -First $top | Sort-Object N | Select-Object -ExpandProperty N)
}

# ---------------------------------------------------------------------------
# Valley of stability: minimise  symmetry (N-Z)^2/A  +  Coulomb Z^2/A^(1/3)  at fixed A.
#   Z_valley(A) = A / (2 + k * A^(2/3))         (both forms derived; k empirical)
# ---------------------------------------------------------------------------
function Get-ValleyZ([int]$Amass, [double]$k) {
  [math]::Round($Amass / (2 + $k * [math]::Pow($Amass, 2.0/3.0)))
}

# ===========================================================================
# RUN
# ===========================================================================
$orb = Build-Orbitals $Nmax
$avg = Build-ShellAvg $orb $Nmax

Write-Host "=================================================================="
Write-Host " GENESIS FOLDING NUCLEAR PREDICTOR"
Write-Host " scales (empirical):  a=$A  p=$P  b=$B  k=$K   | shells N=0..$Nmax"
Write-Host "=================================================================="

$predicted = Get-MagicNumbers $orb $avg $A $P $B $TopDrops
Write-Host ""
Write-Host "PREDICTED shell closures (strongest S2 drops, computed -- not entered):"
Write-Host ("  " + ($predicted -join ", "))

# --- self-verification against reference data (reference is ONLY used here, to score) ---
$knownMagic = 2,8,20,28,50,82,126,184     # 184 = predicted island; the rest are measured
$hit = ($knownMagic | Where-Object { $_ -in $predicted }).Count
Write-Host ""
Write-Host "SELF-CHECK vs known/predicted magic numbers (2,8,20,28,50,82,126,184):"
Write-Host ("  reproduced " + $hit + " / 8   (extras are sub-shell closures, e.g. 6, 40)")

# --- valley of stability + doubly-magic anchors ---
Write-Host ""
Write-Host "VALLEY OF STABILITY  Z_valley(A) = A / (2 + $K*A^(2/3)):"
Write-Host "  A     Z_pred  N_pred  N-Z"
foreach ($Amass in 4,16,40,56,120,208,238) {
  $Z = Get-ValleyZ $Amass $K; $N = $Amass - $Z
  Write-Host ("  {0,-5} {1,-7} {2,-7} {3}" -f $Amass, $Z, $N, ($N-$Z))
}
Write-Host "  (Pb-208 -> Z=82, N=126, both magic = heaviest doubly-magic; N-Z drift reproduced.)"

# --- optional robustness sweep: do predictions survive perturbing the fitted scales? ---
if ($Robustness) {
  Write-Host ""
  Write-Host "ROBUSTNESS SWEEP (are the magic numbers structural, or a fitted point?)"
  $cells=0; $c8=0; $c7=0; $h126=0; $h184=0
  foreach ($a in 0.30,0.35,0.40,0.45,0.50) {
    foreach ($p in 1.10,1.15,1.20,1.25,1.30) {
      foreach ($b in 0.02,0.03,0.04) {
        $cells++
        $m = Get-MagicNumbers $orb $avg $a $p $b $TopDrops
        $h = ($knownMagic | Where-Object { $_ -in $m }).Count
        if ($h -eq 8) { $c8++ }; if ($h -ge 7) { $c7++ }
        if (126 -in $m) { $h126++ }; if (184 -in $m) { $h184++ }
      }
    }
  }
  Write-Host ("  over $cells cells (a:.30-.50, p:1.10-1.30, b:.02-.04):")
  Write-Host ("    all 8 magic:  $c8/$cells      >=7 of 8:  $c7/$cells")
  Write-Host ("    126 survives: $h126/$cells    184 survives: $h184/$cells")
  Write-Host "  (High survival = predictions ride the derived structure, not the fitted scales.)"
}

Write-Host ""
Write-Host "Note: magic numbers above are COMPUTED from the geometry + 4 physical scales."
Write-Host "They are never hardcoded. See 11-NUCLEAR-GENESIS.md for the derivation + falsifications."
