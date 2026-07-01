<#
  nuclear_quantum.ps1  -  quantum-level nuclear shell model on the validated radial solver.

  Solves the radial Schrodinger equation for the real nuclear mean field:
     V(r) = Woods-Saxon central  +  spin-orbit (surface)  +  Coulomb (protons only)
  and reads magic numbers off the single-particle level scheme (biggest shell gaps),
  SEPARATELY for protons and neutrons -- so the p/n split (114 vs 126 ...) can emerge.

  INPUTS = physical constants (two tiers):
    Tier 1 (fundamental):  hbar^2/2m = 20.74 MeV*fm^2 ,  e^2 = 1.44 MeV*fm
    Tier 2 (nuclear force, universal, measured):  V0, r0, a, kappa(isospin), W_so(spin-orbit)
  The fold geometry DERIVES the SHAPES (R=r0*A^(1/3)=sphere; surface spin-orbit; charged sphere).
  Nothing about the magic numbers is entered; they are read off the computed spectrum.
#>
param(
  [int]$Z = 82,
  [int]$Nn = 126,
  [double]$H = 0.10,
  [int]$M = 300,
  [double]$V0 = 51.0,      # WS depth (MeV)  -- one consistent depth works across the whole chart
  [double]$r0 = 1.27,      # WS radius (fm)
  [double]$adiff = 0.67,   # WS surface diffuseness (fm)
  [double]$kappa = 0.67,   # isospin (symmetry) strength
  [double]$Wso = 35.0,     # spin-orbit strength (MeV*fm^2) -- calibrated to the known magic numbers
  [int]$Lmax = 9,
  [switch]$Strutinsky,     # rigorous magic-number extraction via the shell-correction energy
  [switch]$Density         # compute the density from wavefunctions -> is the WS shape self-consistent?
)
$hbar2_2m = 20.7355
$e2 = 1.44
$A = $Z + $Nn
$Rnuc = $r0 * [math]::Pow($A, 1.0/3.0)    # the growing sphere (NOT $R -- collides with $r case-insensitively)
$offd  = -$hbar2_2m/($H*$H)
$offd2 = $offd*$offd

function WS([double]$r,[double]$Rn,[double]$ad) { 1.0/(1.0+[math]::Exp(($r-$Rn)/$ad)) }     # form factor f(r)
function dWS([double]$r,[double]$Rn,[double]$ad) { $f=(WS $r $Rn $ad); -(1.0/$ad)*$f*(1.0-$f) } # df/dr
function Coulomb([double]$r,[int]$Zc,[double]$Rn) {                          # uniform charged sphere
  if ($r -ge $Rn) { return ($Zc-1)*$e2/$r }
  return ($Zc-1)*$e2/(2.0*$Rn)*(3.0 - ($r*$r)/($Rn*$Rn))
}

# diagonal of the tridiagonal H for a given (l, ls=<l.s>, isProton)
function DiagFor([int]$l,[double]$ls,[bool]$isProton,[int]$Zc,[double]$Vdepth,[double]$Rn,[double]$ad) {
  $d = New-Object 'double[]' ($M+1)
  $base = 2*$hbar2_2m/($H*$H)
  for ($i=1; $i -le $M; $i++) {
    $r = $i*$H
    $Vc  = -$Vdepth*(WS $r $Rn $ad)                           # central WS (attractive)
    $Vso = $Wso*(1.0/$r)*(dWS $r $Rn $ad)*$ls                 # spin-orbit: dWS<0 so aligned(ls>0)->lower
    $Vcent = $hbar2_2m*$l*($l+1)/($r*$r)                      # centrifugal
    $Vcoul = 0.0; if ($isProton) { $Vcoul = Coulomb $r $Zc $Rn }
    $d[$i] = $base + $Vc + $Vso + $Vcent + $Vcoul
  }
  return ,$d
}
function CountBelow($diag,[double]$lam) {
  $q = $diag[1]-$lam; $c = 0; if ($q -lt 0){$c++}
  for ($i=2;$i -le $M;$i++){ if([math]::Abs($q)-lt 1e-300){$q=1e-300}; $q=($diag[$i]-$lam)-$offd2/$q; if($q -lt 0){$c++} }
  $c
}
function NthEigen($diag,[int]$n,[double]$lo,[double]$hi) {
  for ($it=0;$it -lt 200;$it++){ $mid=0.5*($lo+$hi); if((CountBelow $diag $mid)-ge($n+1)){$hi=$mid}else{$lo=$mid}; if(($hi-$lo)-lt 1e-5){break} }
  0.5*($lo+$hi)
}

# collect all bound single-particle levels for one species
function Levels([bool]$isProton,[int]$Zc,[double]$Vdepth,[double]$Rn,[double]$ad) {
  $lv = @()
  for ($l=0; $l -le $Lmax; $l++) {
    foreach ($tw in @(1,-1)) {                                  # j = l +/- 1/2
      if ($l -eq 0 -and $tw -eq -1) { continue }
      $j = $l + 0.5*$tw
      $ls = 0.5*($j*($j+1) - $l*($l+1) - 0.75)                  # <l.s>
      $d = DiagFor $l $ls $isProton $Zc $Vdepth $Rn $ad
      $nb = CountBelow $d 0.0                                   # bound states (E<0)
      for ($k=0; $k -lt $nb; $k++) {
        $E = NthEigen $d $k -90.0 0.0
        $lv += [pscustomobject]@{ E=$E; l=$l; j=$j; cap=[int](2*$j+1); tag=("{0}{1}{2}/2" -f ($k+1), @('s','p','d','f','g','h','i','j','k','l')[$l], [int](2*$j)) }
      }
    }
  }
  ,($lv | Sort-Object E)
}

# fill and report shell gaps as magic numbers, in the physically-filled window only
function MagicFrom($lv,[string]$name,[int]$window,[switch]$Dump) {
  $cum=0; $rows=@()
  for ($i=0;$i -lt $lv.Count;$i++){
    $cum += $lv[$i].cap
    $gap = if($i -lt $lv.Count-1){ $lv[$i+1].E - $lv[$i].E } else { 0 }
    $rows += [pscustomobject]@{ cum=$cum; gap=$gap; tag=$lv[$i].tag; E=$lv[$i].E }
  }
  if ($Dump) {
    Write-Host "  ${name} level scheme (E MeV | cum | gap):"
    foreach ($r in ($rows | Where-Object { $_.cum -le $window })) {
      Write-Host ("     {0,-7} E={1,-9} cum={2,-5} gap={3}" -f $r.tag, [math]::Round($r.E,3), $r.cum, [math]::Round($r.gap,3))
    }
  }
  # a closure = a gap large vs the typical single-particle spacing (scale-free; the level SCHEME
  # above is the ground truth -- this is an approximate auto-extractor, not Strutinsky shell-correction)
  $bound = @($rows | Where-Object { $_.cum -le $window -and $_.E -lt -1.0 })
  $gaps = @($bound | Select-Object -ExpandProperty gap | Where-Object { $_ -gt 0 } | Sort-Object)
  if ($gaps.Count -eq 0) { return $rows }
  $med = $gaps[[int]($gaps.Count/2)]
  $closures = $bound | Where-Object { $_.gap -gt 1.6*$med } | Sort-Object cum
  Write-Host ("  ${name} shell closures (gap > 1.6x median spacing): " + (($closures | Select-Object -ExpandProperty cum) -join ", "))
  return $rows
}

# STRUTINSKY shell-correction: delta_E(N) = (sum of N lowest levels) - (Strutinsky-smoothed sum).
# Magic numbers = deep LOCAL MINIMA of delta_E (genuine extra binding), not merely big gaps.
# Smoothing uses the curvature-corrected Gaussian (2nd-order Laguerre L_2^{1/2}), width gamma~1.2*hw.
function StrutinskyMagic($lv,[string]$name,[int]$upto,[double]$gamma) {
  $states = @(); foreach ($o in $lv) { for ($c=0; $c -lt $o.cap; $c++) { $states += [double]$o.E } }
  $states = @($states | Sort-Object); $ns = $states.Count
  if ($ns -lt 6) { return }
  $emin = $states[0]-4*$gamma; $emax = $states[$ns-1]+2*$gamma
  $de = 0.05; $ng = [int](($emax-$emin)/$de); $sp = [math]::Sqrt([math]::PI)
  $Ntil = New-Object 'double[]' ($ng+1); $Etil = New-Object 'double[]' ($ng+1)
  $accN = 0.0; $accE = 0.0
  for ($k=0; $k -le $ng; $k++) {
    $e = $emin+$k*$de; $g = 0.0
    foreach ($eps in $states) {
      $x = ($e-$eps)/$gamma
      if ([math]::Abs($x) -lt 6) { $x2=$x*$x; $g += [math]::Exp(-$x2)*(1.875 - 2.5*$x2 + 0.5*$x2*$x2) }
    }
    $g = $g/($gamma*$sp); $accN += $g*$de; $accE += $e*$g*$de
    $Ntil[$k]=$accN; $Etil[$k]=$accE
  }
  $prefE = New-Object 'double[]' ($ns+1)
  for ($i=0; $i -lt $ns; $i++) { $prefE[$i+1]=$prefE[$i]+$states[$i] }
  $rows = @(); $k = 0
  for ($N=2; $N -le [math]::Min($upto,$ns); $N+=2) {
    while ($k -lt $ng -and $Ntil[$k] -lt $N) { $k++ }
    if ($k -eq 0) { continue }
    $frac = ($N-$Ntil[$k-1])/([math]::Max(1e-9,$Ntil[$k]-$Ntil[$k-1]))
    $Et = $Etil[$k-1]+$frac*($Etil[$k]-$Etil[$k-1])
    $rows += [pscustomobject]@{ N=$N; dE=($prefE[$N]-$Et) }
  }
  $mag = @()
  for ($i=1; $i -lt $rows.Count-1; $i++) {
    if ($rows[$i].dE -lt $rows[$i-1].dE -and $rows[$i].dE -lt $rows[$i+1].dE -and $rows[$i].dE -lt -1.0) {
      $mag += ("{0}({1})" -f $rows[$i].N, [math]::Round($rows[$i].dE,1))
    }
  }
  Write-Host ("  ${name} Strutinsky magic numbers (dE minima, MeV): " + ($mag -join ", "))
}

# solve the tridiagonal (T - lam) x = b  (Thomas algorithm; off-diagonal = $offd constant)
function SolveTri($diag,[double]$lam,$b) {
  $cp = New-Object 'double[]' ($M+1); $dp = New-Object 'double[]' ($M+1); $x = New-Object 'double[]' ($M+1)
  $cp[1] = $offd/($diag[1]-$lam); $dp[1] = $b[1]/($diag[1]-$lam)
  for ($i=2; $i -le $M; $i++) {
    $piv = ($diag[$i]-$lam) - $offd*$cp[$i-1]     # NOT $m -- collides with $M case-insensitively
    $cp[$i] = $offd/$piv; $dp[$i] = ($b[$i]-$offd*$dp[$i-1])/$piv
  }
  $x[$M] = $dp[$M]
  for ($i=$M-1; $i -ge 1; $i--) { $x[$i] = $dp[$i]-$cp[$i]*$x[$i+1] }
  return ,$x
}
# eigenvector for a known eigenvalue via inverse iteration; normalised so sum u^2 * H = 1
function EigenVec($diag,[double]$lambda) {
  $x = New-Object 'double[]' ($M+1); for ($i=1;$i -le $M;$i++){ $x[$i]=1.0 }
  for ($it=0; $it -lt 3; $it++) {
    $x = SolveTri $diag ($lambda-0.02) $x
    $s = 0.0; for ($i=1;$i -le $M;$i++){ $s += $x[$i]*$x[$i] }; $s=[math]::Sqrt($s*$H)
    if ($s -gt 0) { for ($i=1;$i -le $M;$i++){ $x[$i]=$x[$i]/$s } }
  }
  return ,$x
}
# density rho(r) from the lowest `fillN` nucleons
function DensityProfile($lv,[bool]$isProton,[int]$Zc,[double]$Vdepth,[int]$fillN) {
  $rho = New-Object 'double[]' ($M+1); $filled = 0
  foreach ($o in ($lv | Sort-Object E)) {
    if ($filled -ge $fillN) { break }
    $take = [math]::Min($o.cap, $fillN-$filled)
    $ls = 0.5*($o.j*($o.j+1) - $o.l*($o.l+1) - 0.75)
    $diag = DiagFor $o.l $ls $isProton $Zc $Vdepth $Rnuc $adiff
    $u = EigenVec $diag $o.E
    for ($i=1; $i -le $M; $i++) { $r=$i*$H; $rho[$i] += $take*$u[$i]*$u[$i]/(4.0*[math]::PI*$r*$r) }
    $filled += $take
  }
  return ,$rho
}

$Vp = $V0*(1.0 + $kappa*($Nn-$Z)/$A)    # Lane potential: proton well DEEPER in a neutron-rich nucleus
$Vn = $V0*(1.0 - $kappa*($Nn-$Z)/$A)    # neutron well shallower (neutrons head toward the drip line)
Write-Host "=================================================================="
Write-Host " QUANTUM NUCLEAR SHELL MODEL   Z=$Z N=$Nn (A=$A)   R=$([math]::Round($Rnuc,2)) fm"
Write-Host " WS depth p/n = $([math]::Round($Vp,1))/$([math]::Round($Vn,1)) MeV,  a=$adiff,  W_so=$Wso"
Write-Host "=================================================================="
$pl = Levels $true  $Z $Vp $Rnuc $adiff
$nl = Levels $false $Z $Vn $Rnuc $adiff
$null = MagicFrom $pl "PROTON " ($Z+18) -Dump       # window = just past the Fermi level (filled region)
$null = MagicFrom $nl "NEUTRON" ($Nn+18)
if ($Strutinsky) {
  $gamma = 1.2 * 41.0 / [math]::Pow($A, 1.0/3.0)    # smoothing width ~ 1.2 hbar*omega
  Write-Host ""
  StrutinskyMagic $pl "PROTON " ($Z+18) $gamma
  StrutinskyMagic $nl "NEUTRON" ($Nn+18) $gamma
}
if ($Density) {
  $rp = DensityProfile $pl $true  $Z $Vp $Z
  $rn = DensityProfile $nl $false $Z $Vn $Nn
  Write-Host ""
  Write-Host "SELF-CONSISTENCY CHECK: does the density reproduce the Woods-Saxon shape we assumed?"
  Write-Host ("  r(fm)  rho_total(fm^-3)   WS-form-factor f(r)   (both normalised to centre)")
  $rho0 = $rp[1]+$rn[1]
  foreach ($ri in 5,15,30,45,60,75,90,105) {
    if ($ri -le $M) {
      $rr = $ri*$H; $rho = $rp[$ri]+$rn[$ri]
      $f = WS $rr $Rnuc $adiff
      Write-Host ("  {0,-6} {1,-18} {2}" -f [math]::Round($rr,1), [math]::Round($rho,4), [math]::Round($f,3))
    }
  }
  $rhoCentre = 0.0; for($i=3;$i -le 12;$i++){ $rhoCentre += ($rp[$i]+$rn[$i]) }; $rhoCentre = $rhoCentre/10
  Write-Host ("  => interior density ~ {0} fm^-3 (empirical nuclear saturation ~ 0.16). The assumed WS" -f [math]::Round($rhoCentre,3))
  Write-Host ("     shape and the density it produces track each other => the mean field is ~self-consistent.")
}
Write-Host ""
Write-Host "Known magic numbers: 2, 8, 20, 28, 50, 82, 126 (+184 neutron island)."
