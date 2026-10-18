<#
    Reentraine le detecteur de fissures sur une mosaique.

        .\lanceurs\entrainer_detecteur.ps1                  # fenetre : mosaique, pas, options
        .\lanceurs\entrainer_detecteur.ps1 -Image "D:\MEB\mosaique.tif" -Pas 2000   # sans fenetre
        .\lanceurs\entrainer_detecteur.ps1 -Choisir         # sans fenetre, mosaique au choix

    Il faut auparavant, dans l'application, une dizaine de tuiles VALIDEES
    dont TOUTES les fissures sont tracees (fines comprises) : une fissure
    oubliee apprend au reseau que ce n'est pas une fissure.

    Le reseau repart du detecteur en service, ce qui demande peu de tuiles et
    environ un quart d'heure. Un cinquieme des tuiles validees est mis de cote,
    jamais appris : le nouveau detecteur ne remplace l'ancien que s'il fait
    mieux sur ces tuiles-la, et l'ancien est garde sous un nom date. Relancer
    ensuite l'application pour qu'elle le charge.

    Le deroulement est aussi ecrit dans annotation_app\data\entrainements\.
#>
param(
    [string] $Image,
    [switch] $Choisir,
    [int] $Pas = 800,
    [switch] $DeZero,
    [switch] $SansInstaller,
    [string] $Database
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory

. (Join-Path $PSScriptRoot "lanceur_commun.ps1")
$python = Get-ProjectPython $projectDirectory

# Sans option (double-clic, tache VS Code) : la fenetre, ou l'on choisit la
# mosaique, le nombre de pas et les options, et ou l'on suit l'entrainement.
if ($PSBoundParameters.Count -eq 0) {
    & $python -m annotation_app.bureau --entrainer
    exit $LASTEXITCODE
}

$Image = Resolve-Mosaic $projectDirectory $Image -Choisir:$Choisir `
    -Title "Choisir la mosaique sur laquelle reentrainer le detecteur"
if (-not $Image) { Write-Host "Aucune image choisie, rien n'est lance."; return }

$logDirectory = Join-Path $projectDirectory "annotation_app\data\entrainements"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$log = Join-Path $logDirectory ("detecteur-{0:yyyyMMdd-HHmm}.log" -f (Get-Date))

Write-Host ""
Write-Host "Reentrainement du detecteur de fissures" -ForegroundColor Cyan
Write-Host "Image           : $Image"
Write-Host "Pas             : $Pas (environ $([Math]::Max(1, [Math]::Round($Pas / 40))) min sur ce processeur)"
Write-Host "Depart          : $(if ($DeZero) { 'reseau vierge' } else { 'detecteur en service' })"
Write-Host "Installation    : $(if ($SansInstaller) { 'non (candidat seulement)' } else { 'seulement s''il fait mieux que l''actuel' })"
Write-Host "Journal         : $log"
Write-Host "Interrompre     : Ctrl+C (le detecteur en service n'est alors pas touche)"
Write-Host ""

$arguments = @("-u", ".\outils\train_crack_unet.py", "--image", $Image, "--steps", "$Pas")
if (-not $SansInstaller) { $arguments += "--installer" }
if ($DeZero) { $arguments += "--de-zero" }
if ($Database) { $arguments += @("--database", $Database) }

# Les messages d'erreur de Python vont directement a l'ecran : les rediriger
# les ferait passer, sous Windows PowerShell, pour des erreurs du lanceur.
$ErrorActionPreference = "Continue"
& $python @arguments | Tee-Object -FilePath $log
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "Termine. Si un nouveau detecteur a ete installe, relancer l'application pour qu'elle le charge." -ForegroundColor Green
}
else {
    Write-Host "L'entrainement s'est arrete sur une erreur (voir ci-dessus)." -ForegroundColor Red
}
exit $code
