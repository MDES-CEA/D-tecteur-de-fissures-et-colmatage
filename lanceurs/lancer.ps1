<#
    Lance l'application d'annotation des fissures (fenetre de bureau), sur
    n'importe quel poste et depuis n'importe quel dossier. Rien a configurer :

    - au premier lancement, l'environnement Python est cree dans `.venv`, dans
      le dossier du projet, avec les versions exactes du poste d'origine
      (quelques minutes, environ 1 Go) ;
    - l'image ouverte est la derniere utilisee sur ce poste ; la premiere fois,
      une fenetre demande de la choisir (ensuite : Fichier > Ouvrir).

        .\lanceurs\lancer.ps1
        .\lanceurs\lancer.ps1 -Choisir              # changer d'image
        .\lanceurs\lancer.ps1 -Image "D:\MEB\mosaique.tif"

    L'identifiant qui separe les annotations d'une mosaique de celles d'une
    autre est deduit du fichier : une mosaique deja connue retrouve son
    travail, meme deplacee ; une image nouvelle part d'une grille vierge, meme
    si elle porte le nom d'une autre. `-ImageId` force cet identifiant.

    Chaque mosaique a sa base, annotation_app\data\bases\<identifiant>.sqlite3.
    -Database force une autre base, pour toutes les mosaiques ouvertes dans
    cette fenetre : c'est la facon de repartir de zero sur une image deja
    annotee.
#>
param(
    [string] $Image,
    [switch] $Choisir,
    [string] $ImageId,
    [string] $Database
)

$ErrorActionPreference = "Stop"

$projectDirectory = Split-Path -Parent $PSScriptRoot
# Les chemins relatifs passes en argument se comprennent depuis le dossier du
# projet, et non depuis le repertoire courant de l'appelant.
Set-Location -LiteralPath $projectDirectory

. (Join-Path $PSScriptRoot "lanceur_commun.ps1")
$python = Get-ProjectPython $projectDirectory

$arguments = @("-m", "annotation_app.bureau")
if ($Image) {
    if (-not (Test-Path -LiteralPath $Image)) { throw "Image introuvable : $Image" }
    $arguments += @("--image", (Resolve-Path -LiteralPath $Image).Path)
}
if ($Choisir) { $arguments += "--choisir" }
if ($Database) { $arguments += @("--database", $Database) }

Write-Host "Application d'annotation des fissures : la fenetre s'ouvre."
Write-Host "Ce terminal affiche les messages de l'application ; le fermer ferme l'application."

# Valable pour ce lancement seulement, puis remis comme avant.
$previousImageId = $env:CRACK_IMAGE_ID
if ($ImageId) { $env:CRACK_IMAGE_ID = $ImageId }
# Les messages de Python vont directement a l'ecran : les rediriger les ferait
# passer, sous Windows PowerShell, pour des erreurs du lanceur.
$ErrorActionPreference = "Continue"
try {
    & $python @arguments
    $code = $LASTEXITCODE
}
finally {
    $env:CRACK_IMAGE_ID = $previousImageId
}
exit $code
