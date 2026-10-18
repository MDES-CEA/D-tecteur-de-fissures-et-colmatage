<#
    Fonctions communes aux lanceurs (application, reentrainement du
    detecteur). A charger par dot-source, depuis le dossier lanceurs\ :

        $projectDirectory = Split-Path -Parent $PSScriptRoot
        . (Join-Path $PSScriptRoot "lanceur_commun.ps1")
        $python = Get-ProjectPython $projectDirectory
        $image  = Resolve-Mosaic $projectDirectory $Image -Choisir:$Choisir
#>


function Test-PythonEnvironment([string] $python) {
    # Un `.venv` copie depuis un autre poste existe mais ne marche pas : ses
    # chemins internes sont absolus. On essaie donc vraiment d'importer.
    $ErrorActionPreference = "Continue"
    if (-not (Test-Path -LiteralPath $python)) { return $false }
    & $python -c "import PySide6.QtWidgets, numpy, scipy, skimage, sklearn, PIL, joblib, matplotlib, tifffile" 2>$null
    return $LASTEXITCODE -eq 0
}


function Find-BasePython {
    # Python 64 bits, 3.12 ou 3.13 ; 3.12 de preference, la version du poste
    # d'origine. Les versions imposees (numpy 2.5…) n'existent pas pour 3.10
    # et 3.11 : l'installation y echouerait a mi-chemin. Sans guillemets dans
    # le code passe a Python : Windows PowerShell 5.1 les mutile en route.
    $ErrorActionPreference = "Continue"
    $probe = "import sys; print(sys.executable if (3,12) <= sys.version_info[:2] <= (3,13) and sys.maxsize > 2**32 else 0)"
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in "3.12", "3.13") { $candidates += , @("py", "-$version") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @("python") }
    foreach ($candidate in $candidates) {
        $arguments = @($candidate | Select-Object -Skip 1) + @("-c", $probe)
        $output = & $candidate[0] @arguments 2>$null
        if ($LASTEXITCODE -eq 0 -and $output -and "$($output | Select-Object -Last 1)".Trim() -ne "0") {
            return "$($output | Select-Object -Last 1)".Trim()
        }
    }
    return $null
}


function Get-ProjectPython([string] $ProjectDirectory) {
    <#
        Le Python du projet, installe au besoin.

        Au premier lancement sur un poste, l'environnement est cree dans
        `.venv`, dans le dossier du projet, avec les versions exactes du poste
        d'origine. Une empreinte des deux fichiers de dependances evite toute
        reinstallation ensuite ; les modifier declenche une mise a jour.
    #>
    $ErrorActionPreference = "Continue"
    $requirements = Join-Path $ProjectDirectory "requirements_annotation_app.txt"
    $constraints = Join-Path $ProjectDirectory "constraints_annotation_app.txt"
    $localEnvironment = Join-Path $ProjectDirectory ".venv"
    $stampFile = Join-Path $localEnvironment "fissures-installation.txt"
    $stamp = (Get-FileHash -LiteralPath $requirements).Hash + (Get-FileHash -LiteralPath $constraints).Hash
    $venvPython = Join-Path $localEnvironment "Scripts\python.exe"

    $upToDate = (Test-Path -LiteralPath $stampFile) -and
        ((Get-Content -LiteralPath $stampFile -Raw).Trim() -eq $stamp)
    if ($upToDate -and (Test-PythonEnvironment $venvPython)) { return $venvPython }

    # Sur le poste d'origine, l'environnement historique est un cran au-dessus.
    # On le reprend s'il fonctionne, plutot que d'en installer un second.
    $legacyPython = Join-Path (Split-Path -Parent $ProjectDirectory) ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $localEnvironment) -and (Test-PythonEnvironment $legacyPython)) {
        return $legacyPython
    }

    if (-not (Test-PythonEnvironment $venvPython) -and (Test-Path -LiteralPath $localEnvironment)) {
        # Environnement casse (copie d'un autre poste, installation coupee) :
        # on repart de zero plutot que de rapiecer.
        Write-Host "Environnement .venv inutilisable sur ce poste : il est recree."
        Remove-Item -LiteralPath $localEnvironment -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $basePython = Find-BasePython
        if (-not $basePython) {
            throw ("Aucun Python 64 bits 3.12 ou 3.13 sur ce poste.`n" +
                   "Installer Python 3.12 (python.org, ou le catalogue logiciel du poste), puis relancer.")
        }
        Write-Host ""
        Write-Host "Premiere utilisation sur ce poste : installation de l'environnement Python." -ForegroundColor Cyan
        Write-Host "Quelques minutes, environ 1 Go, une seule fois. Python utilise : $basePython"
        Write-Host ""
        # `Out-Host` : sans lui, la sortie se melerait a la valeur rendue.
        & $basePython -m venv $localEnvironment | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Creation de l'environnement impossible dans $localEnvironment" }
    }
    & $venvPython -m pip install --disable-pip-version-check `
        -r $requirements -c $constraints `
        --extra-index-url https://download.pytorch.org/whl/cpu | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw ("Installation des dependances impossible (voir les messages de pip ci-dessus).`n" +
               "Derriere un proxy, definir HTTPS_PROXY avant de relancer, par exemple :`n" +
               "  `$env:HTTPS_PROXY = 'http://proxy.exemple:8080'")
    }
    Set-Content -LiteralPath $stampFile -Value $stamp -Encoding ASCII
    return $venvPython
}


function Read-RecordedImage([string] $record) {
    if (-not (Test-Path -LiteralPath $record)) { return $null }
    $recorded = "$(Get-Content -LiteralPath $record -Raw -Encoding UTF8)".Trim()
    if ($recorded -and (Test-Path -LiteralPath $recorded)) { return $recorded }
    return $null
}


function Resolve-Mosaic([string] $ProjectDirectory, [string] $Image, [switch] $Choisir,
                        [string] $Title = "Choisir la mosaique a annoter") {
    <#
        La mosaique a ouvrir : celle passee en argument, sinon la derniere
        ouverte sur ce poste, sinon la mosaique de reference ; une fenetre de
        choix s'il n'y en a aucune, ou avec -Choisir. Rend $null si l'on ferme
        la fenetre sans choisir.

        La derniere image et la reference sont notees dans annotation_app\data\,
        hors du depot : le code ne contient aucun nom d'echantillon.
    #>
    $data = Join-Path $ProjectDirectory "annotation_app\data"
    $lastImage = Read-RecordedImage (Join-Path $data "derniere_image.txt")
    $referenceImage = $env:CRACK_REFERENCE_IMAGE
    if (-not ($referenceImage -and (Test-Path -LiteralPath $referenceImage))) {
        $referenceImage = Read-RecordedImage (Join-Path $data "image_reference.txt")
    }

    if (-not $Image -and -not $Choisir) {
        if ($lastImage) { $Image = $lastImage }
        elseif ($referenceImage) { $Image = $referenceImage }
    }
    if (-not $Image) {
        Write-Host "Choisir la mosaique dans la fenetre qui s'ouvre..."
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = $Title
        $dialog.Filter = "Images TIFF (*.tif;*.tiff)|*.tif;*.tiff|Tous les fichiers (*.*)|*.*"
        $dialog.Multiselect = $false
        foreach ($folder in @(
                $(if ($lastImage) { Split-Path -Parent $lastImage }),
                (Join-Path $ProjectDirectory "images"),
                $(if ($referenceImage) { Split-Path -Parent $referenceImage }),
                [Environment]::GetFolderPath("MyDocuments"))) {
            if ($folder -and (Test-Path -LiteralPath $folder)) { $dialog.InitialDirectory = $folder; break }
        }
        # Une fenetre proprietaire au premier plan : lancee depuis VS Code, la
        # boite de dialogue s'ouvrait sinon derriere l'editeur.
        $owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }
        try {
            if ($dialog.ShowDialog($owner) -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
            $Image = $dialog.FileName
        }
        finally { $owner.Dispose() }
    }
    if (-not (Test-Path -LiteralPath $Image)) { throw "Image introuvable : $Image" }
    return (Resolve-Path -LiteralPath $Image).Path
}
