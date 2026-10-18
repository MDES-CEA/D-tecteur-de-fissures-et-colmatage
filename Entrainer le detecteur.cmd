@echo off
rem Double-clic : fenetre de reentrainement du detecteur de fissures
rem (mosaique, nombre de pas, options, suivi en direct).
rem Prealable : une dizaine de tuiles validees, TOUTES leurs fissures tracees.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanceurs\entrainer_detecteur.ps1" %*
if errorlevel 1 pause
