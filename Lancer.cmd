@echo off
rem Double-clic : lance l'application d'annotation des fissures.
rem Le premier lancement installe l'environnement Python (quelques minutes).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanceurs\lancer.ps1" %*
if errorlevel 1 pause
