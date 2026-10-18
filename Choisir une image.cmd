@echo off
rem Double-clic : choisit une autre mosaique, puis lance l'application.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0lanceurs\lancer.ps1" -Choisir %*
if errorlevel 1 pause
