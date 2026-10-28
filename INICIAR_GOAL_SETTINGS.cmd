@echo off
setlocal
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\local_launcher.py %*
  goto finished
)
for %%V in (3.14 3.13 3.12) do (
  py -%%V -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    py -%%V scripts\local_launcher.py %*
    goto finished
  )
)
echo No se encontro Python compatible. Instala Python 3.14 de 64 bits desde python.org.
echo Despues vuelve a abrir este archivo. No necesitas permisos de administrador.
pause
exit /b 1
:finished
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
