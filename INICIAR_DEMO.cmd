@echo off
echo Demo con datos sinteticos y revisiones simuladas. No representa ganancias reales.
call "%~dp0INICIAR_GOAL_SETTINGS.cmd" --demo %*
exit /b %ERRORLEVEL%
