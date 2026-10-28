@echo off
echo Goal Settings Analytics Studio con IA: usa la configuracion y clave de tu archivo .env local.
echo Las llamadas al proveedor pueden tener costo y mantienen la revision humana.
call "%~dp0INICIAR_GOAL_SETTINGS.cmd" --enable-ai %*
exit /b %ERRORLEVEL%
