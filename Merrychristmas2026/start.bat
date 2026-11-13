@echo off
rem Abre a arvore de Natal usando o Python instalado no Windows (sem venv).
cd /d "%~dp0"

set "PY=python"
where python >nul 2>nul || set "PY=py"

%PY% -c "import numpy, pygame" >nul 2>nul
if errorlevel 1 (
    echo Instalando numpy e pygame-ce no Python local...
    %PY% -m pip install --user numpy pygame-ce
)

%PY% -B askAI\merry_christmas.py
