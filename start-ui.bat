@echo off
setlocal

cd /d "%~dp0"
if errorlevel 1 goto :failure

if not exist "scripts\train_ui_qt.py" (
    echo Error: scripts\train_ui_qt.py is missing.
    goto :failure
)

if not defined VENV_DIR set "VENV_DIR=%~dp0venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Error: virtual environment not found. Run install.bat first.
    goto :failure
)

echo Using Python "%PYTHON_EXE%"
"%PYTHON_EXE%" --version
if errorlevel 1 goto :failure

"%PYTHON_EXE%" "%~dp0scripts\util\version_check.py" 3.10 3.14
if errorlevel 1 goto :failure

echo Starting PySide6 UI...
if defined PROFILE (
    "%PYTHON_EXE%" -X utf8 -m scalene --off --cpu --gpu --profile-all --no-browser scripts\train_ui_qt.py
) else (
    "%PYTHON_EXE%" -X utf8 scripts\train_ui_qt.py
)
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo Error: UI script exited with code %EXIT_CODE%.
goto :finish

:failure
set "EXIT_CODE=1"

:finish
if not defined ONETRAINER_NO_PAUSE pause
exit /b %EXIT_CODE%
