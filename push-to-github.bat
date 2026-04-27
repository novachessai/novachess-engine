@echo off
REM Initial push of Nova engine docs to github.com/novachessai/novachess-engine
REM Run this once. Re-running after first commit is harmless but unnecessary.

setlocal
cd /d "%~dp0"

echo === Initializing git repo ===
if exist ".git" (
    echo .git already exists, skipping init/remote setup
) else (
    git init
    if errorlevel 1 goto :err
    git remote add origin https://github.com/novachessai/novachess-engine.git
    if errorlevel 1 goto :err
)

echo.
echo === Fetching remote ===
git fetch origin
if errorlevel 1 goto :err

echo.
echo === Attaching to remote main (no file changes) ===
git reset origin/main
if errorlevel 1 goto :err

echo.
echo === Diff: your .gitignore vs GitHub's stub ===
echo (review this. anything you want to keep from theirs?)
echo ----------------------------------------------------
git --no-pager diff .gitignore
echo ----------------------------------------------------
echo.

echo === Files to be added ===
git status
echo.

set /p CONFIRM="Proceed with staging all files? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Aborted by user.
    goto :end
)

echo.
echo === Staging ===
git add .
if errorlevel 1 goto :err

echo.
echo === Commit message ===
set /p MSG="Enter commit message: "
if "%MSG%"=="" (
    echo Empty message, aborting.
    goto :end
)

git commit -m "%MSG%"
if errorlevel 1 goto :err

echo.
echo === Pushing to origin/main ===
git push origin main
if errorlevel 1 goto :err

echo.
echo === Done ===
echo Repo live at https://github.com/novachessai/novachess-engine
goto :end

:err
echo.
echo *** Error encountered. Aborting. ***

:end
echo.
pause
endlocal
