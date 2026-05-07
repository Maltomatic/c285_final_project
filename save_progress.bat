@echo off
setlocal enabledelayedexpansion
set "src_dir=."
set "hh=%time:~0,2%"
set "hh=%hh: =0%"
set "date_str=%date:~4,2%_%date:~7,2%_%hh%_%time:~3,2%"

@REM call :move_group "baseline" "baseline-*"
@REM call :move_group "fault" "fault-*"
@REM call :move_group "normal" "normal-*"
@REM call :move_group "normal_pure" "normal_pure-*"
@REM call :move_group "fault_pure" "fault_pure-*"
@REM call :move_group "rl_7_ft_3_pure_ft" "rl_7_ft_3_pure_ft-*"
goto :eof

:move_group
set "group=%~1"
set "pattern=%~2"

if not exist "%src_dir%\%pattern%" (
    echo No %group% files found. Skipping.
    exit /b
)

set "dest_dir=saved_progress\%group%-%date_str%"
echo Saving %group% progress to %dest_dir% folder...
if not exist "%dest_dir%" (
    mkdir "%dest_dir%"
)

set "moved_any="
for %%f in (%src_dir%\%pattern%) do (
    move /Y "%%f" "%dest_dir%\" >nul
    if not errorlevel 1 set "moved_any=1"
)

if not defined moved_any (
    rmdir "%dest_dir%" 2>nul
)

exit /b