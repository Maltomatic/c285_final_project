@echo off
setlocal enabledelayedexpansion
set "src_dir=."
set "date_str=%date:~4,2%_%date:~7,2%_%time:~0,2%_%time:~3,2%"

@REM Move all "baseline-**" files to "saved_progress/baseline-{$time(MM_DD_HH_MM)}" folder
@echo "Saving baseline progress to saved_progress/baseline-%date_str% folder..."
set "dest_dir=saved_progress/baseline-%date_str%"
if not exist "%dest_dir%" (
    mkdir "%dest_dir%"
)

@REM Move all "fault-**" files to "saved_progress/fault-{$time(MM_DD_HH_MM)}" folder
@echo "Saving fault progress to saved_progress/fault-%date_str% folder..."
set "dest_dir=saved_progress/fault-%date_str%"
if not exist "%dest_dir%" (
    mkdir "%dest_dir%"
)
for %%f in (%src_dir%\fault-*) do (
    move "%%f" "%dest_dir%\"
)

@REM Move all "normal-**" files to "saved_progress/normal-{$time(MM_DD_HH_MM)}" folder
@echo "Saving normal progress to saved_progress/normal-%date_str% folder..."
set "dest_dir=saved_progress/normal-%date_str%"
if not exist "%dest_dir%" (
    mkdir "%dest_dir%"
)
for %%f in (%src_dir%\normal-*) do (
    move "%%f" "%dest_dir%\"
)