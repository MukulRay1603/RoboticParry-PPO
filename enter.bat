@echo off
REM Windows batch script to run Docker with X11 forwarding

REM Set DISPLAY variable for VcXsrv
set DISPLAY=host.docker.internal:0.0

docker run -it --rm ^
    --name samurai_rl_container ^
    -e DISPLAY=%DISPLAY% ^
    -e LIBGL_ALWAYS_INDIRECT=1 ^
    -v "%cd%\SamuraiProject":/workspace/SamuraiProject ^
    -v "%cd%\samurai_rl":/workspace/samurai_rl ^
    -v "%cd%\outputs":/workspace/outputs ^
    -v "%cd%\models":/workspace/models ^
    samuraixt-samurai-rl ^
    bash