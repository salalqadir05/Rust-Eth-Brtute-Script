@echo off
echo Setting up CUDA environment...

:: Set CUDA paths
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set CUDA_PATH_V11_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

:: Add CUDA paths to PATH
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

:: Verify CUDA installation
echo Checking CUDA installation...
where nvcc
if %ERRORLEVEL% NEQ 0 (
    echo Error: CUDA compiler (nvcc) not found in PATH
    echo Please verify CUDA installation at: %CUDA_PATH%
    pause
    exit /b 1
)

:: Print CUDA version
nvcc --version

:: Run the program
echo.
echo Running program with CUDA environment...
cargo run %*

pause 