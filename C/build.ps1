# Builds accelerated.dll and dykstra.dll (x64) with MSVC. Run from this directory:
#   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
if (-not $vs) { throw "No MSVC installation found (vswhere returned nothing)." }

$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

cmd /c "`"$vcvars`" >nul 2>&1 && cd /d `"$PSScriptRoot`" && cl /nologo /std:c11 /O2 /W4 /fp:precise /LD /DACCELERATED_BUILD_DLL accelerated.c /Fe:accelerated.dll && cl /nologo /std:c11 /O2 /W4 /fp:precise /LD /DDYKSTRA_BUILD_DLL dykstra.c /Fe:dykstra.dll"
if ($LASTEXITCODE -ne 0) { throw "cl failed with exit code $LASTEXITCODE" }
Write-Host "Built accelerated.dll and dykstra.dll in $PSScriptRoot"
